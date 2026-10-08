# Plan: multi-token prediction with the Gemma 4 assistant

## Goal

Make the decode produce more than one token for each pass of the target
model. A small drafter model proposes a few tokens. The target model checks
them in one batch. The target keeps the longest correct prefix and adds one
token of its own.

Gemma 4 does not put the MTP head in the main checkpoint. Google gives a
separate drafter for each target, the "assistant" model. llama.cpp calls the
architecture `gemma4-assistant` and the mode `draft-mtp`.

## Why

A decode step of the 26B model reads 2390 MB and takes about 72 ms. The step
is memory bound. A batch of four tokens reads the attention, the dense
feed-forward part, and the output head only one time. Only the expert reads
grow with the batch.

llama.cpp on jackal gives this result for the 26B target and its drafter. The
drafter is bf16. Greedy output, 200 tokens:

    prompt           mode         tokens/s   drafts   accepted
    Python function  none           16.64        -          -
    Python function  MTP, n=3       21.73      165   144 (87%)
    Python function  MTP, n=6       17.14      233   160 (69%)
    sky is blue      none           16.35        -          -
    sky is blue      MTP, n=3       16.62      210   128 (61%)
    sky is blue      MTP, n=6       11.23      357   138 (39%)

Code gives 1.31 times with three drafts. Prose gives no gain. Six drafts are
worse than three. Thus the drafter cost and the verify cost decide the gain,
and the acceptance rate is not sufficient alone.

A 4-bit drafter gives more. Over four prompts, q4_0 gives 1.37 times and
mxfp4 gives 1.45 times, with the same text as the plain decode. The steps and
the full table are in [MTP_DRAFTER_QUANT.md](MTP_DRAFTER_QUANT.md).

## The files

The drafters are in the project-local Hugging Face cache of
`../gemma4-12b-qat-pytorch`. The bf16 GGUF copies are in `models/assistants`.
A QAT target needs the QAT drafter.

    target GGUF on this machine          drafter
    gemma-4-26B_q4_0-it (QAT)            gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant
    gemma-4-12b-it-qat-q4_0              gemma-4-12B-it-qat-q4_0-unquantized-assistant
    gemma-4-E4B_q4_0-it (QAT)            gemma-4-E4B-it-qat-q4_0-unquantized-assistant
    unsloth gemma-4-E4B-it UD-Q4_K_XL    gemma-4-E4B-it-assistant
    unsloth gemma-4-E2B-it UD-Q4_K_XL    gemma-4-E2B-it-assistant

The other GGUF files do not get a Gemma drafter. The Qwen3.6 file has no MTP
tensors, because the unsloth quant removes them. The Mistral and Sky-T1 files
have no MTP head.

## The drafter model

The drafter of the 26B has four layers. Three layers are sliding and one layer
is global. The hidden size is 1024. The backbone size is 2816, which is the
hidden size of the target.

    tensor                   shape            note
    pre_projection           1024 x 5632      input is [embed, h]
    q_proj                   4096 x 1024      8192 x 1024 in the global layer
    o_proj                   1024 x 4096      1024 x 8192 in the global layer
    gate, up                 8192 x 1024      dense GELU feed-forward
    down                     1024 x 8192
    embed_tokens             262144 x 1024    tied output head
    post_projection          2816 x 1024      the next h

The attention has a query and no key or value. The sliding layers read the
keys and the values of the last sliding layer of the target. The global layer
reads the last global layer of the target. The drafter writes nothing to the
cache.

The E2B and E4B drafters have a hidden size of 256. They also have a centroid
head: 2048 centroids, and a map from each centroid to 128 tokens. The head
selects the 32 best centroids and computes only 4096 logits. The 26B and 12B
drafters do not have this head.

One draft step:

    x      = target_embed(token) * sqrt(2816)
    u      = pre_projection([x, h])
    u      = four decoder layers (the query only)
    u      = rms_norm(u, norm)
    logits = u @ drafter_embed.T
    h_next = post_projection(u)

The first h is the hidden state of the target after its final norm. That is
the input of the output head of the target. Model.forward already returns it.
Every draft step uses the position of the last accepted token. The rope of the
query uses that position. Only h and the token change from step to step.

## The verify step

The target runs the last accepted token and the k drafts in one batch at the
positions p to p+k. The row i gives the target token after the prefix. Keep
the drafts while they match. At the first difference, keep the target token.
If every draft matches, the last row gives one more token. The last kept row
also gives the h for the next draft.

The cache then holds rows for tokens that the target rejected. Call
KVCache.truncate to remove them. A sliding layer keeps at least the window
before the new block, so a truncate of k rows is always possible.

## The cost

The drafter weights are large for a memory-bound step. In bf16 one draft step
reads about 300 MB of layers and 537 MB of output head. The target reads 2390
MB. Three bf16 drafts thus cost about one full target step. This explains the
small gain of llama.cpp with a bf16 drafter.

    drafter part      bf16      int8      int4
    layers            302 MB    151 MB     85 MB
    output head       537 MB    268 MB    151 MB

With int4, three drafts read about 700 MB, or 30 per cent of a target step.
Use int4. In llama.cpp, the q4_0 drafter keeps the acceptance of bf16: 72
against 73 per cent. int8 gives no better acceptance for 1.9 times the bytes.

The verify batch reads the experts of each token. Four tokens select up to 32
experts in each layer, against 8 for one token. Measure the union of the
experts first. The union decides the verify cost.

## Parts

1. `np_gemma/assistant.py`: the class Assistant. It loads the safetensors or
   the GGUF file of the drafter. It uses the int4 kernels for the layers and
   the output head. It has the method `draft(token, h, pos, cache, k)`.
2. A query-only attention. It reads the rows of the two shared layers of the
   target KVCache. The decode kernel of the fused attention can serve it,
   because the draft is one query row.
3. The centroid head for the E2B and E4B drafters.
4. The method `Model.verify(ids, cache, start)`. It returns the logits of
   every row and the hidden states. It uses the batch path of the forward.
5. `Session.generate_stream` with MTP. The loop: draft, verify, accept,
   truncate. The sampler stays greedy in the first version.
6. The option `--mtp PATH` and `--mtp-n N` for `scripts/gguf_generate.py`,
   `scripts/e4b_generate.py`, and `scripts/serve.py`. The default N is 3.
7. The environment variable NP_GEMMA_MTP. The value 0 turns the drafter off.

## Phases

Each phase ends with a commit and a test.

- Phase 0: measure. Write `scripts/bench_verify.py`. It gives the time of a
  target batch of 1, 2, 4, and 7 tokens after a context of 512. It also gives
  the expert union for each size. If a batch of four costs more than two
  single steps, stop and report.
- Phase 1: the drafter in NumPy float32 for the 26B. Compare it with the
  transformers reference in the venv of `../gemma4-12b-qat-pytorch`. The
  module `gemma4_assistant` is in transformers 5.17.0.
- Phase 2: the verify step and the accept loop with greedy selection. The
  output must equal the output of the plain greedy decode. See the risks.
- Phase 3: the int4 drafter. Use the existing int4 GEMV and the int4 output
  head. Measure the tokens/s against the plain decode and against llama.cpp.
- Phase 4: the 12B and E4B targets. Add the centroid head for E4B and E2B.
- Phase 5 (optional): sampling with a temperature. Use the standard accept
  rule of speculative sampling: accept a draft with the probability
  min(1, p/q). Resample from the rest of the distribution on a rejection.
- Phase 6 (optional): a draft length that changes with the acceptance rate.
  Stop a draft when the top probability falls below a limit.

## Verification

1. `scripts/check_assistant.py`. Feed the same h, token, and shared cache to
   the NumPy drafter and to the transformers drafter. The logits and h_next
   must agree to float32 precision. The top token must agree.
2. `scripts/check_mtp.py`. Run the 26B with and without MTP on the prompts of
   the table above. The token ids must be the same. Print the draft count,
   the accepted count, and the tokens/s.
3. Run the same prompts with `llama-server --spec-type draft-mtp`. The
   acceptance rate of our drafter must be near the rate of llama.cpp.
4. Run `scripts/check_kv_window.py` with a context longer than the window.
   The truncate of rejected rows must not damage a sliding layer.

## Risks

- The batch path and the one-token path of the target use different kernels.
  The batch MoE uses int8 activations and the one-token path uses float. A
  verify row can thus give a different greedy token than a plain decode. Set
  NP_GEMMA_INT4_Q8=0 for the equality test of phase 2. For speed, a GEMV with
  a small number of token lanes and the same sum order is the correct fix.
  Each weight is then read one time for the whole verify batch.
- The fused attention has a kernel for one query row. The verify batch needs
  a causal mask between its own rows. Check the path that the forward
  selects for two to eight rows.
- The target cache keeps an int8 copy for the fused attention. The truncate
  moves only the end. Check that a rewrite of a row updates the int8 copy.
- The acceptance on prose is low. Phase 6 is the answer if phase 3 gives no
  gain on prose.
- The drafter output head in int4 can lower the acceptance rate. Compare
  int4 with bf16 before you make int4 the default.

## Open items

- The llama.cpp numbers use two prompts. Add the prompts of the decode
  benchmark and a long context of 2000 tokens.
- The HF reference feeds the drafter the final-norm hidden state. An older
  llama.cpp comment says "before the final output norm". The current code
  uses the state after the norm. Phase 1 settles this against transformers.

## Results

Phases 0 to 3 are done for the 26B target. The run uses
OPENBLAS_NUM_THREADS=1 and OMP_WAIT_POLICY=ACTIVE. The BLAS pool of the
default setting fights the OpenMP kernels and makes a small batch up to five
times slower.

### Phase 0: the verify cost

`scripts/bench_verify.py` gave these values for the old kernels at a context
of 512 tokens:

    tokens   int8 path   float path
    1          85.8 ms     79.6 ms
    2         404.3 ms    340.8 ms
    4         685.5 ms    409.3 ms

A batch of four cost five to eight decode steps. Two causes:

- The Q6_K output head sent every batch of two or more tokens to the scalar
  kernel. It took about 170 ms for each token.
- The prompt kernels do not suit two to eight tokens. The int8 expert tile
  computes a block of 16 tokens for each expert, and a verify batch gives one
  or two tokens to most experts.

### The small-group kernels

New kernels read each weight block one time for a group of up to 16 tokens.
The steps for one token are the steps of the one-token kernel, in the same
order. Thus each token gets the same bits as a decode step.

    kernel                  replaces for 2 to 16 tokens
    gemma_q6k_avx512        the scalar Q6_K head
    gemma_int4_linear_mt    the four-row int4 GEMV
    gemma_int4_multi4_mt    the query, key, value, gate, and up calls
    gemma_int4_moe_gemv_mt  the expert GEMV; one read of each expert
    gemma_router_mt         the fused router, one call for the group
    gemma_attn_decode_mt    the fused decode attention, one call

`scripts/check_mt.py` runs a group in one pass and the same tokens one at a
time. The hidden states and the logits are the same bit for bit. The test
uses groups of 2, 3, 4, 5, and 8 tokens at a context of 64 and of 300. A group of four
now costs 137 ms, which is 1.9 decode steps. Set NP_GEMMA_MT=0 to use the
prompt kernels.

### A fix to the decode attention

The fused decode attention read every row of the sliding cache. The cache
keeps up to two windows, because it drops old rows in large steps. Thus a
decode step past 1024 tokens also read keys that the window must hide. The
decode now reads only the rows of the window. This changes the output of a
decode step only after 1024 tokens of context.

### Phase 1: the drafter

`scripts/check_assistant.py` compares the NumPy drafter with
Gemma4AssistantForCausalLM of transformers 5.17.0. It uses the same token,
the same target hidden state, and the same shared keys and values:

    step   top token     logits (relative)   h (relative)
    0      563 = 563     3.8e-07             9.4e-07
    1      506 = 506     7.0e-07             6.2e-07
    2      17856 = 17856 5.2e-07             3.7e-07

The int4 drafter selects the same three tokens. Its relative error is 6 to 12
per cent. The transformers code feeds the drafter the target hidden state
after the final norm, as the plan says.

### Phases 2 and 3: the MTP decode

`scripts/check_mtp.py` runs the plain greedy decode and the MTP decode with
the int4 drafter. Every prompt gives the same token ids with and without
MTP. 200 tokens, tokens/s of the decode:

    prompt   plain   n=2     gain   accepted   n=3     gain   accepted
    code     13.47   17.54   1.30   77%        17.99   1.34   73%
    prose    13.19   14.54   1.10   55%        12.68   0.96   45%
    list     13.64   18.17   1.33   82%        18.43   1.35   80%
    math     13.49   18.28   1.35   83%        17.73   1.31   77%
    ALL      13.47   17.25   1.28   75%        16.71   1.24   69%

Two drafts give the best total. The MTP decode of numpy-gemma is now faster
than the plain decode of llama.cpp, which gives 17.01 tokens/s. The MTP
decode of llama.cpp is faster, with 24.69 tokens/s for mxfp4.

### Phase 4: the 12B and E4B targets

Each target uses its own QAT drafter. The E4B drafter has the centroid
head. `scripts/check_assistant.py --e4b` gives a relative difference of
1.4e-06 against transformers, with the same top token at each step.

The E4B model has its own class, so it needed its own small-group path. Its
bfloat16 matrices use the GEMV kernel for a group, because that kernel runs
each token with the steps of a one-token call. Before this change, a verify
batch of E4B used the prompt kernels. MTP was then 0.77 times the plain
decode, and one prompt gave other token ids after 106 tokens.

`scripts/check_mt.py` gives the same bits for a group and for the single
steps on both targets. Every prompt of `scripts/check_mtp.py` gives the same
token ids as the plain decode. Tokens/s of the decode, all four prompts:

    target   plain   n=2     gain   accepted   n=3     gain   accepted
    26B      13.47   17.25   1.28   75%        16.71   1.24   69%
    12B       6.01    9.52   1.58   73%         9.71   1.62   65%
    E4B      12.54   17.85   1.42   61%        17.35   1.38   53%

The 12B gains the most. It is a dense model, so a group of four costs only
1.3 decode steps. The 26B reads the experts of each token, so a group of four
costs 1.9 decode steps.

The unsloth E2B and E4B files use the Q4_K types. The GGUF reader of this
project does not read them, so this plan does not test those two targets.

### Sampling and the server

`mtp_stream` gives the MTP decode to Session and to the server. At each row
of a verify batch, the target picks its token with the sampler of the plain
decode. A draft stays while it is the picked token. Thus the MTP decode emits
the tokens of the plain decode, also with a temperature. With the same seed,
`scripts/check_mtp_session.py` gives the same tokens in two chat turns, for
both attention modes.

This rule accepts fewer drafts than the rule min(1, p/q) of speculative
sampling. At a temperature of 1.0 the test accepts 36 to 55 per cent of the
drafts. The simple rule needs no drafter probabilities and keeps the exact
output.

Use MTP from the command line:

    PYTHONPATH=. python scripts/serve.py --gguf MODEL.gguf --mtp DRAFTER_DIR --mtp-n 2
    PYTHONPATH=. python scripts/gguf_generate.py --gguf MODEL.gguf --mtp DRAFTER_DIR

DRAFTER_DIR is the snapshot directory of the drafter. NP_GEMMA_MTP=0 turns
the drafter off.

### Phase 6: a limit on the draft

With NP_GEMMA_MTP_PMIN=0.5, the drafter stops when its best token has a
probability below 0.5. The 26B gives these values. The plain decode gave
13.76 tokens/s in this run:

    drafts   tokens/s   gain   accepted
    n=3      17.25      1.25   72%
    n=4      16.27      1.18   67%

The limit raises the acceptance, but it gives no gain over two drafts
without the limit. Prose still loses a little (0.94 times). The limit stays
off, and two drafts stay the default.

### The unsloth E4B file against llama.cpp

The target is gemma-4-E4B-it UD-Q4_K_XL (unsloth). The drafter is
gemma-4-E4B-it-assistant: the snapshot for numpy-gemma, and the bf16 GGUF
file of models/assistants for llama.cpp. The four prompts of check_mtp.py
and bench_mtp_llamacpp.py, 200 tokens, greedy. Tokens/s of the decode:

    runtime                 plain   n=2            n=3
    numpy-gemma CPU         15.12   20.68 (1.37)   22.49 (1.49)
    llama.cpp CPU           13.44   19.72 (1.47)   22.50 (1.67)
    numpy-gemma GPU         66.19   76.47 (1.16)   73.32 (1.11)
    llama.cpp GPU           89.90   150.83 (1.68)  150.39 (1.67)

The CPU runs use 18 threads (OMP_WAIT_POLICY=ACTIVE for numpy-gemma, the
build-vnni binary for llama.cpp). The GPU runs use the RTX 5060 Ti, with
the drafter on the GPU (check_mtp.py --gpu-drafter) and llama.cpp
build-cuda with -ngl 99. The GPU also runs the display, so the GPU rates
of numpy-gemma change by about 10 per cent from run to run. Another run
gave 68.16 plain and 85.24 with two drafts (1.25).

numpy-gemma gives the token ids of the plain decode on each prompt. This
is true on the CPU and on the GPU. llama.cpp changed the text of the prose prompt on the
CPU.

On the CPU, MTP of numpy-gemma is as fast as that of llama.cpp. On the GPU,
llama.cpp is 1.8 to 2 times faster. A verify group of 3 tokens of
llama.cpp costs about one decode step. Here it costs about 1.4 steps, and
a round also reads the logits of 3 rows (3.7 ms) and runs 2 draft steps
(1.3 ms).

The changes for the GPU:

- A verify group of the E4B (MT_CPU tokens or less) has the form of the
  decode step (fused 2, no reuse of the buffers). Before, it used the form
  of the prompt, so its values were not those of the steps. Two prompts
  then gave other tokens, and all drafts failed with the drafter of the
  CPU, which reads a host cache that is not current.
- k_mt_gemv_bf16 adds the terms of k_bf16_linear, so each token of a group
  gets the bits of a step.
- kq_rows_i8 does the int8 products of up to 4 tokens and unpacks each
  weight word one time. Each token adds its terms in the order of one
  token. A step uses the kernel of one token (58 registers). A group uses
  one kernel for the counts 1 to 4. A kernel for each count, or a count
  known only at run time, was 1.5 to 2.5 times slower.

A group of 4 tokens (3 drafts) is still twice the time of a group of 3.
Thus two drafts are the best on the GPU.

### Why the GPU was behind, and the fixes

An nsys profile of an MTP round (2 drafts) gave 25.8 ms: 21.5 ms of
kernels and 4.1 ms with no work on the GPU. llama.cpp takes about 15 ms
for a round. The causes and the fixes:

1. The one-token products had few loads in flight. A lane read 4 bytes of
   a superblock. Thus a warp had 128 bytes in flight, and it did the
   scales of each block again for each 8 values. The kernel of 3 tokens
   uses 96 to 108 registers, so fewer warps fit on an SM. A group of 3 took
   1.5 to 2 times a step.

   Now a lane reads 16 bytes of Q4_K and Q5_K (32 values). Q6_K (32 values) and Q8_0 (16 values) use 2-byte loads. The
   test reads 8 matrices, more than the L2 of 32 MB. A Q4_K gate matrix
   went from 0.051 to 0.046 ms for one token, and from 0.077 to 0.050 ms
   for 3.
2. The gate, the up matrix, and the GELU are one kernel (k_kq_glu_i8). The
   lanes 0 to 15 of a warp take a row of the gate. The lanes 16 to 31 take
   the same row of the up matrix. The two halves read the same x. A step
   went from about 12.1 to 11.2 ms.

   Against the f32 model on 255 steps of
   the chat text, the KL (top 64) is 1.07e-3 with the kernel and 9.2e-4
   without it. The two forms differ from each other by 8.9e-4. That is the
   size of any change of the order of the float sums.
3. The logits of 3 rows (3 MB) went to the host. The host applied the soft
   cap and picked the tokens (1.7 ms). Now the GPU applies the soft cap
   (GP_SOFTCAP) and finds the best token of each row (GP_ARGMAX). With a
   greedy pick, mtp_stream copies only the tokens (E4B.argmax_rows). The
   plain decode also takes the soft cap of the GPU. Thus the two decodes
   pick from the same values.
4. The host made the rows of the token tables with NumPy. That took 4
   calls a round, 0.3 ms each. gemma_kq45_rows does it in C, with
   the same bits.
5. The rope tables of all the positions are on the GPU (E4BGPU._rope). A
   step binds the address of its row and copies no table.

llama.cpp, with the same nsys profile, takes about the same kernel time
for a step. Its gate and up take 3.56 ms, o and down 3.07 ms, and the head
1.18 ms. The first MTP decode of a process compiles the programs of the
verify groups. It also records their graphs, and its scratch buffers
grow. Thus check_mtp.py now runs each form once before it measures.

The same four prompts, warm:

    runtime            plain   n=2             n=3
    numpy-gemma GPU    88.09   149.16 (1.69)   146.97 (1.67)
    llama.cpp GPU      89.90   150.83 (1.68)   150.39 (1.67)

Two drafts, for each prompt (numpy-gemma, llama.cpp): code 152.9 and
154.5, prose 130.4 and 137.1, list 152.1 and 146.9, math 165.8 and 168.3.

### The 26B verify group on the CPU

MTP of the 26B on the CPU gave only 1.05 times the plain decode (20.2 tok/s
plain), and llama.cpp gave 1.29. The drafts were accepted as often (76%).
A group of 3 tokens cost about 2 steps:

    part                     1 token   3 tokens   3 tokens now
    experts (KQ_MOE)         13.9 ms   32.5 ms    25.5 ms
    int4 products (dense)    14.3 ms   35.1 ms    18.1 ms
    output head (Q6_K)       12.5 ms   27.5 ms    16.7 ms

- kq_q4x_rows_f (the KQ_Q4X copies, float32 x) decoded the 16 rows of a
  group again for each token. Now it decodes each step one time for up to
  4 tokens.
- The sums of each token were arrays with a count known only at run time,
  so they stayed in memory. A copy of the kernel for each count (1 to 4,
  and 1 to 8 for the Q6_K head) keeps them in registers.

Each token adds its terms in the order of one token, so a group still
gives the bits of the steps (check_mt.py). A group of 3 now costs about
1.5 steps. The four prompts of check_mtp.py (the CPU at 50% of its clock):

    runtime               plain   n=2            n=3
    numpy-gemma CPU       20.03   28.37 (1.42)   28.81 (1.44)
    llama.cpp CPU         17.89   23.07 (1.29)   24.60 (1.38)

### The QAT file of the E4B on the GPU

The target is the QAT file of Unsloth (gemma-4-E4B-it-qat-UD-Q4_K_XL, all
Q4_0). The drafter is google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant.
MTP works on the GPU and gives the token ids of the plain decode. The four
prompts of check_mtp.py with --gpu-drafter, 200 tokens:

    n    plain    MTP      gain   accepted
    1    106.07   139.98   1.32   74%
    2    106.07   134.83   1.27   64%
    3    106.07   119.59   1.13   53%

On the CPU (18 threads, the drafter in int4 on the CPU), the same prompts
gave these values:

    n    plain    MTP      gain   accepted
    2    17.37    28.71    1.65   62%
    3    17.37    27.80    1.60   51%

llama.cpp (bench_mtp_llamacpp.py, -ngl 99 -fa off) with the drafter of
Unsloth (mtp-gemma-4-E4B-it.gguf, Q4_0) gave these values:

    n    plain    MTP      gain   accepted
    1    100.09   149.16   1.49   74%
    2    100.09   172.23   1.72   65%
    3    100.09   178.05   1.78   55%

The BF16 drafter of Unsloth gave 143.8, 164.9, and 165.4. The two runtimes
accept the same share of the drafts. Thus the difference is the cost of a
verify group:

    tokens   1         2          3          4
    time     9.25 ms   12.12 ms   16.23 ms   20.29 ms

A group of 3 costs 1.75 steps. With the UD file of the E4B (K quants), a
group of 3 cost about one step (see above). The K quants use int8 x and
dp4a (kq_rows_i8). The Q4_0 matrices use float32 x. An nsys profile of a
group of 3 gave 11.5 ms of int4 products, against 6.7 ms for one token.

A test of the kernels (10240 x 2560, Q4_0, 14.7 MB) showed the cause. Each
weight reads 4 bytes of x for each token from L1, 21 times the bytes of the
weights for 3 tokens. Each token added about 13 us:

    kernel                       1 token   3 tokens   4 tokens
    read the weights only        41 us
    float32 x (now)              51 us     78 us      90 us
    x in shared memory           48 us     90 us
    2 or 4 rows for each warp    47 us     67 us      64 us
    int8 x and dp4a              48 us     53 us      56 us

Changes:

- k_mt_int4_rows replaces k_mt_gemv_n for a group of 1 to 4 tokens. It uses
  the lanes and the order of the terms of a decode step (int4_part, then
  the scale of the block). The old kernel multiplied each weight by the
  scale first. Now check_mt.py --e4b on the GPU gives the same bits for a
  group and for the steps. Before, it gave a difference of 1e-5. Thus a
  verify group can change a token with the old kernel (k_mt_gemv_n,
  NP_GEMMA_GPU_MT_ROWS=0).
- int4_part makes the float of a nibble with two operations: the bits of
  2^23 + nibble, minus 2^23 + 8. The value is the same.
- ops.quantize_int4 finds the grid of a QAT block (the scale with the least
  error of three). Before, the int4 weights of the drafter (and of any QAT
  safetensors) had 5 to 8 per cent of error. Now they have 0.18 per cent,
  as the Q4_0 drafter of Unsloth. gpu.quantize_q4_0 (the GPU drafter) uses
  it too. The share of accepted drafts did not change (74%, 64%, 53%
  before; 74%, 64%, 53% after).

The 26B on the GPU does not give the same bits for a group and for the
steps (check_mt.py: 5e-4 to 1.6e-3). That was so before these changes.

### Int8 x for the Q4_0 matrices of the E4B and the E2B

The GPU products of the K quants (GP_KQ_LINEAR, kq_rows_i8) now take Q4_0
too (KQ_Q4_0). A lane takes 16 values of a block, from the low or the high
4 bits of its 16 bytes. The first dp4a gives the sum of q x. The second
dp4a gives the sum of x. The result is the sum of (q - 8) x.

E4B.kq_q4 gives the Q4_0 matrices to these products. The GPU step and the verify
groups use them. A prompt pass keeps the int4 records. NP_GEMMA_GPU_Q4_I8=0
keeps float32 x. The data is a view of the int4 blocks, so the GPU holds
one copy.

Three kernels (k_kq_linear_i8, k_kq_multi_i8, k_kq_glu_i8) have a form with
a fixed type (FT = KQ_Q4_0). The general form took 92 to 110 registers,
and the step took 9.98 ms. The fixed form takes 40 to 48 registers. The
step then takes 9.49 ms, and 9.25 ms with float32 x. Also, check_mt.py
--e4b gives the same bits for a group and for the steps.

The time of a verify group of the E4B:

    tokens          1         2          3          4
    float32 x       9.25 ms   12.12 ms   16.23 ms   20.29 ms
    int8 x          9.49 ms   10.79 ms   13.10 ms   13.85 ms

The four prompts of check_mtp.py with --gpu-drafter (E4B):

    n    plain    MTP      gain   accepted   llama.cpp
    1    105.65   154.64   1.46   75%        149.16
    2    105.65   167.59   1.59   65%        172.23
    3    105.65   177.84   1.68   54%        178.05

Each prompt gives the token ids of the plain decode. The E2B with its QAT
drafter (google/gemma-4-E2B-it-qat-q4_0-unquantized-assistant):

    x         plain    n=1      n=2      n=3
    int8      165.70   221.21   222.70   221.50
    float32   179.57   206.37   187.23   175.17

The plain decode of the E2B is 8% slower with int8 x. Its matrices are
small, so the quantization of x is a larger part of a step. MTP is faster
with int8 x, and MTP is on by default for these models.

The KL (top 64) against the float32 release, the three chat prompts:

    model   float32 x                int8 x
    E4B     0.00002, 363/364         0.0005, 363/364
    E2B     0.00003, 363/364         0.0005, 360/364

These values stay far below those of the Q4_0 files of Google (0.020 and
0.025).

### Drafts with sampling

The drafter gives one token x, its best token. The target samples its token
y, and x stays when y is x. Thus x stays with the probability p(x), and a
rejected row emits y from p without x. That is the rule min(1, p/q) of
speculative sampling for a drafter with one token, so the text follows p.

A test measured the first draft of each row of sampled text. It used the
E4B on the CPU, 4 prompts, and 400 rows for each setting:

    settings                  current   sampled draft   in the set
    T 1.0, top_k 64, p 0.95   61.0%     63.5%           83.2%
    T 0.7, p 0.95             71.6%     73.1%           86.8%
    T 1.0                     59.5%     63.5%           100%

"sampled draft" samples x from the drafter (q) and keeps it with min(1,
p/q). It keeps the distribution, but it gains only 1.5 to 4 points. "in the
set" keeps x when the settings allow it. With the temperature alone, all the
tokens are allowed, so every draft stays.

The Sampler now has the option mtp_accept "in_set" (opt-in) and mtp_floor.
A draft also stays when the settings allow it and its probability is at
least mtp_floor times the best probability (Sampler.draft_ok). Without top_k,
top_p, min_p, or mtp_floor, the rule is "exact". The server takes
--mtp-accept and --mtp-floor, and a request can give "mtp_accept" and
"mtp_floor".

The test used the E4B and the drafter on the GPU, with T 1.0, top_k 64, and
top_p 0.95. It ran 4 prompts and 4 seeds, 200 tokens each. The drift compares each emitted
token with the distribution of its own row. "best" is the share of the best
token less its expected share. "logP" is the mean log probability less its
expected value. The noise is about 1.5% and 0.02:

    rule          n=2 tok/s   n=3 tok/s   kept (n=2)   best    logP
    exact         121.29      127.87      55%          -1.3%   -0.019
    in_set        136.43      148.16      77%          -4.1%   -0.094
    in_set 0.5                            65%          +2.1%   +0.053
    in_set 0.3    128.03      136.31      68%          +1.0%   +0.038
    in_set 0.1    129.87      144.69      71%          -1.9%   -0.013

Without a floor, the text takes tokens with a low probability too often.
The top_p set holds tokens of a few per cent, and a draft of such a token
stays. A floor of 0.3 or 0.5 moves the text toward the best token. A floor
of 0.1 shows no drift above the noise, and it keeps most of the gain: 1.13
times the speed of "exact" with 3 drafts.

The cost of sampling was then on the host. A verify group copied the logits
of each row (1 MB), and a pick took 1.6 ms. The GPU now gives the
candidates of each row (gg_topk, see README.md, "Sampling"). With 3 drafts,
"exact" went from 127.4 to 174.2 tok/s (greedy: 175.8), and in_set with a
floor of 0.1 from 142.8 to 214.6.

The quality of in_set with a floor of 0.1. The tests used the E4B on the
GPU, 3 drafts, T 1.0, top_k 64, and top_p 0.95:

- GSM8K, the first 200 problems of the test split, at most 512 tokens:

      rule          accuracy   finished   tok/s   tokens
      greedy        67.5%      133        220.7   411
      exact         68.0%      140        229.4   397
      in_set 0.1    74.0%      151        259.1   378

  Many answers did not finish in 512 tokens. All three rules finished 116
  problems, and on those the accuracy was 98.3%, 97.4%, and 99.1%. Thus the
  rule does not change the answers. Its answers are shorter, so more of them
  finish. A kept draft moved 0.083 of the mass to the draft on average (the
  median was 0).
- Open text (4 prompts, 8 seeds, 300 tokens): the answers are less varied.
  The test used the answer parts (after <channel|> when the model writes a
  draft first). Their distinct-2 across the seeds was 0.815 for "exact" and
  0.743 for in_set. "exact" at lower temperatures gave these values:

      T                1.0     0.9     0.8     0.7
      distinct-2       0.815   0.782   0.736   0.706
      distinct-3       0.894   0.865   0.831   0.800

  in_set (distinct-3 0.835) is thus like a temperature of about 0.8. A kept
  draft moved 0.203 of the mass. The texts had no loops. The repeated
  4-grams (0.021, against 0.002) come from texts in which the model writes a
  draft and then copies it after <channel|> word for word. With "exact"
  the copy changes some words.
- Speed with 3 drafts: "exact" at T 1.0 gave 166.4 tok/s, "exact" at T 0.8
  gave 171.6, and in_set 0.1 at T 1.0 gave 208.9.

Thus in_set 0.1 suits tasks with one right answer (math, code, tools). For
open text it acts as a lower temperature, so "exact" stays the default.

### The 26B on the GPU with the QAT file

The target is the QAT file of Unsloth (gemma-4-26B-A4B-it-qat-UD-Q4_K_XL),
with the hot experts on the GPU (the default budget). The drafter is
google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant on the GPU. Four
prompts, 200 tokens. MTP gained little, although the drafter is good:

    mode            plain   n=1     n=2     n=3     accepted
    greedy          94.0    101.2   103.0   100.8   80/70/62%
    sample exact    88.8    100.9   105.6   106.7   81/70/63%
    in_set 0.1              104.3   112.6   118.5   85/78/74%

A verify group of 2, 3, and 4 tokens took 14.25, 18.37, and 22.36 ms, and a
step 9.59 ms. An nsys profile of a group of 3 tokens gave three large parts:

    int4 products                5.5 ms (3.3 ms for a step)
    separate norms               2.2 ms (211 kernels)
    waits for the cold experts   about 8 ms (the CPU computes them)
 A group
of more tokens takes more cold experts, so the last part does not go away.

Changes:

- Model.argmax_rows and ModelGPU.argmax: the greedy token of each row from
  the GPU (gg_argmax_rows, the first index of the largest value, as
  np.argmax). The E4B uses it too when its head is Q6_K or Q4_0. Before,
  both copied the logits of the rows to the host. The tokens are the same
  as those of np.argmax (120 steps and a group, on each model).
- A small group (an MTP verify group) takes the fused forms of the step
  (add_norm2 and ffn_out, compile_split_group). A group of 2, 3, and 4
  tokens now takes 12.85, 16.61, and 20.41 ms. NP_GEMMA_GPU_GROUP_FUSED=0
  keeps the separate norms. The KL of the test of the chat prompts did not
  change.
- The option NP_GEMMA_GPU_Q4_I8_DENSE=1 is off. It gives int8 x to the
  dense Q4_0 matrices of the step and of the small groups. They then use
  the products of the E4B (program._Q4KQ).

The four prompts after the changes:

    mode            plain   n=1     n=2     n=3
    greedy          94.7    105.8   109.3   107.1
    in_set 0.1              106.2   114.3   120.6
    greedy, int8    91.7    106.6   116.4   118.1
    in_set, int8            110.1   121.1   125.7

With int8 x, a group of 3 tokens takes 15.08 ms. But the KL (top 64) of
the chat prompts goes from 0.0001 to 0.0009, and 357/360 top tokens agree
(359/360 with float32 x). The activations of the 26B have larger outliers,
so int8 costs more than on the E4B. Thus the 26B keeps float32 x.

A verify group of the 26B does not give the bits of the steps (that was so
before these changes). At a context of 300, the logits differ by up to
0.13, and in a group of 4 the best token of one row changed. Thus an MTP
decode of the 26B can give other tokens than the plain decode.

### The 12B on the GPU with the QAT file

The 12B is dense, so it has no cold experts on the CPU. Its verify groups
give the bits of its steps. But with float32 x, a group of 2, 3, and 4
tokens took 25.11, 35.40, and 45.16 ms, and a step took 21.03 ms. That is
the cost of the loads of x that the E4B had. With int8 x (program._Q4KQ, as on
the 26B), they take 21.98, 24.93, and 25.39 ms. The four prompts with the
drafter google/gemma-4-12B-it-qat-q4_0-unquantized-assistant on the GPU:

    mode                plain   n=1     n=2     n=3
    greedy, float32 x   46.0    65.1    61.0    56.6
    greedy, int8 x      45.7    72.5    83.2    91.8
    in_set 0.1, int8            76.1    89.0    99.3

The KL (top 64) of the three chat prompts went from 0.00012 to 0.0004,
and 376/376 top tokens agree in both forms. The image prompt went from
0.00050 to 0.00062. These costs are like those of the E4B. Thus the int8
form is now the default of a dense model, and the 26B keeps float32 x.
NP_GEMMA_GPU_Q4_I8_DENSE=1 or 0 sets the form for both.

### The 12B against llama.cpp

The test used the four prompts of bench_mtp_llamacpp.py and 200 greedy
tokens. The target was the 12B QAT file of Unsloth on the GPU. llama.cpp (build-cuda, -ngl 99) used the
drafter of Unsloth (mtp-gemma-4-12B-it.gguf, Q4_0). llama-bench gave pp512
2250 tok/s and tg128 52.8 tok/s.

The template of Gemma 4 in llama-server turns thinking on when a request
does not give enable_thinking. The text then holds a thought, which the
drafter predicts better (90%, 84%, 74% for 1, 2, 3 drafts). The option
--no-think of bench_mtp_llamacpp.py sends enable_thinking false, as
check_mtp.py does. With thinking off:

    runtime       plain   n=1     n=2     n=3     accepted
    numpy-gemma   45.50   77.26   89.78   102.84  83/75/67%
    llama.cpp     50.00   81.66   99.63   110.21  86/75/65%

The two runtimes accept the same share of drafts. Each prompt of
numpy-gemma gives the tokens of its plain decode. In llama.cpp, the text of
MTP differed from its plain text on 3 of 4 prompts with 1 draft. The plain
decode of llama.cpp is 10% faster (a step of 20.0 ms, against 22.0 ms), and
the gap of MTP is 5% to 10%.

Assistant and GPUDrafter take weights=GGUF: a drafter GGUF (arch
gemma4-assistant, as the MTP files of unsloth) gives the weights. The
snapshot still gives the config. check_mtp.py takes
--drafter-gguf. The weights of the drafter of Unsloth differ from those of
the release by 0.18 per cent, as those of our quantizer. With them, the
drafts and the speed were the same (83/75/67%, 77.26/89.78/102.84 tok/s).
The CPU drafter in float32 and in int4 also accepted 83% with 1 draft.

## What is left

- The 26B verify batch. A group of four costs 1.9 decode steps, and 55 ms of
  its 137 ms reads the experts of the four tokens. The output head (27 ms)
  and the Python of the layer loop are the other large parts.
- An MXFP4 drafter. In llama.cpp it is 6 per cent faster than q4_0. It needs
  a new kernel here.
- The E4B verify group on the GPU. The attention runs one record for each
  query (2.7 ms of a round). A draft step waits for the host two times
  (about 0.3 ms each).
- The verify group of the 26B on the GPU. It does not give the bits of the
  steps (see "The 26B on the GPU with the QAT file").
