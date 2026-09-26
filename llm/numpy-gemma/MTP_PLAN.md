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
