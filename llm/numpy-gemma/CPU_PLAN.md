# Plan: the lessons of the GPU path for the CPU paths

GPU_NOTES.md gives the method and the changes of the GPU path. This plan
applies them to the CPU paths: the decode of the E4B and of the 26B, the
prompt pass, and MTP. The machine is a Xeon W-2295: 18 cores, AVX-512 with
VNNI, 67 GB/s to read memory, 24.8 MB of L3.

## The start: a profile of the decode on the CPU

The step programs ran in C (gemma_run: one OpenMP region for the step, and
a barrier after each record). The values below come from the Python
interpreter (Program.run_py). It runs one record at a time, each in its own
parallel region. Thus the sum is larger than the step. Some records run as
NumPy code in that interpreter (KV_WRITE, for example), so their values are
too high.

    E4B, 200 tokens of context, one step in C: 49.6 ms, 954 records
    record          time      records   each
    INT4_MULTI4     28.1 ms   66        426 us
    INT4_LINEAR     18.6 ms   186       100 us
    ATTN_F32H       6.2 ms    42        148 us
    GELU            3.2 ms    84        38 us
    MUL             3.0 ms    84        36 us
    RMS_NORM        2.8 ms    212       13 us
    ADD             1.7 ms    127       13 us
    MUL_S           0.7 ms    44        17 us

    26B, 200 tokens of context, one step in C: 41.4 ms, 1351 records
    record          time      records   each
    MOE             20.9 ms   30        695 us
    INT4_MULTI4     8.4 ms    30        281 us
    RMS_NORM_MULTI4 5.0 ms    30        167 us
    INT4_LINEAR     4.7 ms    30        158 us
    ATTN_QC         4.6 ms    30        154 us
    KV_WRITE        3.0 ms    30        101 us (NumPy in run_py)
    GELU_MUL_INT4   2.6 ms    30        85 us
    RMS_NORM        2.1 ms    181       12 us
    ROUTER          1.7 ms    30        57 us
    ADD             1.2 ms    90        13 us

What the numbers say:

- The 26B reads about 2.3 GB for each token (8 experts in each of 30
  layers, the dense part, and the head). At 67 GB/s that is 35 ms. The step
  takes 41 ms, about 85% of that limit. The gain must come from fewer
  bytes, or from the parts that do not use the full bandwidth.
- The experts of the 26B read about 800 MB in 20.9 ms: about 38 GB/s, not
  67. This is the largest gap.
- The E4B has many small records: 212 norms, 127 adds, 84 GELU, 84 MUL, 44
  MUL_S. The GPU path fused them. The CPU path of the E4B does not use the
  fused forms yet.

## Phase 0: a profile of each record in C

The GPU had gg_profile (the time of each record on the GPU). The CPU has no
such tool, and the values above come from a different runner. Make
gemma_profile. It has the loop of gp_exec, and thread 0 reads the clock
after the barrier of each record. It gives the time of each record inside the
real parallel region, with the real barriers.

Then, for each record type, compute the bytes that it reads and its rate
in GB/s. A record far below 67 GB/s is a target. A record with few bytes
is a latency cost: it is a target for fusion, not for a faster kernel.

Also time the host part of a decode step (bind_step, the rows of the
embeddings, the logits, the sampler), as for the GPU. The argmax in C and the rows of
the embeddings in one C call already help the CPU path too.

Phase 0 is done: gemma_profile (C) and Program.profile_ops (Python). The
profile of the E4B step in C (55 ms) corrects the table above:

    record          time      records   each
    INT4_MULTI4     30.9 ms   66        469 us
    INT4_LINEAR     19.2 ms   186       103 us
    ATTN_F32H       2.8 ms    42        67 us
    RMS_NORM        0.9 ms    212       4 us
    BF16_LINEAR     0.9 ms    1         881 us
    MUL, GELU, ADD  1.6 ms    295       4 to 7 us

In the real parallel region the small records take about 3 ms, not 11 ms.
The products take 50 ms. Thus phase 1 gives about 2 ms, and the products
are the target.

## Phase 1: the fused forms for the E4B on the CPU (low risk)

The GPU step uses e4b_step_form(fused=2): add_norm2 and gelu_mul. The CPU
step uses the form without fusion. Add ADD_NORM to the C runner.
The Python runner has it already. Keep the same operations in the same
order, so the values stay the same. GELU_MUL_ROWS is in the C runner already.

Expected gain: about 380 fewer records in each step, and fewer passes over
x. From the table, the small records take about 11 ms in run_py. A gain of
5 to 8 ms of the 49.6 ms step is possible. Test: the same bits as the step
without fusion, and scripts/check_program.py.

## Phase 2: the experts of the 26B at the full bandwidth

The experts run at about 38 GB/s. gp_moe_one has two parallel parts: gate
and up for all 8 experts, then down. A single thread works before,
between, and after them. Find the cause with the profile of phase 0:

- the balance of the threads: 8 experts of 1408 rows (gate and up) and
  2816 rows (down), divided over 18 threads;
- the single parts: the sort of 8 indices, and the sum of 8 outputs of
  2816 values on one thread;
- the kernel itself. Find the rate of gemma_moe_gemv_gelu_body on one
  expert. Compare it with INT4_LINEAR on a matrix of the same size.

Possible changes:

- the sum of the outputs in parallel (each thread sums its part of the
  rows);
- one parallel part for gate, up, and down, with a finer split;
- a prefetch of the next expert (see phase 4).
 A rate of 60 GB/s
saves about 7 ms of 41 ms (about 17%).

## Phase 3: the other records of the 26B

- KV_WRITE: one thread copies the row and makes the int16 copy. Time it in
  C (phase 0). If it is more than a few us, split it over the threads.
- ATTN_QC (4.6 ms at 200 tokens): it grows with the context. The GPU
  attention became fast when a block took all the query heads of a key
  head, so it read the keys one time. Check that the CPU kernel reads each
  key row one time for all its query heads.
- The fused forms of the 26B: the 181 RMS_NORM and 90 ADD records can use
  add_norm2, as in phase 1.

## Phase 4: use the bandwidth during the small records

On the GPU, a prefetch of the next matrix did not help: the kernels had no
gaps. The CPU has gaps: during the small records and the attention, the
memory is almost idle. Test a software prefetch of the first part of the
next large matrix (an expert, or the next INT4_MULTI4) into L3 during those
records. L3 is 24.8 MB: 7 experts, or the first part of a dense matrix.

For the 26B, the experts of the next layer are not known before its
router. Test this first with the traces of real answers (the file of the
HotCache test). Find how often the router of layer l + 1 selects an expert
that a simple guess gives, for example the hot set of HotCache. Build the prefetch only
if the guess is often right.

## Phase 5: the prompt pass on the CPU

- The rows of the embeddings: done (one C call, gemma_q6k_rows).
- Quantize x one time for each group of matrices (q, k, v; gate, up), as on
  the GPU. Check whether the int8 path of the CPU (gemma_int4_q8)
  quantizes x again for each matrix of a group.
- Attention of a group: check that the CPU kernel reads the keys one time
  for the query heads that share them. On the GPU, this gave 88 ms to 16
  ms.
- The projection of the layer input of the E4B in bfloat16: check its rate
  against the int8 products.

## Phase 6: MTP on the CPU

The data of HotCache apply: a verify group of 3 tokens selects 16 different
experts in each layer, not 8. The CPU reads each expert one time for the
group (gp_moe_group). Thus the group reads about twice the bytes of a
step, for up to 3 tokens.

Measure the rate of MTP on the CPU with the drafter on
the CPU against the plain decode. A test on the GPU let each draft use the
experts of the first token and its own 4 best experts. It gave 98% of the
best tokens. On the CPU it cuts the bytes of a verify group by about
a third. It changes the result, so it stays a test with numbers, as on the
GPU.

## Order and tests

1. Phase 0 (the profiler). All the other phases need it.
2. Phase 1 and the fused forms of phase 3: low risk, the same bits.
3. Phase 2: the largest gain for the 26B.
4. Phase 4 and phase 6: tests with data first; build only on a clear gain.
5. Phase 5: for long prompts on the CPU.

Each change keeps the values (the same bits, or the same tokens), with the
tests of scripts/check_program.py, scripts/check_mt.py, and the decode
tests. The measure is the rate of the decode (scripts/bench_decode.py) on
an idle machine, if possible. The A/B test of OMP_PROC_BIND is still open,
and it needs an idle machine too.

## The 26B on AVX2: results

The test runs the AVX2 library on the Xeon (NP_GEMMA_ARCH=avx2) with 6
threads, as the Core i5-8500 of the earlier AVX2 work. The file is the
Unsloth 26B. Its token table and head are Q4_0, not Q6_K. The method of
llama-bench (scripts/bench_llama_method.py, tok/s):

    runtime                        pp512   tg64
    numpy-gemma before             61.8    17.0
    numpy-gemma now                64.6    20.4
    llama.cpp (build-avx2, -t 6)   33.1    14.3

A profile of a step (gemma_profile, 383 tokens of context) first gave
49 ms for the step and 16.5 ms for the head. The changes:

1. The head. The Q4_0 token table gets a KQ_Q4X copy (ops._Q4X_HEAD), for
   every token count, with int8 x. A token: 16.5 to 8.2 ms. Four tokens:
   65.9 to 13.4 ms.

   On VNNI the int4 head took float x for a token and int8 x for a group.
   Thus check_mt failed there (logits 0.13). One kernel for every count
   fixes it, and the VNNI head is faster too (8.2 to 6.1 ms).
2. The decode attention (gemma_attn_split_i16_body). An item is a token, a
   key head, and a chunk of keys. It does all the query heads of the key
   head, and a second pass joins the chunks. The chunk length depends only
   on the count of keys, so a group keeps the bits of the steps. On AVX2:
   - the scores take int16 q with vpmaddwd;
   - the values keep blocks of the output in registers, two heads at a
     time;
   - the rows are fetched 3 keys ahead, because each row starts a new page;
   - a weight below exp(-64) is 0, because its products were denormals;
   - a vector exp replaces libm expf.
   ATTN_QC: 9.0 to 5.1 ms for each step.
3. The router. The scalar loop did not vectorize. router_dot has 4 vector
   accumulators, and the step and the group use it. 2.85 to 1.09 ms.
4. The products of a prompt (kq_q4x_rows on AVX2) unpack the codes once for
   4 tokens, not 2. pp512: 61.3 to 64.9 tok/s.

The step now takes 42.6 ms. The products take 34.4 ms of it, at about
50 GB/s, which is the rate of the memory with 6 threads. Thus the plain
decode is near its limit. The rest: the attention 5.1 ms, the router
1.1 ms, and about 2.5 ms for the small records.

MTP on the CPU (scripts/check_mtp.py, CPU drafter, 100 tokens):

    drafts   before   now
    1        1.06x    1.15x
    2        0.99x    1.15x

- The drafter of the 26B has no centroid head. Its head has 262144 rows of
  1024 values. The Assistant now gives its matrices KQ_Q4X copies. A draft
  step: 13.6 to 8.1 ms.
- A verify group of 2 tokens takes 65 ms, 1.5 steps: the second token
  selects other experts. This limits MTP on the CPU.

Later changes to the decode attention on AVX2 (gemma_attn_split_i16_body):

- The queries are quantized once for each token and head.
- A head has one scale (14 bits), so a key group needs one scale for all
  the heads.
- The value pass of more than 2 heads converts the rows of 128 keys once,
  for all the pairs of heads.

At 4301 tokens of context, 6 threads:

    part                   before   now
    attention (a step)     18.5 ms  15.6 ms
    a global layer         1462 us  1091 us
    a layer with a window  432 us   404 us

MTP with 1 and 2 drafts is then 1.16x and 1.17x.

The int8 x of the AVX2 decode makes each small change of a kernel look
large. Two variants of the attention differ by a KL of about 4e-3, as much
as the int8 x itself. Against the float decode of VNNI over 160 steps, the
old kernel gives 3.0e-3 and the new one 3.9e-3. Thus the test of a kernel
is its error against float32 (7e-5 here), and the KL against the float
decode over many steps.

The prompt pass at 2048 tokens: 53.4 tok/s (llama.cpp AVX2: 29.3). The
attention takes 10.7 s of 38.3 s, at about 95 GFLOP/s. That is the next
large part for long prompts. A profiler (perf) needs
kernel.perf_event_paranoid at 1 or lower on this machine.

## The memory of the target CPU

The Core i5-8500 has two channels of DDR4 (DDR4-2666: at most 42.7 GB/s,
about 35 GB/s to read in practice). The Xeon has four channels.
scripts/membw (read f32) on the Xeon:

    threads   1      3      6      18
    GB/s      13.8   34.3   57.7   68.4

Thus the tests with 6 threads on the Xeon give the decode about 1.6 times
the bandwidth of the i5. They overstate the decode of the i5. With 3
threads (about the bandwidth of the i5, but half its cores), llama-bench
method, AVX2 code:

    runtime                        pp512   tg64
    numpy-gemma (3 threads)        35.5    12.9
    llama.cpp (build-avx2, -t 3)   18.1    8.6

A token of the decode reads about 2.2 GB. The dense matrices and the
experts are 1.73 GB, and the head is 0.42 GB. At 35 GB/s
that is about 63 ms, so the i5 can give at most about 16 tok/s. On the i5:

- the bytes of each token set the rate of the decode. The work of the
  attention and the router counts less than on the Xeon.
- MTP counts more: a verify group reads the dense matrices and the head
  once for 2 or 3 tokens.
- The head is about 19% of the bytes of a token.
- The int16 cache grows with the context. At 4301 tokens it is about
  300 MB a token (25 layers with a window of 1024 rows: 210 MB). An int8
  cache reads half of that.
- The prompt pass is bound by compute, so the tests with 6 threads apply.
- The L3 of the i5 is 9 MB, not 24.8 MB.

## The int8 cache

NP_GEMMA_KV_INT8=1 makes the copy of the cache that the decode reads int8.
KVCache kv="int8" and the server option --kv-attn int8 do the same. Each group of
32 values gets a scale of max |x| / 127. The decode programs then take the
attention mode "q8", with the records KV_WRITE8, ATTN_Q8, and ATTN_Q8_MT.
The split attention (as_split_body) reads int8 or int16 values with the
same code. A group keeps the bits of the steps (check_mt passes on AVX2 and
VNNI), and MTP gives the tokens of the plain decode.

The 26B at 4301 tokens of context, AVX2 code (a step and the head):

    threads   int16      int8
    3         9.95       10.91 tok/s
    6         16.19      17.73 tok/s

The KL of the decode against the int16 cache is 1.3e-3 over 160 steps
(float x, VNNI), and all 160 top tokens agree. With the int8 x of AVX2, the
KL against the float decode goes from 3.9e-3 to 4.3e-3.

Later the int8 rows became the only copy: the int8 form keeps no float32
rows. KVCache.read gives dequantized rows to the prompt attention and to the
other float readers. Thus the prompt attention also reads int8 values, and
the KL of the decode against the int16 cache is 3.7e-3 (158/160 top
tokens).

A test rounds the keys and values of the int16 cache to int8 in
the prompt. It gives the same values. Thus the cost comes from the int8
values, not from the code. The cache takes about a sixth of the memory. A prompt of 2048
tokens is 3.5% slower (the dequantization of the rows).

## The int8 cache on the GPU, and int16 keys with int8 values

The GPU takes the int8 cache too (GPUKV "int8"). The records KV_WRITE8,
ATTN_Q8, and ATTN_Q8_MT have kernels on the GPU. The decode uses k_attn_fdt
and k_attn_part, and a prompt uses k_flash_qc_h. The GPU drafter of MTP
reads the int8 cache of the target.

A third form keeps int16 keys and int8 values (NP_GEMMA_KV_INT8=v,
KVCache kv="k16v8"). Its records are KV_WRITEV8, ATTN_V8, and ATTN_V8_MT. The
kernels have one element type for the keys and one for the values.

The cost of each half: a test runs the 12B on the CPU with a float cache
and float attention. It rounds the keys or the values to int8 when it
stores them, and it runs 160 decode steps:

    rounded     KL against none   top tokens
    keys        6.7e-4            156/160
    values      7.7e-4            154/160
    both        1.3e-3            154/160

The keys and the values cost about the same, and the costs add. The NLL of
the text does not move (within 0.01). On random data the keys gave most of
the error, so the test on the model is the one that counts.

The 26B on the GPU: check_mt passes with each form when the hot experts stay
fixed (NP_GEMMA_GPU_HOT_DYN=0). HotCache can move experts between the group
and the steps, and then even the int16 cache gives other bits.

## No float32 rows in the cache

KVCache keeps only quantized rows in each form, also int16: the float32
rows are gone. The cache of the int16 form takes a third of its old memory.
The prompt attention on the CPU reads dequantized rows (KVCache.read).

- A prompt of 2048 tokens of the 26B (AVX2, 6 threads): 56.2 to 55.3
  tok/s.
- The KL against the chat references: 0.0011, 0.0004, 0.0012 (before:
  0.0016, 0.0004, 0.0010).
- check_program.py passes for each form. The Python path of a group
  (gemma_attn_decode_i16_mt) now takes the split attention, as the program
  does. Before this change, the two gave different bits.
