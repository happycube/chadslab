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
