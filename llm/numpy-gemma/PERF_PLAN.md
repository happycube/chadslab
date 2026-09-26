# Plan: close the speed gap to llama.cpp

## Goal

Make the decode of the 26B model as fast as the decode of llama.cpp on
jackal. Then apply the same changes to the 12B and E4B models, to the MTP
verify step, and to the prompt pass.

## The gap today

jackal, 18 threads, OPENBLAS_NUM_THREADS=1, OMP_WAIT_POLICY=ACTIVE. The
llama.cpp build is `build-vnni`. The model is the 26B QAT Q4_0 file.

    measure                  numpy-gemma       llama.cpp        ratio
    decode, one step          68.7 ms best      52.5 ms          1.31
    decode, 200 tokens        13.5 tokens/s     19.06 tokens/s   1.41
    prompt of 512 tokens      67 tokens/s       85.06 tokens/s   1.27
    MTP decode (2 drafts)     17.25 tokens/s    24.69 tokens/s   1.43

The README gives the same kind of gap for the other two models: 1.35 times
for the decode of the 12B and 1.44 times for the E4B.

## Where the decode time goes

`.cache/kern_time.py` puts a timer on each C call of one decode step. The
step reads 2390 MB. The table gives the read rate of each part:

    part                          ms/step   calls   bytes     rate
    output head (Q6_K)              11.2       1    605 MB    54 GB/s
    experts, gate and up            11.5      30    534 MB    46 GB/s
    experts, down                    5.9      30    267 MB    45 GB/s
    query, key, value                9.1      30    398 MB    44 GB/s
    output projection                5.1      30    227 MB    45 GB/s
    dense gate and up                5.1      30    200 MB    39 GB/s
    dense down                       2.8      30    100 MB    36 GB/s
    router (float32)                 2.9      30     43 MB    15 GB/s
    norms                            3.2     211      -         -
    softmax, softcap                 0.8      31      -         -
    Python and NumPy                11.2       -      -         -
    total                           68.7                        35 GB/s

The machine reads at 59.8 GB/s. llama.cpp reaches 45.5 GB/s for the whole
step. The kernels of this project reach 42 GB/s for the C part alone. Thus
the kernels are not the main cause.

The main cause is the work between the kernels:

1. Python runs the layer loop. It takes 11.2 ms of each step, which is 16
   per cent. llama.cpp has no such cost.
2. A step makes about 420 calls to C. Each call pays for ctypes and opens a
   new OpenMP region. A norm call costs about 12 microseconds, and most of it
   is the fixed cost of the call.

   In contrast, llama.cpp keeps one pool of threads for the whole step, with
   a barrier between two operations.
3. A small matrix reads more slowly than a large one. The README shows this:
   a 3.3 MB matrix reads at about 31 to 38 GB/s. A region that covers more
   work, or that starts the next read before a barrier, reaches a higher rate.

The router is the one slow kernel. It reads 1.4 MB of float32 for each
layer at 15 GB/s.

## A second result: the tokenizer

The tokenizer is quadratic. Gemma splits no words before the BPE, so the
whole text between two special tokens is one piece. For each merge, `_bpe`
scans every pair of the piece again.

    text           tokens   encode
    2000 chars        579    0.37 s
    8000 chars       2279    6.29 s
    32000 chars      9446  105.20 s
    README.md       31852  598.00 s

The server encodes the full conversation for each request. A chat of 2000
tokens thus waits about 5 s before the prompt pass starts. The check
scripts of this project spent about 10 minutes on this step for each run.

## Phases

Each phase ends with a commit and a test. The kernels of a phase must give
the same bits as before, unless the phase says otherwise. Then the existing
checks stay useful: `scripts/check_mt.py`, `scripts/check_mtp.py`, and the
Paris test of `scripts/gguf_generate.py`.

### Phase 1: a linear tokenizer

Replace the scan of `_bpe` with a heap of the adjacent pairs and a linked
list of the symbols. Take the pair with the lowest rank, and the first
position for an equal rank. That is the order of the current code, so the
token ids do not change. The cost falls to n log n.

- Test: `scripts/check_tokenizer.py`, and a comparison of the old and the
  new encode on README.md and on the chat prompts of the check scripts.
- Expect: README.md in less than 2 s in place of 598 s.

### Phase 2: one C call for each layer

Add `gemma_decode_layer`. It runs one decoder layer of the 26B model in one
OpenMP region. The region covers the norms, the projections, the rope, the
cache write, and the attention. It also covers the dense MLP, the router,
and the experts. An `omp for` with a barrier replaces each call.
A step of one row, such as a norm, runs in an `omp single`.

The layer calls the same inline dot functions as the current kernels. Each
output value comes from one thread, as it does now. Thus the result keeps
its bits.

- Python keeps the embedding, the loop over the 30 layers, the output head,
  and the cache management. A table of weight addresses is made one time at
  load. The step then makes about 32 calls in place of 420.
- The sliding cache drops old rows in Python before the call, as it does
  now. The C code writes the new row into the buffer that Python gives.
- The same function takes a group of up to 16 tokens. It then uses the
  small-group kernels of the MTP verify step.
- Test: a new check that runs the old path and the new path on the same
  tokens and compares the hidden state bit for bit. Use one token and
  groups of 2 to 8, and a context below and above 128 and 1024.
- Expect: 11 to 14 ms less for each step, about 55 ms. That is 18 tokens/s,
  near llama.cpp.

### Phase 3: the slow kernels

After phase 2, profile the layer function with `perf stat` and with a timer
around each part inside the region.

- The router: use 512-bit vectors and give each thread a block of experts.
  The read is 1.4 MB, so the target is about 0.03 ms for each layer. The
  sum order changes, so the router can select another expert in a rare
  case. Accept that change only with a check of the token ids on the Paris
  test and on the four prompts of `scripts/check_mtp.py`.
- The dense MLP reads at 36 to 39 GB/s. In one region, the gate and up
  read and the down read can share one barrier-free schedule of row blocks.
- The experts: run gate and up, the GELU, and the down projection of all 8
  experts in one region.
- Expect: 3 to 5 ms less, about 50 ms for each step, or 20 tokens/s.

### Phase 4: the MTP verify step

A group of four tokens costs 137 ms, which is 1.9 decode steps. Phase 2
removes the Python part. Two further parts:

- The Q6_K head of a group costs 27 ms for four rows, against 11 ms for one
  row. The kernel is compute bound: it decodes each block one time, but it
  reduces each token with `_mm512_reduce_add_ps` for each block. Keep a
  vector sum for each token over the whole row and reduce one time at the
  end. This changes the bits, so apply the same change to the one-token
  kernel and check the Paris test again.
- The experts of the group read 55 ms. That read is necessary, because the
  tokens select about 21 experts in each layer.
- Expect: a group of four at about 1.5 decode steps. With the numbers of
  MTP_PLAN.md, that gives about 22 tokens/s for the MTP decode of the 26B.

### Phase 5: the prompt pass

The prompt pass is 1.27 times slower for the 26B and 1.46 times for the
12B and the E4B. First measure the stages again with
`scripts/profile_e4b_prefill.py` and `scripts/measure_prefill_glue.py`.
The known items are:

- The int8 tile computes a full block of 16 tokens for an expert with fewer
  tokens (INT4_Q8_PLAN.md, "What is left").
- The VNNI instruction `vpdpbusd` in the int4 tile. INT4_Q8_PLAN.md gives
  it as phase 4 of that plan. The 12B int8 tile gained 10 to 14 per cent
  from VNNI.
- The attention of a prompt keeps the scores in float32. The attention
  stage is about 38 per cent of the prompt pass of the 26B.

Make the plan of this phase after the measurement.

### Phase 6: the default settings

- OPENBLAS_NUM_THREADS is 8 by default (`np_gemma/__init__.py`). With 8 BLAS
  threads a small batch was up to five times slower in MTP_PLAN.md, phase 0.
  Measure the prompt pass with 1 and with 8 threads. If 1 is not slower,
  make 1 the default.
- Fix the report of `.cache/kern_time.py`. It subtracts the time of all
  steps from the time of one step, so the Python row is negative.

## Verification

1. The bit checks of each phase, as given above.
2. `scripts/check_mtp.py` on the 26B, 12B, and E4B: the same token ids with
   and without MTP.
3. The speed: `scripts/bench_gguf_models.py --prompt 512 --gen 128` and
   `llama-bench -p 512 -n 128`, in turn, three rounds. Use the load average
   of the machine as a check; a load above 4 makes the round invalid.

## Risks

- Phase 2 moves the layer loop out of Python. The hooks of the trace and
  check scripts then see only the layer output. Keep the Python path, with
  NP_GEMMA_LAYER_C=0, for those scripts.
- The C layer must know the cache layout, the sliding window, and the int8
  copy of the cache. An error there gives a wrong result with no crash. The
  bit check against the Python path is the defense.
- The 12B, 26B, and E4B models have different layers. Start with the 26B,
  then add the dense layer of the 12B. The E4B has the per-layer input and
  the shared key and value layers, so it comes last.
- A change of the sum order in phase 3 or phase 4 can change a token. Keep
  each such change behind a variable until the token checks pass.
