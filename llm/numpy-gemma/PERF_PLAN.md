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

Done in commit 28ab64e. README.md now takes 0.71 s. The token ids are the
same as the old code on 308 random pieces. They are the same as the ids of
the Hugging Face tokenizer on four whole files of up to 31852 tokens.

### Phase 2: a program that C runs

Python describes the forward pass as data, and a small interpreter in C runs
it. The model structure stays in Python, and C holds only general kernels.
One call to C then runs a whole decode step.

#### The expression form

A graph builder for each model gives the forward pass as nested lists, in
the style of Lisp. One layer of the 26B model:

    (layer i
      (let h   (rms_norm x (w input_layernorm)))
      (let qkv (int4_multi4 h (w q_proj) (w k_proj) (w v_proj)))
      (qkv_norm_rope qkv (w q_norm) (w k_norm) pos)
      (kv_write i qkv pos)
      (let a   (attn_decode qkv i pos))
      (let o   (int4 (w o_proj) a))
      (set x   (add x (rms_norm o (w post_attention_layernorm))))
      (let g   (rms_norm_multi4 x (w pre_feedforward_layernorm)
                                  (w gate_proj) (w up_proj)))
      (let m   (gelu_mul_int4 g (w down_proj)))
      (let r   (router x (w router)))
      (let e   (moe (rms_norm x (w pre_feedforward_layernorm_2)) r (w experts)))
      (let f   (add (rms_norm m (w post_feedforward_layernorm_1))
                    (rms_norm e (w post_feedforward_layernorm_2))))
      (set x   (mul (add x (rms_norm f (w post_feedforward_layernorm)))
                    (w layer_scalar))))

In Python this is a tree of tuples. The 12B, 26B, and E4B models each get
a builder. A new fusion is a new operation name and a new kernel.

#### The compiler

The compiler walks the tree and makes a flat list of operations:

1. Put the operations in the order of the tree. Give each value a buffer.
   Use the buffer of a value again after its last use.
2. Mark a barrier before an operation that reads a buffer that an earlier
   operation wrote after the last barrier. Two operations with no such link
   run with no barrier between them, for example the router and the dense
   MLP.
3. Write each operation as one record of fixed size: an operation code, a
   barrier flag, integer arguments, and addresses. A NumPy structured array
   holds the records, so C reads them with no conversion.

The compiler also prints the list in a readable form. Use the print to
debug a program.

#### The program is a lambda

The values that change for each step are not in a separate record. They
are the parameters of the program, and they live in the program itself. A
step program of the 26B starts like this:

    (lambda (pos ntok base kcache vcache)
      (let lo (max 0 (- pos (- window 1) base)))
      (let n  (- (+ pos ntok) base lo))
      ...
      (attn_decode q (row kcache lo) (row vcache lo) n)
      ...)

The record array has two parts:

1. The environment. It has one slot for each parameter and for each
   variable of the program. A slot holds an int64, a float64, or an address.
2. The code. It has one record for each operation. An operand of a record
   has a tag. The tag says that the operand is a literal, a slot of the
   environment, or a buffer.

Python calls the program with `prog(pos=37, ntok=1, base=0, ...)`. The call
writes the arguments into the slots of the environment, and then it calls
C. Thus the state is part of the code, and a print of the program shows the
current values of its parameters.

The parameters are:

- The position and the token count. The rope, the cache write, and the
  attention compute their rows and their window from these values, with
  the scalar operations below.
- The first row of each sliding cache. The cache drops old rows in Python
  before the call, as it does now.
- The address of each cache buffer. A buffer can move when it grows. Python
  then writes the new address into its slot. The program stays the same.

The selected experts are data, not parameters. The router writes them into
a buffer, and the moe operation reads that buffer.

#### Scalar operations

A small integer language computes the values that the kernels need: `+`,
`-`, `*`, `min`, `max`, and `select`. It has no loop. The rules for the
window, the row of the rope, and the key count are thus in the program.
They are not in each C kernel. The kernels then take plain counts and
addresses.

Each thread evaluates each scalar operation for itself, into a private copy
of the environment. The operations are cheap and give the same value in
every thread. Thus they need no barrier, and no thread writes a shared
slot.

#### The interpreter

`gemma_run(program)` opens one OpenMP region. Each thread copies the
environment, and then all threads walk the records. A kernel operation is a
kernel body with an orphaned `omp for`. An orphaned `omp for` binds to the
region of the caller, so the threads stay in the region for the whole step.
A one-row operation, such as a norm, runs in an `omp single`. A barrier
flag gives an `omp barrier`.

The kernels of today open their own region. Split each one into a body and
a wrapper. The wrapper opens the region and calls the body, so the Python
path keeps its kernels and its bits.

#### Why this form

- One program serves the decode step, the MTP verify group of up to 16
  tokens, and the three models. Each needs a builder, not new C code.
- The program holds its own state. A call binds the parameters, so a
  cache that grows does not force a new build of the program.
- A Python interpreter of the same records calls the kernels of today. It
  runs one operation at a time, so a check can compare the C result and the
  Python result after each operation, bit for bit. The first difference
  names the faulty operation.
- The interpreter can record the cycles of each operation. That replaces
  `.cache/kern_time.py` with an exact profile inside the region.

#### Steps

- 2a: split the decode kernels into a body and a wrapper. Test: the Paris
  test and `scripts/check_mt.py` give the same bits.
- 2b: the record format, the environment, the scalar operations, the C
  interpreter, the Python interpreter, and the compiler, with the
  operations of one 26B layer. Test: the C program and
  the Python path give the same bits for one layer.
- 2c: the whole decode step of the 26B in one program, with the output
  head. Python keeps the embedding, the sampler, and the cache management.
- 2d: the group of 2 to 16 tokens, then the 12B and the E4B builders.

Expect 11 to 14 ms less for each step of the 26B, about 55 ms. That is 18
tokens/s, near llama.cpp. Set NP_GEMMA_PROGRAM=0 to use the Python path.

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
  check scripts then see no values inside the step. Keep the Python path,
  with NP_GEMMA_PROGRAM=0, for those scripts.
- The operations must know the cache layout, the sliding window, and the
  int8 copy of the cache. An error there gives a wrong result with no
  crash. The check of each operation against the Python interpreter is the
  defense.
- An orphaned `omp for` in a kernel body must see the same schedule in and
  out of the program. A schedule that is not static can give another split
  of the rows. The bits do not change when each output comes from one
  thread, but a reduction over threads can change its bits.
- The absence of a barrier gives a race, and a race can pass a short test.
  The compiler marks the barriers from the buffers, not by hand. Run the bit
  checks many times, with 2, 6, and 18 threads.
- A change of the sum order in phase 3 or phase 4 can change a token. Keep
  each such change behind a variable until the token checks pass.
