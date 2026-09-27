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

  Done. Each decode kernel of the 26B, the MTP group, and the E4B now has a
  body and a wrapper. The fused entry points have a body that calls two
  bodies.

  `scripts/check_kernels_ab.py` saves the output of a fixed workload and
  compares it later. The workload covers the prompt pass, three decode
  steps, groups of 2, 3, and 5 tokens, and contexts of 40, 300, and 1100.
  All 39 arrays of the 26B are the same bit for bit, with the int8 and the
  float attention. With 6 threads in place of 18 they are also the same.
  All 26 arrays of the E4B are the same.

  The step time does not change: 3
  alternate runs gave 68.5 to 74.9 ms for the old code and 69.4 to 71.0 ms
  for the new code.
- 2b: the record format, the environment, the scalar operations, the C
  interpreter, the Python interpreter, and the compiler, with the
  operations of one 26B layer. Test: the C program and
  the Python path give the same bits for one layer.

  Done. `np_gemma/program.py` has the compiler, the 26B layer form, and the
  Python interpreter. `gemma_run` in the C file is the interpreter. The
  program needs the int8 cache, which is on after 128 tokens. The cache
  work stays in Python, in `KVCache.prepare`.

  `scripts/check_program.py` gives the same bits as the Python path. It
  tests layer 0 (sliding), 5, and 29 (global) at a context of 200 and of
  1100, with the C and the Python interpreter. It checks the hidden state
  and the new cache row. All 30 layers in one program (1350 records) also
  give the same bits.

  The 30 layers take 44 ms in the program, against about 57 ms in the
  Python loop. The bind of the parameters takes 0.8 ms. With the output
  head, a step is thus about 56 ms.
- 2c: the whole decode step of the 26B in one program, with the output
  head. Python keeps the embedding, the sampler, and the cache management.

  Done. Model.forward runs a step of one token as one program when
  `program.ready` allows it. The program holds the 30 layers and the final
  norm. The output head stays one call of Model.logits, because Session,
  the MTP loop, and the server use forward and logits as two calls.

  The program has two attention modes. The mode "q8" reads the int8 cache.
  It needs the int8 copy of every layer, which is on after 128 tokens.
  Before that, the Python loop runs the step.

  The mode "f32" reads the
  float cache (NP_GEMMA_ATTN=0, the default of the server). It uses a new C
  kernel, `gemma_attn_decode_f32s`. The Python path now uses the same kernel
  for one query, so the two paths and the MTP verify rows keep the same
  bits. NP_GEMMA_F32_ATTN=numpy gives the old NumPy attention.

  The builder also makes the dense layer of the 12B. The sum of the expert
  outputs keeps a separate multiply and add. The cost is small, and the
  program then keeps the bits of the Python path.

  Tests:

  - `scripts/check_program.py`: the same bits as the Python path in both
    modes. The test covers single layers, all layers, and eight decode steps
    through Model.forward. On the 12B, all 48 layers give the same bits.
  - `scripts/check_mt.py`: an MTP group gives the same bits as single steps
    in both modes.
  - `scripts/check_kernels_ab.py`: with the int8 cache, the arrays at a
    context of 300 and 1100 are the same as before. At a context of 40 the
    step uses the float cache, so the new C kernel changes the result by at
    most 4e-5.
  - `scripts/check_hf_decode.py`: see "Accuracy against the reference".

  The speed on jackal, in turn with llama-bench, two rounds:

      runtime                     pp512           tg128
      numpy-gemma, program        80.0, 87.2      15.92, 16.21
      numpy-gemma, Python loop    80.9            12.64
      llama.cpp                   86.9, 83.8      19.21, 18.92

  The decode gains 1.27 times. The gap to llama.cpp falls from 1.41 to
  about 1.19 times. The prompt pass is at the same speed as llama.cpp.
- 2d: the group of 2 to 16 tokens, then the 12B and the E4B builders.

  Done for the group. The layer form is the same for one token and for a
  group.

  Each kernel operation selects the kernel of one token or of a group
  from the row count of its input, as the Python path does. The attention
  of a group finds the key rows of each query itself. New operations:
  INT4_LINEAR_MT, INT4_MULTI4_MT, GELU_MUL_ROWS, ATTN_Q8_MT, ATTN_F32_MT,
  ROUTER_MT, and MOE_MT. The model keeps one program for each attention
  mode and group size.

  `scripts/check_program.py` gives the same bits as the Python path for
  steps of 1, 2, 3, 4, and 8 tokens through Model.forward, in both modes.
  A group of four takes 114.5 ms in the program, against 145.7 ms in the
  Python loop, and one token takes 54.5 ms.

  One fault of the first version is worth a note. The form gave the
  attention mode as the string "q8", and the compiler read the string as
  the name of a parameter. The group then used the float cache in the int8
  mode. The attention mode is now part of the operation name. The compiler
  also stops at a name that no let gives and that is not a parameter.

  With the program, the MTP decode of the 26B gives 19.55 tokens/s with two
  drafts. The plain decode gives 14.79 in the same run (1.32 times).

  Done for the E4B. The E4B model has its own builder, `e4b_step_form`. The
  step computes the per-layer inputs, then the 42 layers, then the final
  norm. The table lookup of the embeddings stays in Python.

  A layer that
  reuses the key and the value of an earlier layer reads the buffers of
  that layer. The bfloat16 kernel now has a body and a wrapper too. New
  operations: BF16_LINEAR, GELU, MUL, QKV_NORM, ROPE, KV_WRITE_HEADS, and
  ATTN_F32H.

  `scripts/check_program.py --e4b` gives the same bits as the Python path
  for steps of 1 to 8 tokens, at a context of 40 and of 600. The E4B gains
  more than the 26B, because it has more small matrices:

      tokens   Python     program    (context 600)
      1         88.7 ms    71.2 ms
      4        167.7 ms   117.2 ms
      8        278.6 ms   211.9 ms

  The MTP decode of the E4B gives 21.37 tokens/s with two drafts. The plain
  decode gives 14.99 in the same run, and it gave 12.54 before the program.
  Every prompt gives the same token ids with and without MTP.

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

## Accuracy against the reference

`scripts/check_hf_decode.py` compares the logits of 9 decode rows with
Gemma4ForCausalLM of transformers in float32. The reference has the weights
of the GGUF file, so a difference comes from the arithmetic. The prompt pass
has 199 tokens.

    prompt pass        cache             max |d|   mean |d|   top-1
    int8 activations   int8              8.21      1.134       89%
    int8 activations   float, C kernel   8.19      1.135       78%
    float              int8              2.64      0.283      100%
    float              float, C kernel   0.0003    0.00004    100%
    float              float, NumPy      0.0003    0.00004    100%

The new C kernel of the float cache is as close to the reference as the old
NumPy attention. The two differ by at most 3e-5.

**Fixed: the cache copy is now int16.** The attention over the int8 cache
copy gave an error of up to 2.64 in a logit. A study of the error shows the
cause. The query, the keys, and the values each gave about one third of it.
Thus a change to one part does not fix it:

    cache copy                      error of the attention output
    int8 q, k, v (the old code)     1.1e-02
    int8 k, v, float q              9.9e-03
    int8, groups of 16              8.4e-03
    bfloat16 k, v, float q          2.8e-03
    int16 k, v, float q             4.0e-05

The copy is now int16, with one scale for each group of 32 values, and the
query stays float32. The copy reads 2.125 bytes for each value, against
1.125 for int8 and 4 for float. The table gives the logits against the
reference, with a float prompt pass:

    cache            max |d|   mean |d|    top-1
    int8 (old)       2.64      0.283       100%
    int16            0.0014    0.00019     100%
    float            0.0003    0.00004     100%

The decode of the 26B has the speed of the old int8 copy:

    context   int16      float      int8 (old)
    512       59.0 ms    59.4 ms    58.4 ms
    2048      69.9 ms    74.3 ms    70.2 ms
    4096      73.7 ms    84.1 ms    72.2 ms

The server now uses the int16 copy by default (--kv-attn int16). The
program and the Python path keep the same bits in the new mode.

**The int8 activations of the prompt pass.** By default, the prompt pass
uses int8 activations for the int4 products (NP_GEMMA_INT4_Q8=1). The
llama.cpp code uses the same method: Q8_0 activations for Q4_0 weights.
Each int8 product differs from the float product by 0.45 to 1.0 per cent,
for every kind of matrix:

    product              mean error
    self_attn.o_proj     1.01e-02
    self_attn.v_proj     1.00e-02
    experts.down         8.01e-03
    mlp.down_proj        7.92e-03
    mlp.up_proj          7.51e-03
    self_attn.k_proj     7.12e-03
    mlp.gate_proj        6.76e-03
    self_attn.q_proj     6.69e-03
    experts.gate_up      4.50e-03

A small change of the residual can change the experts that the router
selects for a token. One such change at layer 7 grew to 34 tokens with
other experts at layer 29. Thus the logits of a few rows are a poor measure
for this model. A better measure is the perplexity of a text and the share
of positions with the same most probable token as the float products.

The table uses 1024 tokens of README.md and of np_gemma/model.py:

    activations                    ppl README   ppl model.py   same token
    int8 (NP_GEMMA_INT4_Q8=1)      59.28        33.45          84 per cent
    int16 for every product        54.91        35.30          97 per cent
    float matrices, int16 experts  55.12        35.84          99.5 per cent
    float (NP_GEMMA_INT4_Q8=0)     55.08        35.71          100 per cent

The int8 products change the most probable token at about 16 per cent of
the positions. The perplexity is not worse on both texts, so the change is
a perturbation more than a loss.

The speed of each form comes from its kernels. A fused float kernel of the
experts takes 82 ms for one layer of 256 tokens, and the int8 kernel takes
31 ms. The float tile changes each weight to float32 for each token block,
so its work limits it. The new int16 tile keeps the integer multiply at
half the rate of int8, and it takes 39 ms. For the attention and the dense
matrices, the float GEMM is as fast as the int16 tile. Thus the best
accurate form uses float matrices and int16 experts.

NP_GEMMA_INT4_Q8 now selects the form:

    mode   form                           ppl README   pp512
    1      int8 (default)                 59.28        6.06 s
    16     float matrices, int16 experts  55.12        9.15 s
    0      float                          55.08       11.59 s

Mode 0 now runs the experts in one region too. The int16 tile of a dense
matrix (cops.linear_int4_q16) stays in the code for a comparison. No mode
uses it, because the float GEMM is as fast.

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
