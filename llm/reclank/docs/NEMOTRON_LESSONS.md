# Lessons from the Nemotron 3.5 Bolt port

These notes give the practices that worked, and the mistakes to avoid. They
come from the port of Nemotron 3.5 Bolt to the CPU and the GPU paths. The
model is a hybrid of Mamba, attention and a mixture of experts. Its experts
are IQ2_XXS, its Mamba matrices NVFP4, and its other matrices Q8_0. Most
lessons apply to any model family.

NEMOTRON_BOLT_PLAN.md has the data of each step. The GPU is an RTX 3090
(sm_86). The CPU is a 48-core Xeon with VNNI.

## 1. Correctness first

- **Write a float32 reference first.** models/nemotron/model.py is a NumPy
  model that follows the checkpoint. Compare it with two independent
  references (for example the transformers code and llama.cpp) before you
  trust it. The transformers slow path put a floor of 1e-4 on dt, and about
  10% of the dt values were wrong with it. The fast path, vLLM and llama.cpp
  have no floor. When references disagree, find out which one is wrong.
- **Measure every engine against the float32 reference.** Use the KL
  divergence on decode rows and the top-1 agreement. A comparison of one
  engine with another tells you that they differ, not which one is better.
- **Measure the accuracy of each activation format.** On the GPU, int8 x (a
  scale for each 32 values) gave KL 0.0023, and two int8 planes (int16 x)
  gave 0.0007. An estimate said that float16 x adds almost nothing. In the
  NVFP4 products of 8 rows it raised the KL from 0.0007 to 0.0010. The Mamba
  layers are sensitive to the precision of x.
- **Use bit identity as a tool.** Keep dumps of the logits of fixed paths.
  For Nemotron these are the CPU prompt and steps, the GPU groups and steps
  at two expert budgets, and Pascal emulation (RECLANK_GPU_CC=61). The verify
  group of the drafter is another. After a change of layout or a refactor,
  compare the new dumps with the old ones. They must be equal to the bit.
- **When a change sums in a new order, prove that the order is the only
  change.** Force the old kernel with its switch and get the old bits. Then
  measure the KL of the new kernel against the float32 reference. Write the
  change of order in the commit and in the plan.
- **On the GPU, the rounds of the drafter do not give the same tokens as the
  steps.** The verify group and the steps use different kernels, so a near
  tie can flip. Judge the verify path by its KL against the float32
  reference (verify groups of forced tokens) and by the tokens taken in
  each round.
- **Know how an operand encodes.** None encodes as an integer 0, and an
  array as an address. Test the value of a pointer operand, not its tag.

## 2. Layout without weight changes

- **Repack for the access pattern.** The IQ2_XXS blocks of 66 bytes became
  rows of 8-byte parts followed by the scales of the row (KQ_IQ2R, type 67).
  The bytes are the same. A part is now one aligned load. The layout alone,
  with the same arithmetic and the same bits, took a GPU step from 15.2 ms
  to 14.3 ms.
- **Give each layout a marker.** The weightkeeper index records the layout
  (iq2="iq2r"). A keeper without the marker is not used, and the model loads
  from the shards. A stale keeper then cannot give wrong values.
- **Keep the converter in reclank/formats.** The CPU, the GPU and the keeper
  use the same function (iq2.to_rows).

## 3. GPU kernels

### 3.1 Find the real limit before you change a kernel

- Use ncu on one kernel before a rewrite. Find out if the kernel is bound
  by the instructions, by the load and store unit (LSU), by the latency of
  memory, or by DRAM. Each limit needs a different change.
- The IQ2 kernels were bound by the LSU pipe: the lookups of the grid in
  shared memory, the loads of x, and the shuffles. A grouped kernel with 25%
  fewer instructions was not faster. Fewer LSU operations made it faster.
- Judge the IQ2 kernels by instructions for each weight, not by GB/s. A
  weight is 2 bits, so the GB/s look low when the kernel is fast.
- Read the SASS (cuobjdump -sass) and the resource use (cuobjdump
  --dump-resource-usage). STACK shows that a small array with a dynamic
  index went to local memory. Also look for integer divisions in an inner
  loop, and for a count of registers that limits the blocks of an SM.

### 3.2 Practices that gave speed

- **The sums of a warp: use a transposed butterfly.** At the offsets 16, 8,
  and so on, each lane keeps half of its values and adds the values of its
  partner. N values take N - 1 + log2(32 / N) shuffles, not 5 N. The pairs
  and their order are the same as in N plain sums, so the bits are the same
  (nm_wsum in kernels/nemotron/iq2.cuh). The IQ2 kernels of a step went
  from 92 to 79 us.
- **Products of 2 to 8 rows: stream the weights.** Each warp copies the raw
  bytes of its 16 rows to shared memory with cp.async, in long runs of each
  row (256 columns a stage). Each lane then decodes only the values of its
  mma fragments into registers. Split the columns into shares so that the
  grid has about two blocks for each SM. The products of a verify group went
  from 15.1 to 10.9 ms (k_nm_t8v).
- **Pad the rows in shared memory** so that the lanes of a fragment use
  different banks. For 32-bit loads a row is 4 mod 8 words. For 64-bit loads
  a row is 2 mod 4 words.
- **Do not let each lane load its own row in small pieces.** The old kernel
  did 34 two-byte loads for each step of a Q8_0 row. A 4096 x 4096 product
  took 84 us. With the streamed copy it took 35 us.
- **Longer runs of a row help DRAM.** Stages of 256 columns were faster than
  stages of 128 columns. More stages in flight did not help.
- **Group the pairs of a mixture of experts by expert.** The 8 tokens of a
  verify group share experts (40 experts for the 72 pairs of a layer). Decode
  the rows of each expert once for all of its pairs: 7.0 to 5.95 ms. Keep
  the sum of each pair in the order of the one-pair kernel (the same bits).
- **Watch the registers.** A kernel with 147 registers ran one block on an
  SM and was slower than the kernel that it replaced. Process the pairs in
  passes of 2 and keep the list of pairs in shared memory. Cap the registers
  with __launch_bounds__ when the spill is small.
- **Cheaper sign and scale steps.** Apply a sign mask m as (g ^ m) + (m &
  0x01010101), not with __vsub4, which takes several instructions from sm_70
  on. An E4M3 byte moved into the bits of a float16 is its value times 2^-8,
  exactly. Put the 256 into the scale of the row.
- **A float32 matrix with few rows on a small group.** Read each row once
  for all the tokens, not once for each token. The router gates of a verify
  group went from 1.5 to 1.0 ms.

### 3.3 Launches and fusion

- In a CUDA graph the launch gaps go away (the span of a verify group was
  equal to the sum of its kernels). PDL (griddepcontrol) needs sm_90. On
  sm_86 the macro compiles to nothing, so the kernel times of nsys are true
  times.
- A small kernel still costs about 4 us. Put small work into the kernel that
  writes its input. The norm, the gated norm, relu^2 and the sum of the
  experts now quantize x for the int8 products. They use nm_qx_warp, which
  gives the bits of NM_QX. Quantize each source once for all the products on it
  (NmEmit.quant). A verify group went from 272 to 9 quantization launches.

### 3.4 State without copies

- **Keep the recurrent states in two sets.** The verify group and its replay
  read the set in use and write the other set. A swap of the views then
  makes the new set the set in use (NemotronGPU.swap, NM_MAMBA operand sd).
  This removed a copy of 1.3 ms in each round and another on each refusal.

## 4. Measurement

- **Make a small benchmark of the one operation, with real activations.**
  Dump the real rows once (for example the inputs of the products of a
  verify group). The benchmark of the products (t8_bench) ran in seconds. A
  full model run took minutes. For each replay, ncu on the full model saved
  and restored 24 GB of device memory. That was too slow.
- **Time small kernels with nsys, or with N records in one graph.**
  GPUProgram.profile() has a resolution of about 1 us and adds about 4 us to
  each record.
- **Sweep the variants with a form switch.** Compile a few template forms
  and choose one with an environment setting (for example RECLANK_NM_T8F).
  Measure each form, set the best as the default, and remove the forms that
  lost. Remove a path that did not win (the float16 NVFP4 path).
- **Check the arithmetic before a long build.** A factor of 0.5 was wrong
  in the first float16 NVFP4 kernel (the int8 path uses twice the E2M1
  values). A review of each scale against the old kernel found it before the
  first run.
- **Write the results and the failures into the plan.** Give the numbers.
  Write what was tried and why it was not kept, so that the next person does
  not try it again.

## 5. Working practices

- Keep the old path behind a RECLANK_ switch until the new path is proven.
  Test Pascal emulation after each change of dispatch.
- Run GPU work with OMP_NUM_THREADS=1. Leave one or two cores free on CPU
  runs.
- Wait for a background job by its exact PID. A pattern search can find its
  own command line and wait for ever.
- Commit each step that is measured and checked. A small commit with its
  numbers is easy to compare and easy to revert.
