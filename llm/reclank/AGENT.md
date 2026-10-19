# AGENT.md: a guide to this tree for coding agents

Read this file first. It gives the shape of the code, how a model runs, and
the practices that keep the work safe. The detail is in docs/: start with
docs/PROGRAMS.md (the program system) and docs/REFACTOR_PLAN.md (the
layout and the tiers). Each model has a plan, docs/<MODEL>_PLAN.md. Its
"Status" section holds the measured results.

## 1. What reclank is

reclank runs large language models with NumPy, C (OpenMP) and CUDA. There
is no PyTorch and no transformers in the runtime. Those libraries appear only
in check scripts, where they build a reference.

- The families: Gemma 4 (26B, 12B, E4B/E2B), Qwen3.6, Qwen3.8-Flash-Next
  (NVFP4, EXL3, FP4-Local), DeepSeek V4.1 Flash, GLM-5.3-Flash and
  Nemotron 3.5 Bolt.
- The weights run in the form of the file: GGUF K-quants, IQ2_XXS, NVFP4,
  MXFP8, EXL3, BF12 and others. A kernel reads the packed weights in place.
  No float32 copy of a weight is made.
- The machines: from a 4-core AVX2 PC with a Pascal card up to a 2-socket
  AVX-512 VNNI Xeon (NUMA) with an Ampere or Blackwell card. Small
  machines matter: a change must not leave AVX2 or Pascal behind.

## 2. The layout

```
reclank/                  the package
  settings.py             every RECLANK_* setting (get("X", default))
  ir/                     opcodes.py: the one opcode table (it writes
                          backends/include/opcodes.h). program.py: Program,
                          emit, finish, run, profile, run_py
  core/                   sampling, speculative (MTP), session, think_guard
  formats/                the weight files: gguf, safetensors, nvfp4_st, iq2, exl3, ...
  runtime/                threads, numa, weightkeeper (weights kept in shared memory)
  encoders/               the media encoders (vision, audio)
  models/<family>/        the model code; engine.py is the server interface
                          (models/base.py); keep*.py the weightkeeper plugins
  backends/
    cpu/build.py          the tiers, the flags, the unity source, the build
    cpu/cops.py           the ctypes bindings
    cpu/csrc/common/      ISA-free C: interp.c (gp_step), the MoE, the teams
    cpu/csrc/kernels/     the kernels by family; <family>/<tier>.c files
                          or #if blocks of the ISA inside one file
    cuda/build.py         the toolkit, the archs, the build
    cuda/runtime.py       GPUProgram, Mirror, the GPU memory
    cuda/csrc/runtime/    runner.cu (the opcode switch), graphs, copies
    cuda/csrc/kernels/    the kernels by subject (gemm, attn, moe, quant,
                          nemotron, ...); reclank_cuda.cu includes them all
server/                   the one OpenAI-compatible server (an engine a family)
scripts/                  check_* (correctness), bench_* (speed),
                          profile_*, convert_*, study_*
tests/unit/               pytest, no model files (pytest.ini)
tests/golden/             the golden dumps: tokens and logit hashes of
                          each family on each backend
docs/                     PROGRAMS.md, the plans, GPU_NOTES.md
```

## 3. How a model runs

1. A family compiles a step (one token), a prompt group (t tokens), a head
   or an encoder into a **program**: a list of records. A record is an
   opcode and up to 24 operands (literals, float bits, slots, addresses).
   See docs/PROGRAMS.md.
2. The **same buffer** runs on three interpreters:
   - the C interpreter on the CPU (common/interp.c, gp_step);
   - the CUDA runner on the GPU (runtime/runner.cu), mostly as CUDA graphs;
   - a NumPy interpreter for checks (Program.run_py).
3. A family that splits work gives part of a GPU program to the CPU:
   SIGNAL, CPU_TASK (a CPU program), AWAIT. Nemotron and Qwen use this for
   the cold experts.
4. Each family has a plain NumPy reference (for example
   models/nemotron/model.py). The C and CUDA paths are checked against it,
   and against llama.cpp or transformers where those exist.

### 3.1 To add an opcode

1. Add it once to ir/opcodes.py (the name, the number, the operand text).
   The header for C and CUDA is generated from it.
2. Add the CPU case to gp_step (common/interp.c) and its body. A body is
   called inside the OpenMP team of the program: use `#pragma omp for`,
   never a new `parallel` region.
3. Add the GPU case to runner.cu (or mark the opcode GPU-only).
4. tests/unit/test_opcodes.py checks the table. Run pytest.

### 3.2 To add a model family

Follow models/nemotron/ as the newest example: model.py (reference),
kq.py (CPU engine and compiler), gpu.py (GPU engine), keep.py
(weightkeeper), engine.py (detect, load, the tokenizer, the template). Add
the family to server/__main__.py and server/options.py. Write
docs/<MODEL>_PLAN.md first: the model, the weights as stored, the memory,
the speed estimates, the phases. Keep its Status section up to date.

## 4. Rules that are not negotiable

- **No weight changes.** A pre-quantized checkpoint runs with its own
  values. Repacking for layout (row order, block interleave, alignment) is
  allowed. Requantizing or changing a value is not, unless the owner asks.
- **Real data only.** Benchmarks and accuracy checks use real prompts,
  real activations and real routing. Do not use random tokens, random
  activations or synthetic expert choices.
- **Accuracy is measured, not assumed.** Use the KL divergence against the
  float32 reference (decode rows) and top-1 agreement. With MoE models a
  prompt's KL is noisy, since a small change flips an expert. Compare
  several runs before you call a change of 1e-4 real.
- **Speed regressions:** a change that makes any path more than 3% slower
  needs the owner's approval. CPU timing noise is about 8%. Measure twice.
- **Determinism.** The golden runs (tests/golden) must give the same bits
  after a refactor. A kernel that sums in a new order is a behavior
  change: say so in the commit and in the plan.
- **No new dependencies** in the runtime. NumPy, the C compiler and the
  CUDA toolkit are the dependencies.
- **GPU runs need OMP_NUM_THREADS=1.** The CPU teams of a GPU run come
  from RECLANK_GPU_CPU_THREADS. Extra libgomp pools put the workers to
  sleep.
- **Leave a core or two free** on a CPU run (taskset). A team on every core
  is slower and starves the machine.

## 5. Code practices

### 5.1 General

- Read the code around a change and match it: the names, the comment
  density, the idiom. New code reads like the old code.
- Comments say what a piece is and why, in plain sentences, with the
  numbers that justify it (e.g. "a warp a row waited on each block's
  bytes: 165 GB/s"). Do not narrate the change history in comments.
- The docs follow ASD-STE100 (Simplified Technical English):
  scripts/check_ste100.py checks a subset. Short sentences, one idea
  each.
- A switch for a new path is an environment setting, RECLANK_<NAME>, read
  once and cached (see nv4_gemv_off in kernels/quant/nv4.cu, or the
  module constants of models/nemotron/gpu.py). Keep the old path behind it
  until the new one is proven.
- Scripts that are checks print one clear result line (a KL, a max
  difference, PASS or FAIL).

### 5.2 Python

- NumPy only in the runtime. Arrays passed to programs are contiguous and
  kept alive (Program.keep).
- A family's compiler emits records through hooks, so the CPU and the GPU
  share one step compiler (compile_nemotron_step with em=NmEmit).
- Load models lazily and once. Large weights stay memory-mapped or come
  from the weightkeeper.

### 5.3 C (CPU)

- C11, 4-space indent, braces on every block, `static` for everything that
  is not an entry point. Entry points called from Python are `rc_*`.
- One unity translation unit per tier (backends/cpu/build.py ORDER). A new
  file goes into ORDER, after the files whose helpers it uses.
- The tiers: scalar, avx2, avx2_vnni, avx512, avx512_vnni
  (RECLANK_CPU=... forces one). Every kernel needs a scalar or generic
  form, and an AVX2 form if it is hot. Guard ISA code with
  `#if defined(__AVX2__)` and the like, never with runtime checks in a hot
  loop.
- Libraries build to ~/.cache/reclank/libs, keyed by a hash of every
  source. Any edit rebuilds on the next import.
- Watch the compiler. GCC 11 does not vectorize float reductions without
  fast-math, so write the vectors yourself. It can also reassociate the
  sums of madd intrinsics and spill. Read the assembly (compile the unity
  source with -S) when a kernel is slower than its count of operations.
- Products: x is quantized per 32 values. The default is int16 x in two
  int8 planes (exact, KL about 5x lower than int8 x). The tiled prompt
  kernels decode a weight tile once and use it on many tokens (see
  kernels/kq/nmx2.c, nm_task_tiled).

### 5.4 CUDA

- Kernels are `k_*`. Record kernels take (const gp_rec *r, const int64_t *e)
  and read operands with DP(type, i) and DI(i). Start them with
  PDL_START().
- One library per arch, one translation unit (reclank_cuda.cu). Every
  kernel must build for sm_61 (Pascal). Tensor-core forms (mma, ldmatrix,
  cp.async) are sm_80+ and are chosen at launch by the caps, with a dp4a or
  float fallback. RECLANK_GPU_CC=61 makes a newer card act as Pascal for
  tests.
- Float32 math by default: Pascal has no fast fp16. Reference-equal
  formats (bf16 caches) round like torch does.
- Small GPUs have narrow PCIe links (a P102 may be x4 Gen 1). Measure the
  link (h2d_gbps) before a design that copies weights during a step.
- Profile with nsys (`--cuda-graph-trace=node`) and ncu
  (`--graph-profiling node`). Kernel durations from profile() include the
  timer cost of each record. PDL_START() needs sm_90: on older cards it
  compiles to nothing, and the kernel times of nsys are true times.
- docs/NEMOTRON_LESSONS.md gives the practices from the Nemotron port. It
  covers the checks of correctness, layout, the limits of a kernel, and
  measurement. docs/GPU_NOTES.md gives the method of an earlier GPU effort.

## 6. How to check a change

From the fastest check to the slowest:

1. `pytest` (tests/unit, no model files).
2. The format and kernel checks of the subject: scripts/check_*.py (for
   example check_iq2.py runs every CPU tier on real rows and compares
   them).
3. The family's check against its reference: logits KL and top-1 (for
   example a GPU-against-CPU logits test, or a KL against the float32
   reference).
4. The golden dumps for a refactor: tests/golden/run_golden.sh, then
   golden.py compare (the same tokens and the same logit hashes).
5. The speed: scripts/bench_* or Program.profile_ops for the CPU, nsys for
   the GPU. Report the before and the after, on the same CPU set and team.

Report results as they are. If a check was skipped or failed, say so.

## 7. Working practices

- Models and large data live outside this tree, on local disks. Do not
  write models or large outputs into the repository or onto a network
  mount.
- Scratch files, logs and pid files go in a work directory outside the
  tree, not in /tmp.
- Servers start and stop with their scripts. Stop a process by its exact
  PID, never by a pattern that matches your own command line.
- Do not search the whole filesystem (`find /`): network mounts make it
  hang. Search known paths.
- Commits: one subject line in the house style, "reclank: <subject>:
  <what changed, with the key number>". No dates and no private paths in
  commit messages or docs (the repository is public). Commit only what the
  change needs. Leave unrelated working files alone.
- Write the results of a piece of work into the plan's Status section:
  what was built, the numbers, what is open.
