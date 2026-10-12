# Plan: reclank, a layout for more models and more machines

Status: P0 to P3 are done (section 10). P4 and after are a plan.

This plan renames the package np_gemma to reclank. It splits the code
into a core, one package for each model family, one backend for each kind
of machine, and one server. The aims:

- A reader can find the code of a model, a kernel, or a machine type
  without reading 15K-line files.
- A new model family (GLM-5.3-Flash first) adds a package and a server
  adapter. It does not touch the code of the other families.
- A new machine type adds a backend directory. The CPU backends:
  - AVX2;
  - AVX2 with AVX-VNNI;
  - AVX-512;
  - AVX-512 with VNNI.

  The GPU backends:
  - NVIDIA Pascal (GTX 1070, P102-100, GTX 1080 Ti);
  - NVIDIA Ampere (and Ada);
  - NVIDIA Blackwell (RTX 50x0).

  AMD (HIP) and the DGX Spark have no test hardware. The layout leaves
  room for them (section 4.6), but no phase builds them.
- One server serves every family. It lives in its own directory, server/.

The rule of each step: the tokens and the logits stay bit for bit the
same, and the speed stays the same. A step that moves code changes no
numbers. A step that changes numbers is a separate step, with its own
check.

## 1. The code now

The numbers below come from a read of the tree (not a run).

### 1.1 Size and shape

- np_gemma/ has 38 Python files (27.4K lines) and csrc/ (34K lines of
  C and CUDA).
- csrc/bf16_linear.c (11.2K lines) is the only C translation unit. It
  includes moe.c, mlx_affine.c, kquants.c (5.5K), deltanet.c, hyperconn.c,
  and qsa.c. The build gives one library, libgemma_<hash>.so.
- csrc/gpu.cu (15.4K lines) is one CUDA translation unit with file-static
  state. Its runner has one switch of 101 opcodes. The Gemma and the Qwen
  kernels sit in two bands, but the bands mix: the QSA attention of
  Qwen3.8 sits among the Gemma attention kernels.
- There is no pyproject and no test runner. 132 of the 153 scripts import
  np_gemma, through a sys.path line or PYTHONPATH=.

### 1.2 The CPU kernels

- The ISA is fixed at compile time. NP_GEMMA_ARCH=vnni|avx512|avx2 forces
  a build. Otherwise cops.py reads the CPU flags.
- One library is built per process. A few old kernels also have runtime
  multiversions (target("avx2,fma") and target("avx512f,...")).
- Inside the files, `#if GEMMA_X86 && defined(__AVX512F__)` /
  `#elif GEMMA_X86` / `#else` blocks choose the code. About a quarter of
  bf16_linear.c and half of kquants.c is ISA-only code.
- There is no AVX-VNNI code (the 256-bit dpbusd of Alder Lake and later)
  and no ARM code.
- The GGUF K-quant dots of kquants.c (Q8_0, Q4_K, Q5_K, Q6_K, the t2
  tiles, NVX) have only a VNNI form and a scalar form. On an AVX2 CPU and
  on an AVX-512 CPU without VNNI, Qwen runs on the scalar code.
- The "avx512" build is in fact AVX-512 for bf16_linear.c and AVX2 or
  scalar for kquants.c and mlx_affine.c.
- The interpreter (gp_step), the runners, the OpenMP team code, the NUMA
  code, moe.c, deltanet.c, hyperconn.c, and qsa.c have no intrinsics.
  They do not depend on the ISA.

### 1.3 The GPU kernels

- gpu.py builds gpu.cu with `nvcc -O3 -arch=native`. The cache key has
  no GPU arch, and the libraries go to np_gemma/_libs in the source tree.
  A second machine on the same tree with another GPU can load a library
  for the wrong arch.
- There is one `__CUDA_ARCH__` check, for PDL (sm_90 and up). Everything
  else builds for every target, so the file builds only for sm_80 and up.
- The code needs CUDA 13: gg_pdl_edges uses the CUDA 13 graph edge API
  with no CUDART_VERSION guard.
- The tensor-core code is inline PTX: mma.sync m16n8k16 (f16, bf16, s8),
  m16n8k32 s8, m16n8k8 tf32, ldmatrix, and cp.async. Every tensor-core
  kernel needs sm_80.
- FP8 (e4m3) and FP4 (e2m1, NVFP4) are decoded in software. NVFP4 runs on
  the int8 mma after an expansion to int8. No kernel uses the FP8 tensor
  cores of sm_89 or the FP4 tensor cores of sm_120.
- Environment variables and program flags choose the fallbacks (fp32 or
  dp4a), not the GPU type. Two paths have no fallback without tensor
  cores: GGUF groups above 16 rows (k_kq_gemm) and the q4x experts of the
  26B (k_moe_gemm_q4x).
- Warp size 32 is assumed everywhere: 125 full-warp shuffle masks and
  about 120 lane/4 and lane%4 indices.
- There is no AMD, HIP, or ROCm code.

### 1.4 The program system

The opcode list exists three times: in program.py, in bf16_linear.c,
and in gpu.cu. Each copy is kept in step by hand. An unknown opcode
does nothing (`default: break`). PROGRAMS.md describes the system; its
line numbers are old.

### 1.5 The model families

- **Gemma 4:** model.py, e4b.py, unified.py, gemma4_encoders.py,
  assistant.py, parts.py, prompt.py, flash.py, and the Gemma half of
  gpu.py (ModelGPU, E4BGPU, SplitCompiler, GPUKV, GPUDrafter).
- **Qwen3.6 and Qwen3.8:** one inheritance tree.
  - Qwen4(QwenGGUF), Qwen4Cache(QwenCache), Qwen4GPU(QwenGPU).
  - The files: qwen.py, qwen4.py, qwen_gpu.py, qwen4_gpu.py, st_qwen4.py,
    vision_qwen.py, qwen_tok.py, mlx_affine.py (the MLX path; to be removed).
- **Shared by name, specific in fact:**
  - Gemma only: config.py, rope.py, tokenizer.py, chat.py, ct.py.
  - Mixed: gguf.py, ops.py, program.py (the Gemma forms are its last 470
    lines), gpu.py, media.py, cops.py.
- **The imports tie the families together:**
  - `np_gemma/__init__.py` imports model and e4b, sets OpenMP variables,
    and makes a NUMA syscall.
  - gpu.py imports model at the top, and qwen_gpu.py imports gpu.py. So a
    Qwen run loads the Gemma model code.
- **The same job is written more than once:**
  - KV caches: 8 classes.
  - Prefix reuse: model.Session, qwen.QwenSession (unused; to be removed), and the
    snapshots of serve_qwen4.
  - MTP loops: speculative.stream, assistant.mtp_stream,
    Qwen4CPU.generate_mtp, serve_qwen4._decode_mtp.
  - Decode loops, image loaders, and LayerPlan (twice).

### 1.6 The servers

- np_gemma/server.py has the HTTP part and a Backend class.
- The Backend is Gemma code: the Gemma template, the Gemma output parser,
  and a pool of model.Session objects.
- scripts/serve.py (242 lines) serves the Gemma models.
- scripts/serve_qwen4.py (1148 lines) serves Qwen3.6 and Qwen3.8. It
  subclasses Backend and adds its own template (jinja2), tool-call
  parser, cache snapshots, model thread, crash reports, and think levels.
- server.py finds Qwen features by hasattr.
- Both scripts repeat the think-budget, wrap, media, debug, and sampling
  options. --mtp and --thinking mean different things in each.

### 1.7 Settings

- About 195 NP_GEMMA_* environment variables; about 101 of them are
  NP_GEMMA_GPU_*.
- They are read in 30 files, among them the C and CUDA code (62 names
  through getenv). About 36 places in the package and 42 scripts set
  them.
- README.md documents them by hand.

## 2. The target layout

The repo directory becomes reclank/ (see decision D1). The tree:

```
reclank/
  reclank/                    the package
    __init__.py               no side effects; lazy exports
    settings.py               every RECLANK_* setting: name, type, default,
                              text; NP_GEMMA_* accepted as old names
    runtime/
      threads.py              OpenMP defaults, CPU sets (from __init__.py)
      numa.py                 placement, pinning
    formats/
      gguf.py                 reader, writer, dequant, overlay (no model names)
      safetensors.py          st.py
      nvfp4.py  bf12.py  rq8.py  rq6k.py  tq6.py   one file per weight form
      nvfp4_st.py             ModelOpt NVFP4 safetensors as a GGUF-like view
                              (from st_qwen4.py; Qwen3.8 and GLM use it)
    ir/
      opcodes.py              the ONE opcode table; generates the C header
      program.py              Program, records, emit, run_py
      compiler.py             Compiler and the generic k_* forms
    core/
      kvcache.py              one cache interface: append, truncate, snapshot,
                              restore, fork (each family implements it)
      session.py              prefix reuse and snapshots for every family
      sampling.py  speculative.py  think_guard.py  weight_cache.py
      engine.py               the Engine interface (section 5)
    encoders/
      runtime.py              _EncCompiler, _Runner, _Weights
      media.py                image, video, audio loading
    models/
      base.py                 the Family interface (section 5)
      gemma4/
        model.py  e4b.py  unified.py  parts.py  prompt.py  assistant.py
        config.py  rope.py  tokenizer.py  chat.py  ct.py  flash.py
        forms.py              the Gemma part of program.py
        gguf_names.py         the Gemma tensor names and text_config
        encoders.py           Gemma4Vision, Gemma4Audio
        gpu.py                ModelGPU, E4BGPU, SplitCompiler, GPUKV, GPUDrafter
        template.jinja  data/
      qwen/                   Qwen3.6 (qwen35moe) and Qwen3.8 (qwen4exp)
        qwen35.py  qwen38.py  gpu35.py  gpu38.py
        tokenizer.py  chat.py (template, tool-call parser)  vision.py
      glm5/                   new (section 6)
    backends/
      select.py               finds the machine; picks the backends
      cpu/
        build.py              flags, detection, cache key, build
        lib.py                ctypes bindings (from cops.py), by family
        csrc/                 (section 3)
      cuda/
        build.py              toolkit, arch, flags, cache key, build
        caps.py               the capability table of the GPU
        runtime.py            GPUProgram, Mirror, HotCache, ProgramLRU, gpumm
        csrc/                 (section 4)
  server/                     the one server (section 5)
  scripts/
    check/  bench/  profile/  study/  convert/  tools/
  tests/
    unit/                     pytest, no model files
    golden/                   token and logit dumps of each family
    quality/                  the quality suite (as now)
  docs/
    README parts, PROGRAMS.md, the plans, GPU_NOTES.md
```

- The built libraries go out of the source tree, to
  ~/.cache/reclank/libs/<host-arch>/. The source tree is on sshfs for some
  machines. A build there is slow, and two machines can share a stale
  library.
- The old outputs at the top (the gen_*.json files, perfs, example.txt)
  go to docs/history/ or out of the repo.

## 3. The CPU backends

### 3.1 The tiers

| Tier | Flags | Example CPUs |
|---|---|---|
| scalar | none (portable C) | any; also the reference |
| avx2 | -mavx2 -mfma -mf16c | i5-8500 (Coffee Lake), Zen 2/3 |
| avx2_vnni | avx2 + -mavxvnni | Alder Lake and later (P cores), Zen 5 client |
| avx512 | -mavx512f -mavx512bw -mavx512vl | Skylake-SP/W (W-2133) |
| avx512_vnni | avx512 + -mavx512vnni | Cascade Lake (W-2295, the 2-socket Xeon), Ice Lake, Zen 4 |

Later tiers (not in this plan's work, but the layout takes them): AMX
(Sapphire Rapids) and aarch64 NEON (section 4.6).

### 3.2 The source layout

```
backends/cpu/csrc/
  common/                     no intrinsics; one build for every tier
    interp.c                  gp_step, gp_exec, gemma_run, the parts runner
    team.c                    the OpenMP team plan, pinning, GOMP wrap
    numa.c                    node helpers (from kquants.c)
    moe.c  deltanet.c  hyperconn.c  qsa.c
    formats.c                 the loader conversions (Q8_0, Q6_K, NV4/NVX,
                              BF12 packing); scalar code
  include/
    kernels.h                 the kernel interface: every *_body and every
                              entry point that the interpreter calls
    opcodes.h                 generated from ir/opcodes.py
    simd.h                    small shared helpers (hsum, cvt) per tier
  kernels/
    <family>/<tier>.c         e.g. kernels/int4_rows/avx2.c,
                              kernels/kq_dot/avx512_vnni.c,
                              kernels/attn_decode/scalar.c
```

The kernel families come from the sections of bf16_linear.c and
kquants.c:

- **Products:** int8 GEMV/GEMM, int4 rows and tiles, i4q (int8 x), int4
  with int8 x and int4 with int16 x, Q4X (16-row int4), the K-quant dots,
  t2 tiles, NVX, x16 bodies, BF12.
- **Attention:** decode attention (f32, int8, int16, TQ6, split) and
  flash prefill.
- **Small ops:** element ops (rms, gelu, tanh, softcap), router,
  qkv-norm-rope.
- **MoE and fused:** the MoE bodies and the fused decode entry points.

Each family directory has a scalar.c. A tier file exists only where it
is faster.

### 3.3 The fallback chain

build.py builds common/ once for each tier. Then, for each kernel family,
it compiles the first file that exists in the chain of the tier:

```
avx512_vnni -> avx512 -> avx2_vnni -> avx2 -> scalar
avx512      -> avx2 -> scalar
avx2_vnni   -> avx2 -> scalar
avx2        -> scalar
```

(AVX-512 VNNI with VL gives the 256-bit dpbusd as well, so an
avx512_vnni build can take an avx2_vnni file where no 512-bit form
exists. It compiles that file with -mavxvnni off and maps the call to
the EVEX form, as gemma_int4_q8_gemv does now.)

- The choice is per family, at build time. No function pointers sit in
  the hot path, and the chain is visible: `build.py --explain` prints the
  file of each family for a tier.
- The library name has the tier: libreclank_cpu_<tier>_<hash>.so.
- RECLANK_CPU=auto|scalar|avx2|avx2_vnni|avx512|avx512_vnni forces a
  tier. A test machine can build every tier that it can run, and compare
  them.
- Detection reads cpuid through numpy's features and /proc/cpuinfo
  (`avx_vnni` for AVX-VNNI; numpy has no name for it).

### 3.4 How to split without changing a bit

1. Generate opcodes.h first (section 4.5), so the C and CUDA code share
   one list.
2. Move the ISA-free code to common/ (one commit per file). Check: the
   build of each tier gives the same tokens as before.
3. For each kernel family, cut the `#if` branches into tier files. The
   code inside a branch does not change. The scalar `#else` branch
   becomes scalar.c. Check after each family: check_kernels_ab.py
   --compare and the golden dumps, on the avx2 and avx512_vnni builds.
4. Remove the runtime multiversions (gemma_have_avx512 and the target
   attributes). The tier decides.

### 3.5 New code, after the split

- **avx2_vnni:**
  - 256-bit dpbusd (`_mm256_dpbusd_avx_epi32`) forms of the products that
    now have only AVX-512 VNNI or scalar code: the K-quant dots and
    tiles, Q4X rows.
  - gemma_int4_q8_gemv already uses 256-bit dpbusd under AVX-512 VL;
    that is the first port.
  - The int16 x of AVX2 (vpmaddwd) stays; VNNI has no int16 form.
- **avx2 for Qwen:** AVX2 forms of the K-quant dots (now scalar on AVX2).
  This is the largest speed gap of an AVX2 machine.
- **avx512 (no VNNI):** the K-quant dots through vpmaddubsw on 512 bits,
  in place of the AVX2 code that it takes now.

## 4. The GPU backends

### 4.1 The tiers

| Tier | Targets | Toolkit | What the hardware gives |
|---|---|---|---|
| pascal | sm_61 (GTX 1070 8 GB, P102-100 10 GB, GTX 1080 Ti 11 GB) | CUDA 12 only (the last driver branch for Pascal is R580) | dp4a; no tensor cores; no cp.async; 48 KB shared memory a block; fp16 math slow, so fp32 math with fp16 storage |
| ampere | sm_80, sm_86 (3090), sm_89 (Ada) | CUDA 12 or 13 | mma f16/bf16/s8/tf32, cp.async, ldmatrix, up to 99 KB shared memory; sm_89 adds FP8 mma |
| blackwell | sm_120a (RTX 50x0) | CUDA 13 | as Ada, plus block-scaled FP4/FP6/FP8 mma (the `a` targets), PDL |

The kernels do not ask which tier they run on. They ask for a capability.

### 4.2 Build and capabilities

**cuda/build.py:**
- Finds the toolkits: $CUDA_HOME, then /usr/local/cuda-13*, then
  /usr/local/cuda-12*, then PATH.
- Reads the compute capability and the driver version (through a tiny
  probe, or nvidia-smi).
- Picks the toolkit: RECLANK_CUDA=auto|12|13; Pascal forces 12.
- Builds with an explicit `-gencode arch=compute_XY,code=sm_XY` (sm_XYa
  on Blackwell for the FP4 mma), never -arch=native. The cache key has
  the arch and the toolkit.
- `scripts/check/cuda_builds.sh` builds the matrix sm_61, 75, 80, 86,
  89, 120, 121 against CUDA 12.9 and 13.x. It needs no GPU.

**cuda/caps.py and caps.h:** one struct, filled by gg_init from the
device attributes and the build target:

```
tc_f16, tc_bf16, tc_i8, tc_tf32     mma.sync families
tc_fp8, tc_fp4                      native FP8 / block-scaled FP4 mma
cp_async, ldmatrix, pdl, nanosleep
smem_block_max, warp_size, unified_memory, sm_count, h2d_gbps
```

- The runner (gg_launch) picks each kernel from the caps. The env
  variables of today (NP_GEMMA_GPU_*_TC) stay only as overrides.
- The Python compilers read the same caps for their limits: the largest
  GGUF group, Q4X on or off, the encoder form. This replaces the
  scattered checks.

### 4.3 The source layout

```
backends/cuda/csrc/
  include/
    opcodes.h                 generated (shared with the CPU)
    caps.h
    port.h                    the vendor layer: CUDA or HIP names, warp size,
                              shuffles, timers, sleep (section 4.6)
    prims/
      mma.cuh                 mma wrappers per arch: sm_80 PTX; a trap stub
                              below sm_80
      copy.cuh                cp.async (sm_80+), plain loads below
      frag.cuh                fragment load and store (direct register loads,
                              ldmatrix) per arch
  runtime/
    init.cu  memory.cu  copies.cu  graphs.cu (PDL edges behind CUDART_VERSION)
    host_cpu.cu               k_signal, k_await, the CPU program thread
    experts.cu                hot expert copies, fetch, zero copy, pool
    runner.cu                 gg_load, gg_run, gg_launch (the opcode switch)
  kernels/
    gemm/      fp32.cu  tc_f16.cu  tc_bf16.cu  q8.cu  bf12.cu  dp4a.cu
    attn/      decode.cu  decode_tc.cu (fdtc, fdts)  flash.cu  flash_tc.cu
               qsa.cu  dense_tc.cu
    moe/       gemma.cu  qmoe.cu  q4x.cu  router.cu
    quant/     kquants.cu  nvfp4.cu  e4m3.cu
    recurrent/ deltanet.cu  hyperconn.cu
    head/      head.cu  sampling.cu
    enc/       encoders.cu
```

The kernels go into one library per tier. A unity build file includes
them all, so the compiler still sees one translation unit. The split is
for readers. A kernel that one tier cannot run is guarded by its caps;
the build does not leave it out.

### 4.4 The work for each tier

**ampere.** This is the code of today. The split moves it; no kernel
changes. Check on the 3090 (sm_86) and the 5060 Ti (sm_120 built as
sm_120): the golden dumps, the same bits.

**pascal.** A build study (CUDA 12.9 and 13.1) found the changes are confined to about 12 helpers:

1. Guard the mma, ldmatrix, and cp.async helpers by `__CUDA_ARCH__`
   (below 800: a trap stub, and plain copies). Replace `__nanosleep` and
   `%globaltimer` below sm_70.
2. Below caps.tc_f16, gg_launch never picks a tensor-core kernel. The
   Python side caps GGUF groups at 16 rows and turns Q4X off.
3. RECLANK_GPU_CC=61 makes a newer GPU act as Pascal, to test on the
   3090 without a Pascal card.
4. New dp4a tile kernels for large groups (prompts) in 48 KB of shared
   memory. Without them a prompt is 3 to 6 times slower; decode is near
   the speed of today.

The cards:

- **GTX 1070** (GP104, 8 GB, 256 GB/s). The E4B and the E2B fit whole.
  The 12B at 4 bits fits with a short context. The 26B runs with its
  experts on the CPU and a small hot set.
- **P102-100** (GP102, 10 GB, about 440 GB/s). A mining card. Its PCIe
  link is narrow (often x4, Gen 1), so copies from the host are slow.
  Expert copies (HotCache, the copies of a prompt) cost much more there.
  The plan: the hot set is fixed at load, with no copies during a step,
  or the experts stay on the CPU. A test of the link speed at start sets
  this (caps.h2d_gbps).
- **GTX 1080 Ti** (GP102, 11 GB, 484 GB/s). The best Pascal card for the
  26B with hot experts.

None of the three has fast fp16 math (GP102 and GP104 run fp16 at 1/64
of fp32). So the Pascal kernels do fp32 math on fp16 or int8 storage,
and dp4a for the int8 products.

**blackwell.**

1. Build for sm_120a. This tier only builds and runs the ampere
   kernels.
2. Native NVFP4: block-scaled FP4 mma for the NVFP4 weights of Qwen3.8
   and of GLM. Today the code expands FP4 to int8 for the s8 mma. The
   native form halves the bytes in shared memory and the work of the
   expansion.
3. Native FP8 mma, for FP8 checkpoints (sm_89 too). Optional: the
   NVFP4 GLM needs no FP8 path (section 6.3).

### 4.5 One opcode table

- ir/opcodes.py holds each opcode once: the name, the number, the
  operands, and the text.
- A generator writes include/opcodes.h for the CPU and the GPU. The
  Python constants come from the same table.
- An unknown opcode is an error in both runners. A test checks that every
  opcode has a CPU body, or is marked GPU-only.
- PROGRAMS.md moves to docs/ and gets its line references from symbols,
  not numbers.

### 4.6 Not built now: AMD, DGX Spark, aarch64

There is no AMD GPU and no DGX Spark to test on. The layout keeps room
for them, and the CUDA split avoids choices that would block them:

- **The vendor layer.** All CUDA names that HIP spells another way go
  through one header (include/port.h):
  - shuffles with masks;
  - warpSize;
  - `%globaltimer`;
  - `__nanosleep`;
  - the graph calls.

  This costs nothing on NVIDIA. A HIP build later starts from the
  Pascal set of kernels (no tensor cores), then adds WMMA forms of
  mma.cuh and frag.cuh for RDNA3/RDNA4.
- **The warp size.** New code takes the lane arithmetic (lane/4,
  lane%4) from port.h helpers, not literals. The about 120 old sites
  change only when a wave64 target (AMD CDNA) is real.
- **Unified memory.** The DGX Spark and AMD Strix Halo share their
  memory between the CPU and the GPU. caps.unified_memory and a
  "resident" mode (every weight on the GPU, no HotCache, no CPU part)
  cover them. The resident mode is built in P7 anyway: it is the
  natural mode of any GPU that holds the whole model (the E4B on the
  1070, Qwen3.8 on a large card).
- **aarch64.** The Spark's CPU is ARM. The CPU scalar tier builds on
  aarch64 with no x86 flags. numa.py already has the aarch64 syscall
  numbers. A NEON tier is a later directory.

## 5. One server

### 5.1 The interfaces

reclank/models/base.py defines what a family gives the server:

```
class Family:
    name                         "gemma4", "qwen", "glm5"
    detect(meta) -> bool         from GGUF general.architecture or HF config
    load(path, opts) -> Engine

class Engine:
    tokenizer                    encode, decode, special ids
    chat                         render(messages, tools, think) -> ids
                                 parse(text) -> content, reasoning, tool_calls
                                 think_open, think_close, stop ids
    cache                        the core.kvcache interface: append, truncate,
                                 snapshot, restore (the session layer uses it)
    prefill(ids, media) -> logits
    step(token) -> logits
    verify(tokens) -> logits     MTP; None when the family has no drafter
    drafter                      None, or an object for speculative.stream
    media                        None, or the encoders (image, video, audio)
    limits                       context, max batch, think levels
```

- Gemma and Qwen differ in one place here. A Gemma cache can go back (a
  cut). A Qwen or GLM recurrent state can only restore a snapshot.
- core/session.py handles both: it keeps snapshots at turn starts and
  uses a cut where the cache allows one. So one session layer replaces
  model.Session and the snapshot code of serve_qwen4.

### 5.2 The server directory

```
server/
  __main__.py                 python -m server --model PATH ...
  options.py                  one set of options for every family
  http.py                     the handler, SSE, keepalive, auth, CORS, limits
  openai.py                   request and response shapes
  backend.py                  request -> engine: sessions, sampling, think
                              budget, ThinkGuard, MTP, stop rules
  worker.py                   the one model thread and its job queue (from
                              serve_qwen4); every family uses it
  tools.py                    the tool-call formats: Gemma, Qwen XML/JSON, GLM
  media.py                    content parts -> encoder inputs
  debug.py                    --debug turn records, crash reports
```

- The family comes from the file: `--model PATH` alone finds the family,
  and `--family` forces one.
- The engine runs in one thread for all families. This fixes, for
  Gemma, the OpenMP team issue that the Qwen server fixed: a new thread
  for each request made a new team and slowed each step.

### 5.3 One set of options

| Option | Meaning |
|---|---|
| --model, --family, --tokenizer, --template | what to load |
| --backend auto\|cpu\|cuda, --cpu-tier, --gpu-device | where to run |
| --experts, --hot-gb, --resident | expert placement; --resident is the GPU-only mode |
| --ctx, --max-tokens, --max-tokens-floor | sizes |
| --thinking off\|low\|medium\|high, --think-budget, --wrap-left, --no-think-below | think control, the same for every family |
| --drafter PATH, --mtp N | MTP: N drafts; the drafter is the separate Gemma assistant or the built-in MTP layer of Qwen3.8 and GLM |
| --mmproj, --image-budget, --video-frames, --video-budget, --media-dir | media |
| --host, --port, --api-key, --cors-origin, --max-body-mb, --debug, --quiet | the server itself |

- scripts/serve.py and scripts/serve_qwen4.py stay for one phase as thin
  wrappers that map their old options. Then they go.
- check_server.py grows a fake engine of each family (a cut cache and a
  snapshot cache). It runs in pytest with no model file.

## 6. GLM-5.3-Flash

### 6.1 What the model has

This comes from the ExLlamaV3 source (architecture/glm5_next.py and
glm5_next_mtp.py). Check it against config.json before code.

- **Two kinds of layers, 3 to 1:**
  - Linear attention (KDA: a delta rule with a decay per channel, and a
    short convolution).
  - DSA layers: MLA with no RoPE, plus an indexer that picks the top k
    blocks of keys, over a compressed pool of keys.
- **Hyper-connections (mHC) in every layer:** 4 streams, Sinkhorn
  normalization (20 iterations), and a mean at the end.
- **MoE:** sigmoid routing with a bias (noaux_tc), normalized top-k, a
  routed scale, shared experts, and a clamped SiLU. The first 3 layers
  are dense.
- **An MTP layer** (DeepSeek-V3 style).
- **A vision tower** from GLM-4V.

### 6.2 What carries over

| GLM part | Nearest code now |
|---|---|
| KDA | deltanet.c and the gdn_* kernels (Qwen); needs a decay per channel in place of one per head |
| DSA indexer, top-k blocks | qsa.c and QSA_SELECT (Qwen3.8) |
| mHC | hyperconn.c and the HC_* opcodes (Qwen3.8); needs Sinkhorn |
| MoE | the Qwen MoE path and HotCache; needs the sigmoid router |
| MTP | speculative.stream with a GLM adapter |
| MLA with no RoPE | new: the latent cache, with the absorbed form for decode |

### 6.3 The weights

- **The source is the official NVFP4 checkpoint** (ModelOpt safetensors).
  It is not downloaded yet; the download waits for the owner.
- **The loader** reads it through the NVFP4 safetensors view, formats/nvfp4_st.py,
  which is st_qwen4.py made generic. This is the path of Qwen3.8 today.
- **The NVFP4 matrices run as NVFP4:**
  - blackwell: the native FP4 mma (section 4.4);
  - ampere: the int8 expansion of today;
  - pascal: dp4a.
- **The bfloat16 matrices go to BF12** where they are large. A one-time
  conversion from the safetensors writes a side file:
  `scripts/convert/bf12_from_st.py`, the BF12 packer of
  convert_q8_gguf.py --dense bf12 on a safetensors input. A measure on
  the FP8 checkpoint gave 41 ppm of the values zeroed, worst 0.0022 of
  RMS. Small matrices (norms, the router, the indexer) stay bfloat16.
- **To read from the checkpoint first:** its quantization config tells
  which matrices are NVFP4 (the routed experts only, or also the shared
  experts and the attention) and the total size. The 3090 budget (dense,
  the cache at 256K, the hot experts) follows from it.
- **No FP8 path is needed** for this checkpoint. The FP8 decisions made
  for the FP8 checkpoint (FP8 dense with 128x128 scales) stand only if
  that checkpoint is used instead.

### 6.4 The order

1. A NumPy reference of one layer of each kind, against the HF code on a
   few tokens.
2. The CPU program (scalar, then the tiers).
3. The GPU program (ampere first).
4. The server adapter: the template and the tool-call format of GLM.
5. MTP.
6. Vision.

## 6b. DeepSeek V4 Flash

DEEPSEEK_V4_PLAN.md has the plan now, checked against the reference code
of the checkpoint: the math of each part, the weights in the forms of the
checkpoint (MXFP4 experts and FP8 dense matrices as they are: the new
types KQ_MXFP4 and KQ_FP8B, read in place from the shards), the FP8 cache
that the QAT makes exact, the budget of the 3090, and the phases. The
table of 6b.2 below (the old outline) is replaced by it. This section keeps
the first outline.

The checkpoint is DeepSeek-V4-Flash-Vision-Exp (the original weights, 157
GB, 48 safetensors shards, with the reference code in inference/ and the
chat encoding in encoding/). These facts come from its config.json and the
header of its shards.

### 6b.1 What the model has

- **The shape:** 43 layers, hidden 4096, vocab 129280, context up to 1M
  (YaRN, factor 16 over 64K).
- **Attention:**
  - 64 query heads of 512 (64 of them with RoPE) and one key-value head
    (wkv: 512 values, kv_norm).
  - The queries go through a LoRA of rank 1024 (wq_a, wq_b, q_norm).
  - The output goes through a grouped LoRA (o_groups 8, o_lora_rank 1024:
    wo_a, wo_b).
  - An attention sink for each head.
- **The keys of a layer, three kinds:**
  - a sliding window of 128 tokens, in every layer;
  - a compressed pool, the compressor of 41 layers: wkv and wgate in
    bfloat16, a learned position term ape, and a norm. The ratio of the pool
    is 4 or 128 in turn (compress_ratios; layers 0 and 1 have none). The pool
    has a RoPE theta of its own (160000);
  - an indexer on the 21 layers of ratio 4: 64 heads of 128 that pick the
    top 512 blocks of the pool, as the DSA of GLM and the QSA of Qwen3.8.
- **Hyper-connections in every layer:** 4 streams, Sinkhorn normalization
  (20 iterations); the head has its own (hc_head_*). This is the same
  as GLM.
- **MoE:**
  - 256 routed experts (top 6, intermediate 2048) and one shared expert.
  - Routing: scoring sqrt-softplus, the noaux_tc bias (and a second bias
    bias_vl), normalized top-k, a routed scale of 1.5, and SwiGLU clamped
    at 10.
  - The first 3 layers route by a table (tid2eid: the 6 experts of each
    token id), not by the gate.
- **MTP:** 3 MTP layers (num_nextn_predict_layers, with their own experts
  and a confidence head), and "DSpark" drafts: blocks of 5 tokens, a
  Markov rank of 256, from the streams of layers 40 to 42.
- **Vision:** a ViT of 32 layers (dim 1024, patch 14, downsample 3, at most
  384 tokens), and an aligner MLP.

### 6b.2 The weights

| Part | Form in the checkpoint | Plan |
|---|---|---|
| routed experts | FP4 E2M1, two codes a byte, an E8M0 scale for each 32 values (MXFP4) | keep: an MXFP4 kernel (the NVFP4 kernels with an E8M0 scale for each 32 in place of an E4M3 one for each 16) |
| attention, shared experts | FP8 E4M3, an E8M0 scale for each 128 x 128 block | keep FP8 (the FP8 path of section 6.3: e4m3 to f16 on sm_86; native on sm_89 and later) |
| compressor, gate, embeddings, head | bfloat16 | BF12 for the large ones (embed and head: 129280 x 4096) |
| norms, hc, sinks | float32 / bfloat16 | as they are |

- The experts are most of the 157 GB. They fit the RAM of this machine
  (313 GB) with room. The hot set goes to the 3090, as for Qwen3.8.
- The FP8 dense part is about 6 GB (43 layers of attention and shared
  experts). It fits the GPU with the cache.

### 6b.3 What carries over

| DeepSeek V4 part | Nearest code |
|---|---|
| hyper-connections, Sinkhorn | hyperconn.c and HC_* (Qwen3.8); the Sinkhorn step of GLM (section 6) |
| indexer, top 512 blocks | qsa.c and QSA_SELECT |
| sliding window + attention sink | the sliding layers of Gemma 4 (window, sinks: new) |
| MoE with a bias, top 6 | the Qwen MoE path, HotCache; sqrt-softplus and the token table are new |
| MXFP4 experts | the NVFP4 kernels (kq NVX, the GPU kt tiles) with another scale |
| MTP layers | speculative.stream with a DeepSeek adapter; DSpark later |
| new | the compressor (a pool of compressed keys, two ratios), the grouped output LoRA, FP8 dense matrices with E8M0 scales |

### 6b.4 The order

1. A NumPy reference of one layer of each kind (ratio 0, 4, 128) against
   inference/model.py on a few tokens. The reference code needs the TP
   conversion of convert.py; for one rank, MP=1.
2. A loader that reads the safetensors directly (formats/): the FP8 blocks,
   MXFP4 to the NVX form, BF12 for embed and head.
3. The CPU program, then the GPU program (the compressor and the pool
   cache on the GPU; the experts as for Qwen3.8).
4. The engine (models/deepseek4/engine.py) and the chat encoding of
   encoding/encoding_dsv4.py.
5. MTP with the 3 layers, then DSpark.
6. Vision.

The engine comes after P6 (the Family and Engine interface). It shares the
MXFP4, FP8, Sinkhorn, and indexer work with GLM (P8), so the two go
together: P8 does the shared kernels once.

## 7. The rename

| Thing | Now | After |
|---|---|---|
| repo directory | llm/numpy-gemma | llm/reclank |
| package | np_gemma | reclank (a np_gemma shim re-exports it until P9) |
| settings | NP_GEMMA_* | RECLANK_* (old names still read, with one warning each) |
| libraries | libgemma_*.so, libgemma_gpu_*.so | libreclank_cpu_<tier>_*.so, libreclank_cuda_<arch>_*.so |
| C symbols | gemma_*, gg_*, kq_* (about 160 exported) | rc_*, renamed as each file moves in the C and CUDA splits (P4, P5) |
| GGUF metadata | np_gemma.rotation.*, np_gemma.experts_only, np_gemma.bf12.* | written as reclank.*; both names read, forever (files exist) |
| cache dir | ~/.cache/np_gemma | ~/.cache/reclank (the old one read if present) |
| log prefix, server_version | np-gemma | reclank |

- The rename of the directory and the package is one mechanical commit
  (git mv, sed of imports and names), plus the shim. It changes no
  numbers. The run mirror and its sync script follow it.
- The C symbols move with the C split (section 3.4). A separate rename
  of 160 symbols would touch every caller twice.

## 8. The phases

Each phase ends with the golden check (P0) on the CPU and the GPU
machines, and a speed run against the numbers of P0.

| Phase | Work | Check |
|---|---|---|
| P0 | **A net first.** Golden dumps: greedy tokens and the logits of 64 steps for the 26B, 12B, E4B, Qwen3.6, and Qwen3.8, on each backend (CPU avx2, CPU avx512_vnni, GPU). A pytest runner for the checks that need no model (check_server, program round-trips, the format packers). The speed numbers of today. | the dumps load and compare |
| P1 | The rename (section 7), the shim, settings.py with the old names. | golden, bit for bit |
| P2 | The package layout: break the import cycles; an __init__.py with no side effects; move the files into core/, formats/, ir/, models/gemma4, models/qwen. Old import paths re-export. Remove the Qwen3.6 MLX path (mlx_affine.py, mlx_affine.c, the Qwen/QwenCPU/QwenProgram classes, the MA_* opcodes, its checks) and qwen.QwenSession. | golden; `import reclank.models.qwen` loads no Gemma module |
| P3 | One opcode table; the generated header; unknown opcodes are errors. | golden |
| P4 | The CPU split (sections 3.2-3.4), rc_* names, the build cache out of the tree. | golden on every tier the machines can run; check_kernels_ab across tiers |
| P5 | The CUDA split (sections 4.2, 4.3), rc_* names, the explicit arch build, caps, port.h. | golden on the 3090 and the 5060 Ti |
| P6 | One server (section 5); the old scripts become wrappers. | check_server on fake engines; the quality suite's hard tier on Gemma and Qwen gives the same answers as before |
| P7 | New tiers and modes: avx2_vnni, the AVX2 K-quant dots, pascal (CC emulation first, then the 1070, the P102-100, and the 1080 Ti), blackwell sm_120a with native NVFP4, the resident mode. | golden per tier (int8 paths: the KL limits of today) |
| P8 | GLM-5.3-Flash on the NVFP4 checkpoint (section 6). | the reference layers; KL against the HF model |
| P8b | DeepSeek V4 Flash (section 6b), with the kernels it shares with GLM (FP8 dense, Sinkhorn, the indexer) done once. | the reference layers; KL against inference/model.py |
| P9 | Scripts into subdirectories, the docs into docs/, README cut into parts, the settings table made from settings.py. The shim and the old server scripts go. | the pytest runner; every script --help runs |

- P1 to P6 change no numbers. The MLX removal in P2 takes out a path
  that no server uses.
- P7 and P8 add code.
- P9 is the clean-up.
- P8 can start after P6. It does not need P7.
- AMD, the DGX Spark, and aarch64 have no phase (section 4.6).

## 9. Decisions

- **D1.** The repo directory becomes llm/reclank, with the package.
- **D2.** The C symbols become rc_* during the C and CUDA splits.
- **D3.** The Pascal cards: a GTX 1070 (8 GB), a P102-100, and a GTX 1080
  Ti. All three are sm_61. Their machine needs CUDA 12 beside CUDA 13.
- **D4, D5.** No AMD GPU and no DGX Spark. Section 4.6 keeps room for
  them; no phase builds them.
- **D6.** The Qwen3.6 MLX path and qwen.QwenSession are removed in P2.
- **D7.** GLM-5.3-Flash comes from the official NVFP4 checkpoint, read
  directly. The bfloat16 matrices go to BF12 by a conversion from the
  safetensors where it pays. The download waits for the owner.

## 10. Progress

**P0 to P7 are done** (P7 with the open items below).

- **P0.** tests/golden/golden.py and run_golden.sh make the golden dumps:
  - 5 families (26B, 12B, E4B, Qwen3.6, Qwen3.8) on 3 backends (CPU
    native, CPU AVX2, GPU): 14 runs.
  - Two runs of the code before the refactor gave the same bits in all 14
    runs, so the runs are deterministic. Their speed differs by up to 8 per
    cent on the CPU; that is the noise.
  - tests/unit runs the checks that need no model file under pytest.
- **P1.** The rename (section 7), without the C symbols (P4, P5).
- **P2.** The layout of section 2 for the Python code, and the Qwen3.6 MLX
  path and QwenSession removed.
- **P3.** One opcode table (section 4.5).

**The check.** After P3, all 14 runs have the same bits as P0:
- the tokens;
- the sha256 of the logits of each step;
- the tokens of the server path, with MTP where it runs.

The speed is in the noise.

- **P4.** The CPU split (section 3): common/ and kernels/ by family, the
  tiers of build.py (RECLANK_CPU), the rc_* symbols, the libraries in
  ~/.cache/reclank/libs.
- **P5.** The CUDA split (section 4.3), the explicit arch build and the
  toolkit choice (cuda/build.py), the caps (runtime/caps.cu), port.cuh, the
  rc_gpu_* symbols.
- **P6.** One server in server/ with an engine for each family
  (models/*/engine.py) and the session layer (core/session.py). The old
  scripts are wrappers.

**The check of P4 to P6.** The golden runs of P0 give the same bits, on
the CPU tiers the Xeon runs (native, avx2, avx512) and on the GPU. The
server golden (tests/golden/server_golden.py: short answers, SSE, think,
tools, a chat of three turns, an image) gives the same answers on the
deterministic configs; the Qwen GPU answers differ from run to run in the
old code too (the timing of the mixed-group calibration).

**P7.**

- **avx2_vnni.** The tier builds with -mavxvnni. vpdpbusd replaces
  vpmaddubsw and vpmaddwd in the Q4X kernels (q4x.c) and in the new AVX2
  dots. The Xeon has no AVX-VNNI: RECLANK_CPU_TEST_EVEX=1 builds the tier
  with the EVEX form. The golden runs of avx2_vnni give the same bits as
  avx2 (26B, 12B, E4B, Qwen3.6). The speed of that test is not that of an
  AVX-VNNI CPU (the EVEX helper is a call).
- **The AVX2 K-quant dots** (kernels/kq/dot2/): Q8_0, Q4_K, Q5_K, Q6_K, and
  F32 rows on int8 x, for the tiers without AVX-512 VNNI. Before, these
  formats ran a float loop there. Qwen3.6 on the CPU (40 threads):

      tier      decode tok/s    prompt tok/s
      avx2      2.9 -> 14.5     4.2 -> 54.7
      avx512    2.7 -> 17.5     4.2 -> 55.2

  The new avx2 run picks the same 64 tokens as the native (VNNI) run (KL
  of the first step 0.002); the old float loop left them at step 39.
  scripts/check_kq_tiers.py checks every tier against a float reference.
  The Gemma runs keep their bits on every tier.
- **The scalar tier** builds and runs (the intrinsics guarded).
- **pascal.**
  - The mma, ldmatrix, and cp.async helpers are guarded by RC_SM80
    (port.cuh): below sm_80 an mma traps, a cp.async copies at once. sm_61
    builds with CUDA 12.3.
  - RECLANK_GPU_CC=61 makes the 3090 act as Pascal: caps.tc is 0, and the
    library is built with the Pascal forms (RC_NO_SM80), so a tensor-core
    kernel that runs anyway traps. This found three such paths:
    - the window attention of the Gemma sliding layers (FDT_TS, FDT_TC);
    - GP_MOE_GPU of the KQ_Q4X experts (a large group of the 26B);
    - the group attention of Qwen over the int8 and TQ6 caches.
    Without the tensor cores, the compilers now leave these out (has_tc()).
  - All 5 families run there. Against the 3090 with its tensor cores:

        model     prompt tok/s     decode tok/s    KL first step
        E4B       537 -> 322       120 -> 121      3e-8
        12B       614 -> 194        70 -> 67       4e-5
        26B       789 -> 226        55 -> 50       5e-6
        Qwen3.6   447 -> 376        40 -> 42       1e-4
        Qwen3.8   210 -> 177        28 -> 27       2e-3

    The 12B tokens part from the tensor-core run at step 20, where the CPU
    run parts from it too; the Pascal run keeps the 64 tokens of the CPU.
- **The Pascal prompt kernels** (section 4.4, item 4). Without the tensor
  cores the products of a large group ran float32 tiles. Now:
  - k_gemm_dp4a: the int4 matrices of Gemma on int8 x (or the int16 form,
    two int8 planes), dp4a, a tile of 64 tokens by 128 rows;
  - kt_tile_dp4a: the int8 tiles of k_kq_tc and k_qmoe_*_tc on dp4a (the
    rows of w decoded once a block to shared memory): the GGUF matrices
    (Q8_0, Q5_1, Q4_K, Q5_K, Q6_K, NVFP4, Q8_R), the experts of Qwen (in
    the int16 form of RECLANK_MOE_X16), the KQ_Q4X experts of the 26B
    (GP_MOE_GPU, so a large group of the 26B copies its experts again);
  - k_flash_f32t: the attention of a large group in float32 register
    tiles over the int16, int8, and float32 caches (Pascal runs float16 at
    1/64 of float32).
  All fit the 48 KB of shared memory of a Pascal block, and sm_61 builds
  with CUDA 12.3. scripts/check_dp4a.py and check_attn_f32t.py check them
  against float references (2e-7 and 2e-6). The prompt of 744 tokens as
  Pascal on the 3090 (RECLANK_GPU_CC=61; the 3090 runs dp4a at half the
  float32 rate, a Pascal card at the full rate):

      model     tensor cores   Pascal before   Pascal now   KL to the tensor cores
      E4B       537            322             503          4e-5
      12B       614            194             441          3e-3 (*)
      26B       789            226             596          5e-8
      Qwen3.6   447            376             462          5e-4
      Qwen3.8   210            177             189          5e-4 (its dense part is BF12: float)

  (*) The attention of k_flash_f32t and k_flash_qc_mt agree to 1e-6 on the
  records of the 12B; the int8 x of the next products turns that into
  steps of an int8 level, so near-equal runs of the 12B part by a KL of
  about 1e-3.
- **blackwell.** sm_120 and sm_120a build with CUDA 13.2.
- **The resident mode.** --resident (server) or gpu.offload(resident=True)
  puts every expert of the Gemma 26B on the GPU: no CPU part, no HotCache,
  no copies of experts, a clear error when they do not fit. On the 3090:
  prompt 789 -> 900 tok/s, decode with MTP 55 -> 60 tok/s. The values move
  by 6e-4 at most (the cold experts now run on the GPU kernels).

**Open items of P7:**
- The Pascal cards themselves (the 1070, the P102-100, the 1080 Ti): no
  run on the hardware yet; caps.h2d_gbps for the narrow link of the
  P102-100.
- The BF12 matrices on Pascal (Qwen3.8's dense part) still run float32
  tiles.
- Native NVFP4 and FP8 mma on Blackwell: no GPU to test on.
- The resident mode of Qwen: --resident stops with a message; --hot-gb
  sets the memory of its experts.
- 512-bit forms of the K-quant dots for the avx512 tier (it takes the AVX2
  dots now).

**What differs from the plan, for now:**
- reclank/__init__.py still sets the thread counts and the NUMA policy.
  They must come before NumPy and OpenMP make their threads. The code is
  in runtime/threads.py; __init__ only calls it.
- formats/gguf.py keeps the Gemma tensor names and text_config.
- encoders/runtime.py (quantize_q8) imports a helper of the E4B (_KQ) when
  it runs.
