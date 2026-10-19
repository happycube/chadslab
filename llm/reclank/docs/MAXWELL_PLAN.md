# Maxwell mode: Gemma 4 26B-A4B on a Quadro M4000 (8 GB)

The goal: a maxwell tier beside pascal. It is the Pascal path less dp4a:
no tensor cores, no fast float16, 48 KB of shared memory a block, and now
also no int8 dot product. The first card is the Quadro M4000 (8 GB), and
the first model is the Gemma 4 26B-A4B (the QAT Q4_0 file). The 3090 +
Xeon box is the reference and the place where Maxwell is emulated.

Status: a design study. No code has changed. The facts below come from a
read of the code and from two compile-only builds (CUDA 12.3, sm_61 and
sm_52).

## 1. The card

| | Quadro M4000 | GTX 1070 (pascal, for scale) | RTX 3090 |
|---|---|---|---|
| chip, arch | GM204, sm_52 | GP104, sm_61 | GA102, sm_86 |
| SMs, FP32 lanes | 13, 1664 | 15, 1920 | 82, 10496 |
| clock | ~775 MHz (no boost to speak of) | ~1.68 GHz | ~1.7 GHz |
| FP32 | 2.57 TFLOPS | 6.5 TFLOPS | 35.6 TFLOPS |
| memory | 8 GB GDDR5, 256-bit, 192 GB/s | 8 GB, 256 GB/s | 24 GB, 936 GB/s |
| FLOPs per byte | 13.4 | 25 | 38 |
| link | PCIe 3.0 x16 | PCIe 3.0 x16 | PCIe 4.0 x16 |
| shared memory | 96 KB an SM, 48 KB a block | 96 KB, 48 KB | 100 KB, 99 KB |
| L2 | 2 MB | 2 MB | 6 MB |
| power | 120 W, one slot | 150 W | 350 W |

The M4000 has 75% of the bandwidth of a 1070 and 40% of its arithmetic.
That ratio is the main fact of this plan. On Pascal and newer cards a
Q4 GEMV waits for memory. On the M4000 the decode of a 4-bit weight costs
about as much as its read, so a kernel with a few spare instructions for
each weight waits for its arithmetic.

The rates of an sm_52 SM for each clock (CUDA Programming Guide,
"arithmetic instructions"), with what each one means here:

| Operation | Per SM per clock | Note |
|---|---|---|
| FP32 add, mul, FMA | 128 | the budget |
| 32-bit integer add, logic, LOP3 | 128 | |
| 32-bit shift, BFE | 64 | half rate: a nibble shift costs 2 slots |
| type conversion (I2F, F2F, F2I) | 32 | a quarter rate: (float)int costs 4 slots |
| 32-bit integer multiply | several XMAD | no single IMAD |
| dp4a, dp2a | none | sm_61 |
| FP16 arithmetic | none | sm_53 (Tegra) and sm_60 only |
| `__vsub4` and the other SIMD video ops | emulated | several instructions each |
| shared memory | 32 words | an FMA:LDS ratio of 4 to keep the FMA units busy |

The toolkit and the driver:

- CUDA 13 dropped sm_50 to sm_72. CUDA 12 (12.3 is installed) builds
  sm_52; build.py already has MIN_ARCH[12] = 50 and passes
  -Wno-deprecated-gpu-targets. The tier forces CUDA 12, as pascal does.
- R580 is the last driver branch for Maxwell (and for Pascal). The driver
  of this box (595) has no Maxwell support. So an M4000 in this box needs
  a pin at R580 (nvidia-pin); R580 also runs the 3090. Or the M4000 goes in
  its own machine.

## 2. What the code does on an sm_52 build today

**The build fails, for two reasons only** (nvcc 12.3, -arch=sm_52):

1. Direct calls of `__dp4a`. CUDA 12 declares it only for
   `__CUDA_ARCH__ >= 610` (sm_61_intrinsics.h). The callers:
   - Qwen KQ: kernels/quant/i8x.cu (kq_rows_i8, kq_rows2_q40, kq_row_x2,
     kq_row_x2c; 89 calls);
   - Qwen3.8 NVX hot experts: k_kqh_nvx_q (recurrent/deltanet.cu:792,
     798);
   - DeepSeek V4.1: mxx_group_x2 (dsv41/types.cu:46);
   - Nemotron IQ2: nemotron/iq2.cuh (4 helpers) and nemotron/nm.cu
     (nm_hg_rows, k_nm_gemm8, k_nm_gtile, k_nm_gemm8k, k_nm_t8d).
2. The inline PTX `dp4a.u32.s32` of e3_dp4a_us (quant/exl3.cu:60),
   unsigned by signed. ptxas rejects it below sm_61.

A test build with a `__dp4a` shim (a forced include) gets through cicc.
ptxas then rejects 7680 copies of the exl3 asm, and 96 of an unguarded
`cp.async.commit_group`.

**A bug of the pascal tier that this study found.** The real sm_61 build
fails today: nemotron/nm.cu:2625 and :2668 (k_nm_t8v, commit 153cbb7)
issue `cp.async.commit_group` outside `#if RC_SM80`. Pascal emulation
does not see it, because RECLANK_GPU_CC=61 still builds sm_86 PTX, where
the instruction assembles. Only a real sm_61 build catches it, which is
why phase M0 runs scripts/check_cuda_builds.sh.

**What already works.** The study found nothing else that needs sm_60 or
newer:

- `rc_dp4a` (gemm/int4_dp4a.cu:29) has a scalar fallback below sm_61. The
  Gemma dp4a tiles (k_gemm_dp4a, kt_tile_dp4a) call it.
- The tensor-core PTX (mma, ldmatrix, cp.async) is under RC_SM80. PDL is
  for sm_90. `__nanosleep` has a spin fallback.
- No double atomicAdd, no half arithmetic outside the tensor-core paths
  (`__hsub2` in int4_tc2.cu is emulated by cuda_fp16.hpp below sm_53), no
  bfloat16 math, no cooperative groups. The largest static shared memory
  is 45 KB. caps.smem_block_max is 48 KB below cc 70.
- The dispatch already sends cc < 61 to the float paths:
  `dp4a_on()` (int4_dp4a.cu:212) and `int8_tiles()` (runtime.py:189) need
  cc 61.

**The Gemma 26B needs no dp4a for decode.** The 26B keeps float32 x for
its dense Q4_0 matrices (Q4_I8_DENSE is off with the experts,
gemma4/gpu.py:92). The path of a step:

| Part | Kernel | dp4a |
|---|---|---|
| q, k, v; gate, up; o; down | k_int4_multi4, k_int4_linear (gemv.cu) | no, float x |
| attention, sliding and global | k_attn_fdt over the int16 cache (decode_qc.cu) | no |
| router | k_router_* | no |
| hot experts (KQ_Q4X) | k_hot_gu / k_hot_dn: q4x_group_block (hot.cu:89) | no, float x |
| cold experts | the CPU (KQ_MOE), through k_signal / k_await | - |
| head | k_q4_head or k_q6k_head | no |
| MTP verify, 2-16 tokens | k_mt_int4_rows, k_mt_gemv_n, k_hot_split_mt | no |
| MTP drafter | float32 throughout | no |

**The prompt does.** A group of more than 16 tokens takes
k_gemm_dp4a (dense) and k_moe_gemm_q4x_dp4a (all the experts on the GPU,
GP_MOE_GPU). Below cc 61 the runtime falls back:

- the dense matrices: k_gemm<0> (gemm/groups.cu), a float tile of 64 by 64
  with 4 by 4 values a thread and scalar loads from shared memory (an
  FMA:LDS ratio of 2, so at most half the FMA rate on Maxwell);
- the experts: the small form, hot experts on the GPU one pair at a time
  (k_hot_*), cold experts on the CPU. There is no float tile for KQ_Q4X
  experts (GP_MOE_GPU stops with "needs the tensor cores or dp4a").
  runtime.py:931 says the CPU GP_MOE_MT takes at most 16 tokens. A long
  prompt on this path must be checked first (M1).

This was the "Pascal before" column of REFACTOR_PLAN.md: the 26B prompt
at 226 tok/s on the 3090, against 596 with the dp4a tiles.

## 3. Design

### 3.1 The tier

- **caps.dp4a** (cc >= 61) beside caps.tc. Every launch of a dp4a kernel
  asks for it. Today k_kq_linear_i8, k_kq_multi_i8 and k_kq_glu_i8 have no
  cc check (runner.cu:773-813). Under maxwell they take k_kq_linear /
  k_kq_gemm (the float forms that RECLANK_GPU_I8X=0 picks today).
- **rc_dp4a and rc_dp4a_us move to port.cuh**, and every `__dp4a` call
  goes through them. Below sm_61, or with RC_NO_DP4A, they are the scalar
  form (four sign extends and multiply-adds). The fallback exists for
  correctness only. It costs about 12 slots for 4 products, where a float
  FMA costs 1 slot for each product. So no maxwell path chooses it for
  speed.
- **build.py:** RECLANK_GPU_CC below 61 adds -DRC_NO_DP4A (as below 80
  it adds -DRC_NO_SM80). Then the 3090 runs the scalar rc_dp4a, and a
  dp4a kernel that the dispatch picks by mistake gives the right values
  slowly, where a trap would be clearer. So RC_NO_DP4A also makes rc_dp4a
  trap unless RECLANK_GPU_DP4A_EMU=1. The trap shows a wrong dispatch,
  and the switch keeps the slow form for the families that have no float
  path yet.
- **The other families** (not the goal; they must build and must refuse
  cleanly):
  - Qwen3.6 GGUF: the float k_kq_* forms exist. Qwen runs.
  - Qwen3.8 NVX hot experts, DeepSeek V4.1 MXX hot experts, EXL3 MUL1,
    Nemotron IQ2: dp4a only. Under maxwell their programs either run the
    hot experts on the CPU (hot set empty) or stop with "maxwell: no
    float form of X". A float form of each is later work.

### 3.2 Decode: decode a nibble without a shift or a conversion

The int4 GEMV already avoids the conversion (i4f, gemv.cu:13): the bits
2^23 + nibble, minus 2^23 + 8, is (float)(nibble - 8) in two full-rate
operations. On sm_52 the shift before it costs 2 more slots, so a weight
costs about shift 2 + LOP3 1 + FADD 1 + FFMA 1 = 5 slots. The M4000 then
does 1.29 T slots/s / ~5.3 = 245 G weights/s, about 138 GB/s of Q4_0:
72% of the bus.

The shift can go. A nibble at bit 4k (k = 0..4) masked in place gives
2^23 + n 2^4k, exact while n 2^4k < 2^23 (bits 0 to 19). Then

    fma(F, 2^-4k, -(2^(23-4k) + 8)) = n - 8      (exact: power-of-two scale, one rounding)

so a weight costs LOP3 + FFMA + FFMA = 3 slots. The three high nibbles
(bits 20 to 31) take one shift of the word by 12, shared. The product with
x is the same FFMA on the same float value, so **the bits do not change**,
and the golden dumps stay valid. About 3.3 slots a weight is 390 G
weights/s (220 GB/s), above the bus, so the GEMV again waits for memory.

The same change applies to:

- k_int4_linear, k_int4_multi4, k_mt_int4_rows (int4_part, i4f);
- q4x_group_block, the hot experts of the 26B (hot.cu:78). It takes
  `__vsub4` (emulated on Maxwell) and four `(float)(int8_t)` conversions
  (4 slots each). That is about 7.5 slots a weight, ~97 GB/s on the M4000.
  It is the worst kernel of the decode. The codes are q = w + 8 in bytes,
  so the same masked magic numbers (byte k at bit 8k, k = 0..2; one shift
  for byte 3) give q - 8 with no `__vsub4` and no conversion. The order
  of the adds stays, so the bits stay too;
- the head (k_q4_head; k_q6k_head when the file has a Q6_K head).

Pascal gains too (shifts and conversions have the same rates on sm_61),
but less, because its arithmetic is 2.5 times larger.

### 3.3 MTP verify groups

k_mt_int4_rows decodes a block once and uses it for each of the t
tokens. With the 3.2 decode, a weight costs 2 slots once plus 1 slot for
each token. A group of 4 then costs 1.5 slots a weight-token: about 2
steps of arithmetic for 4 tokens, against 1.9 steps on the CPU and 1.3
on the 3090 (MTP_PLAN.md). The union of the experts of a group is larger
too. The gain of MTP on the M4000 is about 1.0 to 1.15 times. The tier
turns MTP off by default (the drafter costs memory and time), and M5
measures it.

### 3.4 Prompt: float tiles that fit Maxwell

Two kernels, both float32 FMA on float x (no quantization of x: on
Maxwell an int8 product is not faster than a float one):

1. **k_gemm<0> with 8 by 8 values a thread.** The tile is 128 tokens
   by 64 rows (or 64 by 128) of 32 columns, with 256 threads. x goes to
   shared memory as float, and w is decoded once a block, as float with
   its scale applied, into a padded tile, with float4 reads. 64 FMA for 16
   words of shared memory is the ratio of 4 that Maxwell needs. With two
   buffers: 2 x (128 + 64) x 32 x 4 = 48 KB. The target is 55-65% of the
   FMA rate (a hand-scheduled Maxwell SGEMM gets ~90%, CUDA C less).
2. **k_moe_gemm_q4x_f32**, the missing float tile for the KQ_Q4X experts
   of GP_MOE_GPU. The structure is that of kt_tile_dp4a (the rows of w
   decoded once a block to shared memory; the groups of 16 rows of Q4X fit
   a tile of 64 rows), with the inner loop of item 1.

**The copies of a prompt.** GP_MOE_GPU needs all the experts of a layer
on the GPU. About 70% of them are cold (section 4), 9 GB of the 12.9 GB.
Over PCIe 3.0 x16 (~11-12 GB/s pinned) that is about 0.8 s for each prompt
chunk. That is hidden behind about 4 ms a token of arithmetic only when
the chunk is long and the copies run ahead. So the layer copies its cold
experts to a ring of 4 buffers of 16 experts each (54 MB a buffer, 0.2 GB
in all), on a copy stream, while the tiles of the buffer before it run.
The host side of the experts must be pinned (or registered) to get the
full rate.

### 3.5 Memory of an 8 GB card

| Item | GB |
|---|---|
| CUDA context | 0.3 |
| dense weights: attention 0.62, shared MLP 0.30, embedding/head 0.42 (Q4_0), router 0.04 | 1.4 |
| KV, int16: sliding 25 x 1024 x 8 KB = 0.21; global (k_eq_v) 10 KB a token | 0.55 at 32K (1.55 at 128K) |
| prompt activations, chunk of 2048 | 0.35 |
| prompt expert ring | 0.2 |
| head, scratch, HotCache copies | 0.2 |
| display, if the card drives one | 0.3 |
| **hot experts** | **~4.0-4.5** (1200-1350 of 3840, 3.35 MB each) |

The default of today (free memory less 6 GB, gpu.py:1565) gives about
1.5 GB on an 8 GB card. The tier sizes the budget from this table
instead: free memory, less the cache of the context asked for, the
prompt buffers and a margin of 0.3 GB.

**The hot share.** On the 5060 Ti, 1.5 GB of HotCache (12% of the
experts, LFU with decay) held about 60% of the selections (GPU_NOTES.md).
The curve is concave, so 4.25 GB (33%) should hold about 75-80%. M4
measures it on the recorded selections with the cache simulator before
any run (the method of GPU_NOTES.md, step 4).

## 4. The estimate for the 26B on the M4000

**Decode** (a context of 4K, ~80% hot, after M2). Bytes for each token:

| Part | GB a token | Where |
|---|---|---|
| attention, shared MLP, router, head | 1.38 | GPU |
| hot experts, 0.8 x 8 x 30 x 3.35 MB | 0.64 | GPU |
| KV reads (sliding 0.21 + global 0.04) | 0.25 | GPU |
| cold experts, 0.2 x 0.80 GB | 0.16 | CPU |

- GPU: 2.27 GB at ~140-160 GB/s (73-83% of the bus, after M2) is 14-16
  ms. The small kernels and the gaps between them (no PDL on Maxwell)
  add about 1.5-2.5 ms.
- CPU: a layer's cold experts (5.3 MB) run while the GPU runs its hot
  experts and its shared MLP (31 MB, ~0.2 ms). On a host with 40-70 GB/s
  for the CPU dots the cold part takes 0.08-0.13 ms a layer, so it hides
  behind the GPU. Only the 30 round trips stay exposed (~15 us each, about
  0.5-0.9 ms).
  On the 5060 Ti it was the other way around: the GPU part was ~4 ms and
  the CPU part was on the critical path.
- The host part of a step: ~0.5 ms.

| Case | ms a token | tok/s |
|---|---|---|
| after M2, context 4K, 4.25 GB hot | 17-21 | **~48 (45-55)** |
| after M2, context 32K (global KV 0.33 GB more) | 19-23 | ~45 |
| today's kernels (138 GB/s GEMV, ~97 GB/s hot experts) | 21-24 | ~43 |
| 1.5 GB hot (today's default budget), 60% hot, an 18-core host | ~22 | ~45; the CPU part grows to 0.32 GB |
| a 4-core AVX2 desktop (dual-channel DDR4, ~25 GB/s for the dots) | 20-24 | ~45; the cold part just hides |
| with MTP | | x1.0-1.15 |

For scale: the 5060 Ti does 58-61 tok/s (65-68 with MTP), and the 3090
with all the experts resident does 60 (MTP).

The decode is near the speed of the 5060 Ti because the 26B reads only
2.3 GB a token, and the larger hot budget takes the CPU off the critical
path. The risk is the efficiency of the GEMVs on GDDR5 with 18-byte
blocks (two 16-bit loads for each 4 bytes of a block). If they reach only
60% of the bus, the decode is ~38 tok/s.

**Prompt** (a chunk of 2048 tokens, after M3). One token needs about
6.2 GFLOP of products (3.08 B active weights; the experts 1.43 B, attention
1.11 B, shared MLP 0.54 B) plus ~0.5 GFLOP of attention at 2K. That is 6.7
GFLOP in all.

| Case | tok/s |
|---|---|
| after M3: tiles at 1.3-1.5 TFLOPS, +15% for the norms, router, sort and attention, copies overlapped | **~160-190** at 2K; ~120-150 at 512 (the copies show); ~140 at 8K (attention) |
| today's fallback: k_gemm<0> at ~0.7 TFLOPS, experts in the small form with the cold ones on the CPU | ~60-100, set by the CPU's expert products (and first M1 must show it runs past 16 tokens) |
| rc_dp4a scalar fallback through the dp4a tiles | ~40-60 (not a path) |

For scale: on the 3090 as Pascal the 26B prompt was 596 tok/s, and with
the tensor cores 789-900.

## 5. Emulation and measurement on the 3090

Three levels, from cheap to faithful:

1. **RECLANK_GPU_CC=52**: the dispatch of Maxwell on the sm_86 library
   (caps.cc 52, tc 0, dp4a 0, int8_tiles False; -DRC_NO_DP4A). This finds
   a wrong dispatch (a dp4a kernel traps) and checks the values. Decode
   should give the bits of RECLANK_GPU_CC=61, because the decode never used
   dp4a.
2. **RECLANK_GPU_ARCH=52**: the real sm_52 library. -arch=sm_52 also
   embeds compute_52 PTX, which the 3090 driver can JIT. Then the 3090 runs
   the sm_52 code itself: RC_SM80 0, the scalar rc_dp4a, the cuda_fp16
   emulation. The JIT of the unity library takes minutes, so
   CUDA_CACHE_MAXSIZE must be large enough to keep it. Not yet tried.
3. **A speed proxy** (optional, needs root): MPS with
   CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=16 (~13 of the 82 SMs) and
   `nvidia-smi -lgc 780`. An Ampere SM has the same 128 FP32 lanes, but
   I2F runs at 16 a clock (Maxwell 32) and the memory is still far faster,
   so this gives a speed within about ±25% only for the arithmetic-bound
   kernels (the prompt tiles).

The real card answers what the proxy cannot: the GEMV on GDDR5, the
link, and the gaps between kernels. A used M4000 costs little.

Each phase is checked as pascal was: scripts/check_dp4a.py and
check_attn_f32t.py extended to the float tiles (against float references,
~1e-6). The golden dumps under RECLANK_GPU_CC=52 (decode: the bits; prompt:
KL to the tensor cores). scripts/check_cuda_builds.sh with 52 in its
default list.

## 6. Phases

| Phase | Work | Check |
|---|---|---|
| M0 | the build: rc_dp4a / rc_dp4a_us in port.cuh and every `__dp4a` call through them; the exl3 asm guarded; the nm.cu commit_group guard (it breaks sm_61 today, fix first); caps.dp4a and the gates of the i8x kernels; RC_NO_DP4A and RECLANK_GPU_CC < 61 in build.py; sm_52 in check_cuda_builds.sh; the other families refuse cleanly | sm_52 and sm_61 build with CUDA 12.3; the 3090 golden dumps unchanged |
| M1 | the 26B under RECLANK_GPU_CC=52, then RECLANK_GPU_ARCH=52: decode, MTP, a prompt of 744 and 2048 tokens on today's fallback (the 16-token limit of the CPU GP_MOE_MT) | decode: the bits of CC=61; prompt: KL to the tensor cores; the tokens of the CPU run |
| M2 | the decode kernels of 3.2: int4_part / i4f, q4x_group_block, the heads | the same bits (golden); an SASS count of slots a weight (cuobjdump -sass of the sm_52 build) |
| M3 | the prompt: k_gemm<0> with 8 by 8 values a thread; k_moe_gemm_q4x_f32; the expert ring on a copy stream; pinned expert memory | check_dp4a.py-style float references; prompt KL; the proxy rate |
| M4 | memory of an 8 GB card: the budget of 3.5; the hot share from the cache simulator on the recorded selections at 2.5 / 3.0 / 3.5 / 4.0 / 4.5 GB; the chunk size of a prompt | the simulator against a live run (as GPU_NOTES.md: 3.37 against 3.32 cold a layer) |
| M5 | the M4000 itself (R580): the read rate of each 256 MB chunk of 8 GB (no 970-style slow segment); bench_decode at 1K / 4K / 32K, a prompt of 512 / 2048 / 8192, MTP on and off, the link rate (caps.h2d_gbps) | this plan's table of section 4, measured |

## 7. Open questions

- The split memory of the GTX 970 (3.5 GB fast + 0.5 GB slow) came from a
  disabled L2/ROP slice of GM204, not from its disabled SMs. The M4000
  also has 13 SMs, but all 64 ROPs, the full 2 MB L2 and the 256-bit bus,
  so all 8 GB should run at one rate. M5 checks it first: the read rate of
  each 256 MB chunk of a near-full allocation (as Nai's 970 benchmark),
  before the hot set is placed in the top of memory.

- Does the M4000's GEMV reach 75-85% of 192 GB/s with blocks of 18 bytes?
  If not, the 26B could keep its experts in the KQ_Q4X layout (aligned
  groups of 16 rows) for the dense matrices too.
- Is a float16 KV cache better than int16 on Maxwell? Both need a
  conversion or a magic-number decode. The int16 cache of today is the
  default, and the bits stay.
- The power state: Quadro cards can stay in P2 under CUDA. The clocks
  should be checked under load (nvidia-smi -q -d CLOCK).
- Which hosts take an M4000 (Haswell/Broadwell workstations with
  quad-channel DDR4 are common). The CPU tier there is avx2 (no VNNI):
  the cold experts in the int16 form of AVX2.
