# Qwen3.8-Flash-Next on two Quadro M4000s

The goal: Qwen3.8-Flash-Next (125B, 6B active) on two Maxwell cards of 2015
and a 6-core Xeon of 2017, with the quality of the 3090 run. The weights are
NVIDIA's FP4-Local (FP4LOCAL_PLAN.md). The host is a Dell Precision 5820, the
box of TWO_GPU.md with its RAM raised to 128 GB.

Status: a design study. No code has changed. The facts come from a read of
the code; the speeds are estimates until phase F7.

## 1. The machine

| | |
|---|---|
| CPU | Xeon W-2133: 6 cores, AVX-512 F/BW/VL without VNNI (the `avx512` tier), one NUMA node |
| RAM | 128 GB DDR4-2666, four channels: about 50-60 GB/s for 6 cores (to measure) |
| GPUs | 2 x Quadro M4000: GM204, sm_52, 13 SMs, 8 GB GDDR5, 192 GB/s, 2.57 TFLOPS FP32, 120 W, one slot |
| links | PCIe 3.0 x16 for each card (48 lanes from the CPU) |
| software | driver R580 (the last branch for Maxwell), CUDA 12 (CUDA 13 dropped sm_52) |

MAXWELL_PLAN.md section 1 has the rates of an sm_52 SM. The two that matter
here: no dp4a and no tensor cores, and shifts and conversions at half and a
quarter of the FMA rate. A kernel that spends more than about 5 slots on
each weight waits for its arithmetic, not for the memory.

## 2. The weights: FP4-Local, not EXL3

| set | slots a weight on sm_52 | an M4000 is bound by | decode-row KL vs BF16 |
|---|---|---|---|
| EXL3 5.05, today (scalar rc_dp4a_us) | about 25 | arithmetic, ~50 G weights/s | 2.3e-3 |
| EXL3 5.05, a byte-sum form of the dp4a | about 14 | arithmetic, ~90 G weights/s | 2.3e-3 |
| FP4-Local (NVFP4 experts, MXFP8 dense) | about 4-5 | memory, ~250 G weights/s | 3.7e-3 / 1.08e-2 (short / long prompt) |

The trellis of EXL3 costs the same slots at any bit rate, so a smaller EXL3
does not help. NVFP4, MXFP8 and BF16 decode in a few integer operations, and
FP4-Local is NVIDIA's own QAD set (no conversion of ours). The 6-core CPU
gains too: the A16 form of KQ_NVX runs near the rate of the memory, where
MUL1 without VNNI runs at about 4-5 G weights/s a core (EXL3_PLAN.md 6.3).

The parts of FP4-Local (FP4LOCAL_PLAN.md section 1):

| part | size | where |
|---|---|---|
| routed experts, 48 layers (NVFP4, CSF scales expanded at load) | ~68 GB (+1.9 GB of plain scales) | host; the hot ones on the GPUs |
| MTP layer experts | 1.4 GB | host |
| n-gram table (NVFP4) | 28.8 GB | host, rows on demand |
| MXFP8: GDN, QSA, indexer, shared experts | ~3.0 GB | GPU |
| BF16: router, HC mixes, PLE projections, small tensors | ~1.4 GB | GPU |
| lm_head, embed_tokens (BF16) | 1.27 GB each | head on the GPU; embedding rows on the host (gpu38.py:314-329) |

A token reads about 4.4 GB of dense weights and 10 x 48 experts of 2.77 MB
(1.33 GB).

## 3. What the code does today

### 3.1 What already works below cc 80

The decode and verify kernels of the FP4-Local contract are FP32 and integer,
with no dp4a and no tensor cores:

- KQ_HOT_MOE with the contract: k_kqh_nvx_ct (recurrent/deltanet.cu:791, ~26 KB
  of shared memory), the CPU's kq_nvx_rows_f order; k_kqh_gu / k_kqh_dn for
  the shared expert; k_kqh_act, k_moe_combine (deltanet.cu:1011).
- MXFP8: k_mxf8_gemv (t = 1) and k_mxf8_gemv_s (2-8 tokens) (quant/mxf8.cu:45, 115).
- BF16: k_kq_linear, k_kq_linear_bf16 (gemm/bf16_small.cu:128).
- The router, the HC records (recurrent/hyperconn.cu), GDN below 64 tokens,
  the head and k_argmax, the MTP head k_nv4_gemv (written for 48 KB).
- k_await spins below sm_70; PDL is sm_90 only; no cooperative groups, no
  half or double atomics.

So MAXWELL_PLAN.md (the lines on "Qwen3.8 NVX hot experts: dp4a only") is out
of date for FP4-Local: k_kqh_nvx_q (dp4a) runs only without the contract.
Under RECLANK_GPU_CC=52 no FP4-Local kernel should trap (from the dispatch;
not run).

### 3.2 The blockers

**B1. The parts need two NUMA nodes.** FP4-Local runs its cold experts only
in the parts (the A16 contract is in kq/moe_part.c:410; plain KQ_MOE has A4
only), and asserts it (models/qwen/gpu35.py:1553). ExpertParts.make returns
None below two nodes (runtime/moe_parts.py:302). The W-2133 has one: today
FP4-Local stops at the load.

**B2. The CPU kernels of KQ_NVX need VNNI.** kq_nvx_rows_f (A16) and
kq_nvx_rows_a4 sit under `__AVX512VNNI__` (kernels/kq/nvx.c:17); the `avx512`
tier gets their C forms. The A16 form uses only AVX-512F (permutexvar,
fmadd, cvtph), and its C form has "the same operations in the same order",
so it can move under `__AVX512F__` with the same bits. A4 needs a form
without vpdpbusd (vpmaddubsw + vpmaddwd, as the AVX2 NVFP4 work of Nemotron);
its sums are integers, so the bits stay.

**B3. A bug: the FP8 cache in the attention without tensor cores.** attn_kv8
(attn/decode.cu:336-343) reads a byte of the 8-bit cache as int8 times the
scale. With the E4M3 cache of FP4-Local (kv_form e4m3, form 1) the bytes are
E4M3: the other readers go through rc_kv8_val (small.cu:445; decode_qc.cu:58,
group_f32.cu:357). So k_attn_part, the per-query attention when part_tc_on()
is false (Pascal and Maxwell, decode, verify, the MTP drafter), gives wrong
scores. The fix is rc_kv8_val there. It breaks FP4-Local in Pascal mode too.

**B4. The expert device does not take FP4-Local.** gpu35.py:1547 turns it off
with b12x. The device of today returns a sum for each token and the join adds
it to out (GP_REMOTE_JOIN, expert_dev.cu:599). That changes the sum order and
breaks "a decode gives the same bits whatever the hot set" (FP4LOCAL_PLAN.md
D5). What it needs, the rows form:

- The device returns the de row of each pair it computed; a join scatters them
  into host_rows, at the pair's row (row n of the cold list in a step, row p
  in a group). GP_MOE_COMBINE then adds all the pairs in the router's order,
  as now.
- The CPU gets a copy of ip with the device's pairs at -1, not compacted:
  KQ_MOE_PART skips ids below 0, keeps the row positions, and writes only the
  rows it computed (moe_part.c:456-461, 827-847).
- The kernel exists: k_kqh_nvx_ct on the device, through rc_gpu_ex_kq_job
  (expert_dev.cu:336) with types 53/53. The device's library copy needs
  rc_gpu_set_moe_contract(1) (and rc_gpu_set_moe_rot when rot_experts);
  QwenExpertDevice sets neither today, and without them the act is the plain
  silu and the bits differ. v[10] = 0, so k_kqh_sum does not run. KQH_CT_NH
  (320) is above Qwen3.8's 2560/16 = 160 and 640/16 = 40.
- A pitfall of today: with RECLANK_EXPERT_GPUS set, Qwen3.8 builds a
  QwenExpertDevice and HotCache.exclude drops its experts (gpu35.py:591, 628),
  but nothing posts to it, so the CPU computes them all.
- Verify groups go to the device up to 16 tokens and 128 pairs (EXG_TMAX);
  prompt groups do not use it (as the other models).
- The device's library is built for its own arch (devices.py:126), so an
  sm_52 device gets RC_NO_DP4A by itself. RECLANK_GPU_CC is process-wide
  (build.py:136) and lowers the primary too.

**B5. The prompt below cc 80 is off the contract.** The A4 branch needs the
tensor cores (runner.cu:1104-1119; KT_SMEM ~89 KB, over 48 KB). Below it, cc
61 takes the dp4a tiles with int8 x and the plain silu, cc 52 the float
k_qmoe_gu / k_qmoe_dn (runner.cu:1185-1233): no A4 quantization, no b12x act,
no bf16 of alpha x acc. The CPU half of a mixed group still does A4. It runs,
with other numbers. The MXFP8 linears of more than 16 tokens fall to
k_mxf8_gemv<R,8>, a loop over the tokens (slow); BF16 ones to k_kq_gemm
(float). The QSA select falls to k_qsa_query<0> (serial: ~4 ms a layer at
200K for a verify group of 4; small at the contexts an 8 GB card holds).

**B6. The toolkit.** CUDA 12 and R580 for sm_52; the sm_52 build of the
library (scripts/check_cuda_builds.sh with 52). R580 also runs a 3060, a
1070 or a 3090, so one driver serves any mix.

## 4. The design

### 4.1 Two layouts

| | A: the main card has the dense part, the second card experts | B: the dense part split over both cards, the CPU the cold experts |
|---|---|---|
| card 1 | dense 4.4 GB, head, KV, scratch: ~7 GB | half of each dense matrix, half the KV, hot experts ~4.4 GB |
| card 2 | expert device: ~7.2 GB, ~2,600 experts (~10%) | the other half, hot experts ~4.4 GB |
| new code | B4 (the rows form on the device) | a split of each linear by rows over two devices, its exchange each layer, a hot set on each card |
| decode (estimate) | ~19 tok/s | ~25-27 tok/s |

A comes first: it needs only B4, and it also helps the 3060 + 1070. B is the
stretch.

The estimate of A, a token at 8K: the dense part 4.4 GB at ~150 GB/s, ~30 ms,
plus ~5 ms of gaps between kernels (no PDL); the experts ~13-15 ms (the CPU's
cold pairs at ~50 GB/s, the device's beside them); the round trips ~2 ms. Of
B: the dense part ~15 ms on each card, +3-4 ms of exchanges (3 a layer over
host memory, ~25 us each), the experts ~13 ms on the CPU with ~13% hot.

For scale: the CPU alone (5.7 GB a token at ~55 GB/s) is ~9-10 tok/s; EXL3
5.05 on the same cards ~8 tok/s today, ~11-12 with the byte-sum form.

### 4.2 Memory

The main card in A, an 8 GB card with no display: context 0.3, MXFP8 3.0,
BF16 1.4, head 1.27, the E4M3 cache (to measure; about 0.4-0.7 GB at 32K),
scratch 0.5: about 7 GB. The MTP layer (BF16 attention, its NVFP4 head) needs
more: off by default in A. The head in NVFP4 (as the MTP head, 0.36 GB) would
free 0.9 GB, but it changes the weights of the main logits: the user's call.

The host: one copy of the experts (one node): ~70 GB with the scales and the
MTP experts, the n-gram table 28.8 GB, the rest a few GB: ~105 GB of 128,
with the server and the OS. To check in F6: what keep_fp4l.py pins on one
node, and whether the n-gram rows can stay in the file's page cache.

### 4.3 The kernels for Maxwell

- Decode: k_kqh_nvx_ct reads E2M1 values from a table in shared memory (e2s);
  the masked magic numbers of i4m (gemv.cu:26) give the same exact values in
  registers. E4M3 in k_mxf8_gemv: the same. The bits stay (the values are
  exact, the FMA order the same).
- Prompt: an on-contract A4 tile from k_moe_gemm8 (moe/group.cu:183, the 8 x 8
  float tile of Maxwell mode): the A4 x and the E2M1 codes are small integers,
  so their products and the sums of a block of 16 are exact in FP32 and give
  the int8 mma's bits. A float group tile for MXFP8 from k_gemm8
  (gemm/groups.cu:120). Both within 48 KB.
- The cold experts of a prompt: the cards stream them (77 GB over two x16
  links, ~3.2 s a chunk) behind the arithmetic (~22 s for a chunk of 4096 at
  ~1.3 TFLOPS a card), or the CPU computes them (slow on 6 cores). F4
  measures which.

## 5. Emulation before the cards

1. **The 3090 box, RECLANK_GPU_CC=52**: the dispatch of Maxwell on the sm_86
   library (no tensor cores, no dp4a, traps on a wrong dp4a). Finds the
   refusals and checks the values of the step and verify paths (B3 shows
   here). RECLANK_GPU_CC=61 for Pascal.
2. **One node on the 3090 box**: the server under numactl on node 0 (CPUs
   and memory), for B1 and the host budget.
3. **The `avx512` tier on the 3090 box**: the tier without VNNI built and run
   on the Xeon (it has AVX-512 F/BW/VL), for B2.
4. **The 5820 with the 3060 + 1070**: the expert device of B4 on real cards
   (the 1070 as the second M4000; with ballast to 8 GB, as TWO_GPU.md
   section 8). A Maxwell expert device needs an arch override for the
   device's copy only (a new switch, say RECLANK_EXPERT_CC=52, built as
   RECLANK_GPU_ARCH for that copy), since RECLANK_GPU_CC lowers the primary
   too.

## 6. Phases

| phase | work | check |
|---|---|---|
| F0 | B3: rc_kv8_val in attn_kv8. FP4-Local under RECLANK_GPU_CC=61 and 52 on the 3090: every refusal and trap listed | decode rows against the FP4-Local reference (qfn_fp4l_reference.py): KL and argmax as the 3090's own run; the default (cc 86) bits unchanged |
| F1 | B1: ExpertParts with one node (one part, all the CPUs) | the same logits as two parts (each row's sum is the same; to confirm), on the 3090 box under numactl |
| F2 | B2: kq_nvx_rows_f under `__AVX512F__`; an A4 form without VNNI | the bits of the C forms and of the VNNI tier (A16: the same order; A4: integer sums); speed a core |
| F3 | B4: the rows form on the expert device (post with -1, join by rows, the contract on the device's copy, the held mask), the device's arch switch | the device on and off give the same bits (D5), steps and verify groups; on the 3060 + 1070 first; speed |
| F4 | B5: the on-contract A4 tile and the MXFP8 group tile for sm_52; where the cold experts of a prompt run | the bits of the CPU's A4 for the experts; prompt-row KL against the reference; the rate under CC=52 |
| F5 | the decode kernels of 4.3 (E2M1 and E4M3 in registers) | the same bits; slots a weight from cuobjdump -sass of the sm_52 build |
| F6 | the memory of an 8 GB card (the budget of 4.2 from free memory, the KV of the asked context, the hot set); the host on one node in 128 GB | a load at 32K and 64K that does not run out; the hot share from the cache simulator |
| F7 | the 5820 with the two M4000s (R580): the read rate of each 256 MB of the 8 GB, the clocks under load (P2), the link rate; decode at 1K / 8K / 32K, prefill 2K / 8K, MTP on and off | section 7, measured; decode-row KL against the float reference with the 3090's scripts (the same model, not a smaller one) |
| F8 | layout B (optional): the dense split, P2P if the Quadros have it | the bits of A or a KL of the order of the sum change; the decode rate |

F0-F3 need no new card, and F3 also gives the 3060 + 1070 the FP4-Local
expert device.

## 7. The numbers to beat (estimates)

| | decode | prefill (7K prompt) |
|---|---|---|
| the 5820's CPU alone, FP4-Local | ~9-10 tok/s | ~20-40 tok/s |
| two M4000s, layout A | ~19 tok/s | ~110-150 tok/s (after F4) |
| two M4000s, layout B | ~25-27 tok/s | the same |
| with MTP (the CPU's cold experts gain most) | x1.2-1.4 | |
| the 3090 + 48-core Xeon (FP4-Local, measured) | 40 tok/s | ~820 tok/s |

## 8. Open questions

- Does the GEMV of the NVFP4 groups reach 75-85% of 192 GB/s on GDDR5?
- The host part of a step on 6 cores at 3.6 GHz (Python, the graph launches).
- Peer access between two Quadro M4000s (Quadros often have it; GeForce cards
  do not): it would halve the exchanges of B.
- The P-state of a Quadro under CUDA (P0 or P2).
- The 5820's power supply: two 6-pin cables (the 685 W or 950 W unit).
- The prompt rows of the QAD set are far from BF16 (KL 6.7e-2, against
  EXL3's 1.9e-2: FP4LOCAL_PLAN.md section 9); that is the weights', on every
  card.
