# Qwen3.8-Flash-Next-FP4-Local in reclank

The goal: run nvidia/Qwen3.8-Flash-Next-FP4-Local (NVIDIA's QAD checkpoint: NVFP4
routed experts, MXFP8 attention and shared experts, an NVFP4 n-gram table) on the
2-socket Xeon and the 3090 with the arithmetic its documentation and its serving code
define, exactly. A corner is cut only when the exact form would cost performance, and
then it is named in section 7 with its measurement. The serving path is the one of
Qwen3.8-Flash-Next today (M=qfe): the weightkeeper, the dense part and the hot experts on
the GPU, the cold experts on the CPU in parts, MTP drafts.

The checkpoint is byte-identical to local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD
at dc5b8ab2 (the SHA-256 of all 21 shards, config.json and the index). The model card names
two community vLLM containers; the code behind the RTX one, pinned by the image digest of
ghcr.io/local-inference-lab/vllm:release-10152026 (GitHub release karmic-kraken-20261015.8):

| Repository | Commit | What it gives |
|---|---|---|
| local-inference-lab/vllm (dev/karmic-kraken) | 8244d1c4 | the model (models/qwen4_exp/nvidia), modelopt_mixed, the CSF MoE method, the loaders |
| local-inference-lab/flashinfer (b12x under experimental/b12x) | 6023d031 | the kernels and their FP32 references: CSF decode, W4A16 and A4 MoE, MXFP8 linears, PLE, QSA |
| local-inference-lab/blackwell-llm-docker | 6a833b69 | the profile qwen38-flash-next (the environment of the RTX command) |
| NVIDIA/Model-Optimizer | 0.47.0 (c10fb3bf) | the NVFP4 and MXFP8 export semantics (no CSF code: CSF is LIL's post-processing) |
| local-inference-lab/InstantTensor | 747390dc | I/O only |

File references below are to these commits (vllm/..., flashinfer/... with b12x/ short for
flashinfer/experimental/b12x).

The architecture is that of the original Qwen3.8-Flash-Next: config.json differs from the
original's only in quantization_config and ple_embedding_dtype "nvfp4". reclank already runs
that architecture (qwen38.py, gpu38.py; QWEN38_PLAN.md); this plan is the weight set, its
contracts, and the kernels they need.

## 1. The checkpoint

L = 48 layers (36 Gated DeltaNet, 12 QSA), hidden 2560, 512 routed experts (top 10) of inner
640, one shared expert, 4 hyper-connection streams, the PLE n-gram layer at layer 1, one MTP
layer. 375,487 tensors, 102 GB.

| Part | Stored | Size |
|---|---|---|
| Routed experts, main 48 layers | NVFP4: E2M1 codes (2 a byte, low nibble = even k), E4M3 scale for each 16, F32 weight_scale_2 and input_scale for each matrix; the block scales CSF-compressed (below) | 73,728 matrices, about 68 GB |
| Routed experts, MTP layer | W4A16_NVFP4: the same codes, plain E4M3 scales [N, K/16], weight_scale_2, no input_scale | 1.4 GB |
| GDN projections (in_proj_qkv, z, a, b, out_proj), QSA q/k/v/o and indexer index_qk_proj, shared experts | MXFP8: F8_E4M3 weight, U8 E8M0 scale for each 1 x 32 (w = fp8 x 2^(s - 127)); no global or input scale | about 3.0 GB |
| PLE n-gram table (128 shards, 320,001,536 rows of 160) | NVFP4 group 16: U8 [rows, 80] codes, F8_E4M3 [rows, 10], one F32 weight_scale_2 | 28.8 GB |
| lm_head, embed_tokens | BF16 [248320, 2560], untied | 2 x 1.27 GB |
| Router, shared_expert_gate, HC matrices (input_mix_down/up, block_inject), PLE key/value proj and conv, GDN A_log/dt_bias/conv1d/norm, QSA norms | BF16 | about 1.4 GB |
| 97 HC norms and 3 PLE norms | F32 (the original bytes) | |
| MTP attention, shared expert, fc_embedding, fc_hidden, its norms and HC | BF16 | |
| Vision: attention qkv/proj | MXFP8 | 54 matrices |
| Vision: MLP fc1/fc2, merger fc1/fc2 | W4A16_NVFP4 (plain scales, F32 global) | 56 matrices |
| Vision: the rest | BF16 | 223 tensors |

**CSF** (weight_scale_encoding "csf", codec byte-window4-fixed-stream-u24-exceptions/1): for a
matrix of R rows and C = K/16 scale columns, X.weight_scale is stored as
X.weight_scale.nvfp4_csf_fixed, U8 [R/16, 16 (1 + C/2)] (gate/up (40, 1296), down (160, 336)),
and X.weight_scale.nvfp4_csf_exceptions, U32 [n]. Each slab of 16 rows: 16 row bases, then the
4-bit codes of the 16 rows (low nibble = even column); scale[r, c] = base[r] + nibble[r, c]
(base <= 240, no carry); then for each exception word, flat[w & 0xFFFFFF] = w >> 24 (row-major
positions, strictly ascending). The result is the E4M3 block-scale byte. Oracle:
flashinfer/scripts/convert_csf_blocks.py:68-77; GPU decode b12x/_lib/quant/nvfp4_csf.py:214-359;
the vLLM side vllm/model_executor/layers/quantization/nvfp4_csf.py, modelopt.py:1557-1590,
1851-1861. A sample of 40 experts: codes 78-126 (E4M3 7 to 448), no zero or subnormal code,
largest base 111, 0.27% exceptions, gate and up weight_scale_2 equal.

## 2. The contract (what "exactly" means here)

Two levels:

- **A. The weight set**: every stored value decoded to the exact number its format defines,
  and every product computed with the activation form its quantization contract defines.
  No deviation at this level.
- **B. The serving configuration** of the documented RTX PRO 6000 command (the closer of the
  two profiles to this machine: a discrete GPU, the n-gram table offloaded to the host). We
  follow it; where reclank chooses otherwise it is listed in section 7.

The parts:

1. **Routed experts, steps and verify groups: W4A16.** VLLM_B12X_MOE_FP4_FORCE_A16=1 (both
   profiles): x as BF16, the weights dequantized, FP32 accumulation:
   out = sum_k w_k alpha2_e W2 (silu(alpha1_e Wg x) * (alpha1_e Wu x)), W = E2M1 x e4m3(scale),
   alpha = weight_scale_2 (gate == up required; nvfp4_csf_utils.py:164-165). Reference:
   b12x/moe/_shared/kernels/reference.py:733-832 (moe_reference_w4a16_f32). The W4A16 packer
   drops nonzero scales with s f < 1/64 (b12x w4a16/prepare.py:357-393; f the power of two that
   brings the largest scale of a projection to <= 448): with codes 78-126, f = 1 and nothing
   drops; P0 proves it for all 73,728 matrices, else the drop is part of the contract.
2. **Routed experts, prompt rows: A4.** B12X_W4A16_A4_PREFILL=1 (both profiles). The rows of
   a forward that are prompt tokens (vllm/utils/b12x.py:301-331: the scheduler's prefill
   ranges, any length) run with NVFP4 activations; decode and verify rows stay A16. FC1: x to
   NVFP4 with one global scale for the layer, the minimum over experts of
   1 / max(input_scale_gate, input_scale_up) (b12x A4 README; with the Spark env vLLM takes the
   same amin, nvfp4_csf.py:330-343), sf = e4m3(amax / 6 x g) for each 16, E2M1 round to nearest
   even (b12x gemm/blockscaled/_quantize.py:52-71); silu(alpha g) * (alpha u) rounded to BF16,
   requantized to NVFP4 with the expert's intermediate scale 1 / input_scale_down; FC2 writes
   unweighted per-route BF16 rows, summed with the router weights in FP32
   (prefill_a4.py header). One activation plane (B12X_W4A16_A4_PREFILL_TERMS default 1).
3. **MXFP8 linears: A16.** The RTX profile has VLLM_B12X_DENSE_ACTIVATION_MODE=auto, which lets
   the b12x tuner of the target GPU pick BF16 x or dynamic MXFP8 x for each shape and token
   count (b12x gemm/blockscaled/_tuning.py:437-447): not reproducible off that GPU. We use a16,
   one of auto's two outcomes, deterministic and the more exact (section 7, D3); dynamic MXFP8
   (E8M0 ceil(log2(amax / 448)) for each 32, x / 2^e to E4M3; _quantize.py:80-85) as an option.
   Fused layers concatenate rows with their scales (qkv, gate_up, in_proj_qkvz, in_proj_ba;
   vllm/models/qwen4_exp/nvidia/model.py:667-679): the same products.
4. **The heads.** lm_head BF16 (VLLM_MXFP8_LM_HEAD=0 in the RTX profile, BF16 x). The MTP head
   is its own: lm_head quantized online to NVFP4 (VLLM_MTP_NVFP4_LM_HEAD=1: one global scale
   6 x 448 / amax(W), scaled_fp4_quant for each 16; mtp.py:766-790, online/nvfp4.py:56-78), A16.
   It changes only the drafts.
5. **Norms.** The 100 F32 norm tensors load into BF16 parameters (params_dtype bfloat16; the
   LIL card says so): rounded to BF16 once. GemmaRMSNorm (1 + w) in FP32
   (common/hyperconnection.py:57-90).
6. **Hyper-connections** (hyperconnection.py:199-245): y = GroupedGemmaRMSNorm(x) per stream of
   2560; gate = sigmoid(up(silu(down(y) / 4))); mixed = mean over streams of gate * y; out = x +
   block_out x 2 sigmoid(inject(y) / 4); the down projection's FP32 output cast to BF16 on b12x
   (nvidia/hyperconnection.py:296, 675-680). The final mixer has no inject.
7. **PLE** (layers.1.ple): hash over orders 2..3 with EOS-bounded windows, id = (mixed mod
   size_h) + offset_h (b12x sequence/ple_hash/reference.py); each of the 16 rows of 160 decoded
   E2M1 x e4m3 x weight_scale_2 and rounded to BF16 (ple_embedding/reference.py:49-187); key
   and query grouped GemmaRMSNorm, sim = sum(k q) / sqrt(2560), gate = sigmoid(sign(sim)
   sqrt(|sim|)), u = gate value, norm, depthwise conv of 4 with dilation 3, out = u + silu(conv)
   (sequence/ple/reference.py:11-176, ple_layer.py:120-121, 447-500).
8. **MTP** (mtp.py:196-300, 602-668): the draft input fc_embedding(GemmaRMSNorm(embed)) +
   fc_hidden(GemmaRMSNorm over 10240 of the pre-mixer streams), one QSA layer (BF16 but the
   W4A16 experts), the final mixer, the NVFP4 head; 3 drafts, probabilistic draft sampling,
   standard rejection sampling (the RTX command): the target's distribution is kept.
9. **Caches.** --kv-cache-dtype fp8: the QSA cache in E4M3 (b12x_qsa.py:329-539; the scales
   and the indexer's cache form in P0). GDN recurrent state float32 (config mamba_ssm_dtype
   float32), conv state BF16 (the model dtype).
10. **Vision**: MXFP8 attention with dynamic MXFP8 activations (the LIL card), W4A16 MLP and
    merger (BF16 x, input_scale dropped; modelopt.py:2726-2729).

What the containers do that is not arithmetic (prefix caching, chunked prefill sizes, CUDA
graphs, async scheduling, TP and its all-reduce, the instanttensor and b12x loaders) is left
out; none of it changes a value for one sequence. A4 is per row with a fixed global scale, so
the chunking of a prompt does not change it either.

## 3. The machine and the exact forms

The 3090 (sm_86) has no E2M1 or E4M3 tensor-core input; the Xeon 8268 (Cascade Lake) has
AVX-512 VNNI and no BF16. Each product is put in a form whose multiplies are exact, so only the
order of the FP32 sums differs from b12x (as between any two of its own kernels):

| Product | Exact form here |
|---|---|
| NVFP4 W x BF16 x (A16) | E2M1 (2 significant bits) x E4M3 (4) fits BF16's 8: W dequantized to BF16 exactly, alpha applied in FP32 after. GPU groups: BF16 mma m16n8k16 (FP32 accumulate). GPU steps: FP32 FMA. CPU: the codes through a 16-entry FP32 table (vpermps) times the scales, FMA on x (BF16 values held in FP32) |
| NVFP4 W x NVFP4 x (A4) | 2 x E2M1 is an integer in [-12, 12]: int8 x int8 products, exact int32 sums of each 16, times e4m3(sx) e4m3(sw) in FP32. GPU: int8 mma m16n8k16 (one group of 16 a k-step). CPU: vpdpbusd with the +12 offset of KQ_NVX |
| MXFP8 W x BF16 x | E4M3 x 2^k is exact in BF16 and TF32 (scale bytes 64-119): groups on the tensor cores (k_fp8b32_tc with a scale for each 1 x 32 in place of 32 x 32, x one TF32 part since BF16 is exact in it; or BF16 mma), steps FP32 FMA (k_fp8b32_gemv, the same change) |
| MXFP8 W x MXFP8 x (option) | E4M3 x E4M3 exact in FP16: FP16 mma, each block of 32 summed then times 2^(sa + sw) |
| BF16 W | as stored (KQ_BF16; never requantized: dense_mode keeps the types) |

The CSF scales are decoded once at load into plain E4M3 bytes, the experts packed as KQ_NVX
(groups of 16 rows: codes, E4M3 scales, the F32 global scale; the layout reclank already has
for NVFP4, lossless). b12x keeps them compressed on the GPU and expands the routed experts at
each call; that is a memory choice (D1).

## 4. Memory and places

Host, with the GLM-5.3 weightkeeper stopped (one weightkeeper at a time):

| | |
|---|---|
| routed experts, KQ_NVX, a copy on each node (the parts) | 2 x 68 GB |
| n-gram table, 96-byte rows (type 51 rows of 160: 80 code bytes, 10 scales, the F32 global, padding), split by rows over the nodes | 30.7 GB |
| dense, head, embed, HC, MTP, vision | about 9 GB |
| total | about 176 GB (88 a node) |

One copy split by expert (68 GB) is the fallback if the parts gain nothing.

GPU (24 GB): MXFP8 dense 3.0 GB, HC and router BF16 1.4, lm_head 1.27, the MTP layer and its
NVFP4 head 2.0, the QSA cache in E4M3 at 262K about 3.6 GB (12 layers x 1152 bytes a token: the
int8 form of the cache, its scales all 1), GDN state 0.1; the rest hot experts (2.76 MB each: 105 a
layer at 8K, 64 at 262K, 47 with the image encoder's room).

## 5. Phases

**P0. The source facts and a full scan (no weights run).**
- From the pinned source, with file:line: the BF16 rounding points of the A16 kernels between
  FC1, the activation and FC2, and how the top-k weights are applied; the A4 quantizer and its
  rounding (sf rounding, E2M1 ties, the clamp, the FC1 global scale min); the E4M3 cast of the
  QSA cache (scale, saturation, which tensors: K, V, the indexer keys) and the indexer's cache
  form; the conv-state dtype; which MoE backend the MTP experts get and whether it drops
  subnormal scales; the same for the vision W4A16 linears.
- The full checkpoint after the download: SHA-256 of the shards against the Hub; all 73,728
  CSF matrices decoded (bases <= 240, exceptions sorted, unique, in range, outside their row's
  window; any zero or subnormal scale codes, and the W4A16 drop rule's effect); all MXFP8 scale
  bytes (no 255); all MTP and vision scales.
- Optional: the decoded scales of a few experts against a plain-scale revision of
  local-inference-lab/Qwen3.8-Flash-Next-NVFP4 (single tensors by HTTP range), if one matches.
- The result goes into section 2 of this file.

**P1. The reader** (formats/fp4local_st.py, FP4LSource: the GGUF view of NVFP4Source).
CSF decode in C (rc_csf_decode) into KQ_NVX stacks; MXFP8 as a new type KQ_MXFP8 (FP8 rows,
E8M0 for each 1 x 32; the KQ_FP8B32 kernels with row scales); the MTP experts KQ_NVX; the n-gram
rows type 51; the norms rounded to BF16 (plus the +1 where the runtime expects it); BF16
tensors as stored. scripts/check_fp4l_source.py: the C decode equal to a NumPy transcription
of convert_csf_blocks._logical_source on every matrix; dequant of each type equal to the
format's definition on sampled rows.

**P2. The reference** (scripts/qfn_fp4l_reference.py). As qwen38_flash_reference.py: the
transformers model in float32 on the CPU, the routed experts loaded by a pre-hook, the n-gram
rows on demand; the weights from FP4LSource dequantized exactly; the contract of section 2 in
hooks (A16 x rounded to BF16, A4 for prompt rows with the quantizer of P0, MXFP8 linears with
BF16 x, the BF16 norms, the PLE rows rounded); the MTP layer for the drafts. Outputs: the same
prompts as the float reference of qwen38_flash_reference.py (top-64 logits, lse, argmax) and a trace of 256 prompt
+ 256 decode rows (every layer's input, output, routing) as the GLM-5.3 trace.
Two measures from it: the weight set against the float BF16 reference (decode-row KL; next to
BF12 4.6e-4, EXL3 5.05 1.6-2.3e-3, the earlier NVFP4 4.7e-3), and A4 against A16 on prompt
rows (what the documented A4 prefill costs).

**P3. The CPU kernels.** kq_nvx_rows_a16 (the FMA form of section 3, tokens in groups as
kq_nvx_rows), kq_nvx_rows_a4 (vpdpbusd, the A4 x of P0), both in GP_KQ_MOE_PART (the parts)
and the plain MoE records; KQ_MXFP8 rows for the CPU program. Checks against NumPy on trace
rows: A4 bit-equal (integer sums, the same FP32 order), A16 within FP32 sum order; a token alone
and in a group the same bits.

**P4. The GPU kernels.** Hot experts: k_kqh_nvx_a16 (steps, FP32 FMA), the verify groups on it
or on BF16 mma, prompt groups A4 on int8 mma (qmoe_tc with per-16 k-steps) with the A4
quantize, the BF16 intermediate and its requantization, per-route BF16 rows and the FP32 top-k
sum (as P0 confirms). MXFP8: k_mxfp8_gemv and k_mxfp8_tc. The QSA cache in E4M3 in the step,
split and tensor-core attention kernels. Each against the CPU form on trace rows (dB, and
bit-level where integer).

**P5. The model.** qwen38.py / gpu38.py on FP4LSource: A4 for the prompt rows and A16 for steps
and verify, on both the GPU and the CPU side of a mixed group; dense_mode as stored; the PLE,
norm and head forms of section 2. Checks: the full model against P2 (decode-row KL and argmax
on the trace and the two prompts; prompt rows A4), a verify group equal to its steps.

**P6. The weightkeeper** (keep_fp4l.py): the KQ_NVX experts on huge pages (a copy on each node
for the parts), the n-gram rows split by rows, the rest; the fill time; --check against a direct
read. It needs the GLM-5.3 keeper stopped: asked first.

**P7. MTP.** The MTP layer and the online NVFP4 head of section 2; 3 drafts; greedy and
probabilistic + standard rejection; the drafts against the reference's MTP on the trace; the
acceptance and the speed.

**P8. Vision.** The encoder from this checkpoint (the forms of 2.10) against the reference's
vision tower on real images.

**P9. Server and speed.** start_server.sh M=fp4l (taskset -c 0-46, 22 part threads a node,
OMP_NUM_THREADS=1); prompt and decode at 2K and 8K, MTP; the profile, then optimizations that
keep the bits.

## 6. The expected speed

Decode is the CPU's cold experts, as for qfe (EXL3 5.05: 36-37 tok/s at 8K from the
weightkeeper, its kernel bound by port 5 at 154-205 G weights/s). The A16 FMA form needs about
a quarter of a vector uop a weight for one token, so it should run at the memory's rate:
about 200 GB/s over both nodes at 0.5625 bytes a weight, about 355 G weights/s. With about 40%
of the picks hot, that is roughly 40-50 tok/s plain at 8K, more with MTP. Verify groups cost
more on the CPU (each token adds its FMAs), as they do now. Prompts: the A4 int8 mma is the
fast path of the 3090 (twice BF16), and the copies of cold experts over PCIe bound the groups,
as qfe's 830-870 tok/s at 8K; the NVFP4 experts are smaller than EXL3 5.05's.

## 7. Where reclank differs (named)

- **D1. The CSF scales decoded at load.** Lossless: the same E4M3 bytes. b12x keeps them
  compressed and expands per call to save GPU memory; plain scales cost 1.9 GB more on the host
  (each copy) and about 5% of the hot slots. The A4 global scales ride in the 12 spare bytes of
  each KQ_NVX group's head.
- **D2. Float32 activations outside the contracts.** vLLM runs --dtype bfloat16; reclank keeps
  float32 for the residual streams, the norms' outputs, the router's logits, the attention and the
  DeltaNet (more precise, never less). The BF16 points of the serving kept: the outputs of the
  hyper-connection mixes (the inputs of the blocks, the MoE and the head: BF16_ROUND), the inputs
  and outputs of the MXFP8 linears, everything inside the routed experts (A16 and A4), the MoE's
  output (bf16 of the routed sum, the gated shared expert, their sum), the norm weights, the
  n-gram rows, the heads' logits, the vision linears.
- **D3. MXFP8 activations a16 in the decoder** in place of auto (whose choice depends on the RTX
  PRO 6000's tuning cache; untuned it picks a16 up to 8 rows and quantized above). Vision follows
  the card (dynamic MXFP8 on its attention).
- **D4. The small MXFP8 matrices** (the DeltaNet's a and b, the indexer's q and k: 48, 512, 128
  rows) run as float32 (their exact values) on float32 inputs, without the a16 rounding of their
  inputs and outputs.
- **D5. The sums.** Each kernel sums in its own order (as b12x's kernels do, some with BF16
  atomics in a nondeterministic order); the CPU's and the GPU's expert kernels of the contract
  have the same order (a decode gives the same bits whatever the hot set). A mixed prompt group
  adds the GPU's and the CPU's partial sums in an order that depends on the plan's split.
- **D6. MTP.** A draft stays when it is the token the plain sampler picks at its row (the
  existing loop: the target's distribution, as standard rejection sampling of probabilistic
  drafts, with the plain decode's text under a seed); the MTP head's quantization uses IEEE
  reciprocals for the kernel's approximate ones, and its product sums in an order of its own
  (k_nv4_gemv; the drafts only); the drafts of a round end where the product of their
  probabilities falls under RECLANK_MTP_STOP 0.9 (the server's setting, as for EXL3), at most 3.
- **D7. Not arithmetic:** the serving mechanics of section 2 (prefix caching, chunk sizes, CUDA
  graphs, TP); the prompt rows of 64 tokens or fewer run A16 as vLLM's CUDA graphs of 64 (the
  serving source, P0's "uncertain" 1), larger ones A4.
- **D8. Another drafter (opt-in).** RECLANK_FP4L_MTP=<gguf> puts the MTP layer of that file
  over FP4-Local's (fp4local_st.attach_mtp: the BF12 GGUF's blk.48 taken verbatim into its own
  file, BF12 matrices and experts); the drafts keep the MTP head of section 2. Its experts run
  outside the contract (the combine weighs their rows, GP_MOE_COMBINE operand 12; the act is
  the plain one), each node holds a copy. Off by default: it drafts no better than FP4-Local's
  own layer (below).

Not taken: an int16-x form of A16 on the CPU (the exact FMA form costs 0.8% of decode); any
requantization of a stored tensor.

## 9. Status and measurements

All phases are done (commits of FP4-Local in the git log). From the weightkeeper (88.7 GB a node,
ready in about 5 minutes as the user; the model in 30 s), the 3090 and the Xeon (taskset 0-46,
22 part threads a node), real text:

| | |
|---|---|
| prefill 2K / 8K | 807 / 822 tok/s |
| decode 2K / 8K (105 hot experts a layer, context 8K) | 40.0 / 40.4 tok/s |
| the server (context 262144, 47 hot experts, the encoder lent) | decode 39.0 tok/s; an image of 192 tokens 0.72 s |

Quality (scripts as for EXL3: work/qfn/tf.py and gpu_steps.py on the float32 reference of the
original checkpoint, two prompts; decode-row KL over the reference's top 64):

| decode rows (93 + 156) | KL vs float BF16, short / long prompt | argmax |
|---|---|---|
| this runtime (the contract: A4 prompt over 64 tokens, A16 decode, MXFP8, E4M3 cache) | 3.7e-3 / 1.08e-2 | 98.9 / 98.1% |
| the FP4-Local reference (qfn_fp4l_reference.py: the contract in float32) | 3.9e-3 / 1.05e-2 | 98.9 / 99.4% |
| this runtime against that reference | 1.25e-3 / 5.1e-3 | 100 / 97.4% |
| turboderp EXL3 5.05 (tf.py, both prompts) | 2.3e-3 | 97.6% |

The distance from the original is the weight set's (NVIDIA's QAD), not the kernels': the runtime
is as far from BF16 as the contract's own reference is. The prompt rows of the QAD model are
further (KL 6.7e-2 against EXL3's 1.9e-2): its distillation trained on the teacher's answers.
MTP: greedy drafts give the plain decode's tokens (127 of 127); 1 / 3 drafts accepted 74% / 49-56%.

MTP drafters on a 1K-token prompt, 3 drafts (plain 40.8 tok/s), before the verify work below:

| drafter | accepted, fixed 3 | accepted, RECLANK_MTP_STOP 0.9 | tok/s (stop 0.9) |
|---|---|---|---|
| FP4-Local's MTP layer (NVFP4 experts) | 52% | 75% | 33.3-38.2 |
| the BF12 GGUF's MTP layer (D8) | 52% | 71% | 33.6 |

The drafter was not what held MTP back (the CPU drafter and the int16 cache gave the same 52-55%);
the verify was: a group cost 24.3 / 32.3 / 41.0 / 50.0 ms for 1-4 tokens. Then, each with the bits
of before (all 401 logit rows of steps and verify groups the same):

- k_mxf8_gemv_s: 2-8 tokens stage x in shared memory, rounded once (its time at 4 tokens 2.3 times
  a step's, now 1.4);
- k_kqh_nvx_ct: the scale bytes in shared memory (the serial sums read no global memory), a word
  of codes and a float4 of x a load, the E2M1 values in shared memory, 256 threads: 4.0 -> 2.6 ms a
  step;
- k_nv4_gemv (the drafts' NVFP4 head): 2.7 -> 0.59 ms (the generic kernel's 64-bit shift a code);
- the host: HotCache's scores of a verify group in one pass of counts, its copies without a scan of
  the pool for each call or np.unique, the routers' choices downloaded before the commit's kernels,
  the commit's addresses kept: 5.6 -> 5.1 ms of host work a round (the same copies, bit for bit);
- the server builds the verify and MTP programs at start.

A chat reply of 256 tokens (the same text plain and with drafts), from the weightkeeper:

| | tok/s |
|---|---|
| plain | 42.3 |
| 3 drafts, stop 0.9 (80% accepted) | 62-63 |
| 3 drafts, fixed | 59.7 |
| 2 drafts / 4 drafts, stop 0.9 | 61.5 / 57.4 |
| the server (262K context, 47 hot experts), two replies | 52.7, 56.2 (39.0 plain before) |

A verify token costs about 7.5 ms more (EXL3 5.05: 4.0): the router of the QAD model spreads its
picks (77% of a step's experts hot against EXL3's 84% with the same 100 slots a layer), and the cold
experts of a verify group are the CPU's memory bandwidth (about 118 GB/s a node in their phases).
RECLANK_GPU_HOT_INS 4 or 2 (fewer copies) gave nothing.

## 8. Open questions

- The rounding points of the real b12x A16 kernels (only the FP32 reference and the A4 header
  were read) and the QSA E4M3 cache details: P0.
- The CSF encoder (how a base is chosen) is not published; decoding does not need it.
- Whether the parts gain for NVFP4 as for EXL3 (each node its columns of every cold expert)
  or the A16 FMA form is memory-bound enough that one copy split by expert does as well: P6/P9.
- MTP on this machine: each verify token adds about 10 cold experts a layer on the CPU; the
  earlier NVFP4 path gained from 1-3 drafts (44.6 -> 58.7 tok/s), qfe 41.7 -> 59.5.
