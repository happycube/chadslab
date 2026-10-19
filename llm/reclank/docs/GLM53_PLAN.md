# GLM-5.3-Flash in reclank (EXL3 4.05 bpw, DSpark)

The goal: serve zai-org/GLM-5.3-Flash on the 2-socket Xeon and the 3090 as
Qwen3.8-Flash-Next runs today (M=qfe): turboderp's EXL3 quant from the
weightkeeper, the dense part and the hot experts on the GPU, the cold experts
on the CPU in parts (each node computes its columns of every cold expert from
its own copy), and speculative decoding with RedHatAI's DSpark drafter.

## 1. The model

GLM-5.3-Flash (glm5_next, transformers 5.19:
models/glm5_next/modeling_glm5_next.py) is a 45-layer hybrid, hidden 4096:

| Part | Layers | What |
|---|---|---|
| KDA linear attention | 34 (all but 3, 7, ..., 43) | Kimi Delta Attention: qkv 4096 -> 3 x 64 x 128, short conv 4 + silu, l2norm q/k, a decay for each channel, beta per head, state 64 x 128 x 128 fp32, per-head RMSNorm x sigmoid gate |
| DSA attention | 11 (3, 7, ..., 43) | MLA without RoPE (q_lora 1536, kv_lora 512, 64 heads of 256), the 512 latent the only KV; an indexer (32 heads x 128) over keys pooled by 4, top 512 pools + the open tail (2051 keys at most) |
| Hyper-connections | all | mHC as DeepSeek V4: 4 streams, Sinkhorn (20 iterations) of the comb matrix; the head takes the mean of the streams |
| MoE | 3-44 | 288 experts, top 8, sigmoid + correction bias, normalized, x 2.5; inner 2048; one shared expert (ungated); SwiGLU clamped: silu(min(g, 10)) * clamp(u, -10, 10) |
| Dense MLP | 0-2 | the same clamped SwiGLU, inner 12288 |
| Head | | untied lm_head (154880), embed bf16 (1.27 GB, on the host) |

No RoPE in the text model at all (the indexer flag indexer_rope_interleave is
dead). The tokenizer is byte-level BPE of the LFM2 kind (models/qwen/
tokenizer.py handles its split pattern and ignore_merges). EOS: 154820,
154827, 154829. The chat template always opens with
`[gMASK]<sop><|system|>Reasoning Effort: Max` (Low/High override it) and the
generation prompt ends with `<|assistant|><think>`.

## 2. The weights

- **The model:** turboderp/GLM-5.3-Flash-exl3, branch 4.05bpw
  (/home/cpage/models/GLM-5.3-Flash-exl3-4.05bpw, 165 GB). MUL1 codebook.
  Routed experts K=4 (152.6 GB, 12.6 MB an expert); qkv_proj, o_proj, shared
  experts and lm_head 6 bits; dense MLP 5 bits; q_a/q_b/kv_a/indexer wq_b
  EXL3. In f16/f32 as they are (about 0.9 GB): kv_b_proj, the router, the
  KDA low-rank pairs f_a/f_b and g_a/g_b, b_proj, conv1d, A_log, dt_bias,
  the indexer wk/compress_gate/weights_proj/ape, the hc fn/base/scale.
  The MTP layer (layer 45) is in mtp.safetensors, outside the index: EXL3
  (eh_proj K=5, -18 dB against the FP8 original; the experts K=4), kv_b and
  the indexer's small tensors in f16, no hyper-connections. An alternative
  drafter to DSpark (not planned; P7 stays DSpark).
  The repo's chat template differs from the original (text only, past
  reasoning dropped unless clear_thinking=false).
- **The drafter:** RedHatAI/GLM-5.3-Flash-speculator.dspark-preview
  (/home/cpage/models/GLM-5.3-Flash-speculator.dspark-preview, 5.1 GB bf16):
  5 Qwen3-style layers (64 x 64 heads of 64, RoPE theta 1e4, sliding window
  2048, MLP 12288), fc 6 x 4096 -> 4096 over the hidden states of target
  layers 20, 28, 32, 36, 40, 44, block 8, Markov head rank 256, confidence
  head; its own embed_tokens and lm_head (1.27 GB each). A preview (end of
  epoch 2 of 3). Its bfloat16 matrices go to BF12 on the GPU (the bfloat16
  values in 12.25 bits, as Qwen3.8's dense part: 3.4 GB in place of 4.5).
- **The reference:** the original FP8 checkpoint (/space/models/GLM-5.3-Flash,
  306 GB, E4M3 128 x 128 blocks; the KDA projections and kv_b in bf16).

No weight of the model is converted beyond what the runtime already does to
EXL3 (the stacks of experts, the GGUF view); its f16/f32 tensors stay as
stored. The drafter's matrices are BF12.

## 3. Memory

**Host (2 x 192 GB):** the experts on each node, 152.6 GB a node (79%: the
weightkeeper with --copies both, since auto picks split above COPY_SHARE
0.45), page-locked since commit 5408506 (305 GB pinned). About 39 GB a node
left for the server (dense host copies, embed, the CPU parts' scratch) and
the OS. One weightkeeper at a time: the Qwen3.8 EXL3 one stops first.

**GPU (24 GB, about 23.3 usable):**

| What | GB |
|---|---|
| CUDA context and reserve | 0.8 |
| Dense EXL3 (qkv 2.57, o 1.41, shared 0.79, MLA/indexer 0.33, dense MLP 0.28) | 5.4 |
| lm_head (6 bits) | 0.48 |
| f16/f32 tensors (kv_b 0.37, hc fn 0.14, router 0.1, KDA pairs 0.21, ...) | 0.88 |
| DSA cache, 128K tokens (12 KB a token: latent 1 KB + pooled index keys 64 B a layer) | 1.6 |
| KDA state (136 MiB) + conv (10 MB) | 0.15 |
| DSpark in BF12 (layers 2.18 + fc 0.15 + Markov 0.12 + lm_head 0.97; embed rows from the host) | 3.4 |
| Programs and scratch (4096-row groups) | 2-4.5 |
| **Left for hot experts** | **about 7-9 (580-740 experts, 14-17 a layer of 288)** |

DSpark takes much of what Qwen gives to hot experts (BF12 saves 1.1 GB of
its 4.5 in bfloat16). If the drafter's embed_tokens equals the target's
(check), it reads the target's rows on the host; its lm_head is its own.
A 256K context adds 1.6 GB more cache.

## 4. What exists and what is new

Most parts are close to something reclank runs already:

| Part | Reuse | New work |
|---|---|---|
| EXL3 products (CPU VNNI, GPU GEMV/MMA, groups, int16 x) | all shapes fit (every dimension a multiple of 128) | none |
| Expert stacks, HotCache, CPU parts, NUMA copies, page-locked keeper | gpu35.py, moe_part.c, exl3_moe.c (generic in E and sizes) | budget: MIX_RING 192 -> about 64 (12.6 MB experts), the divisor 42 MoE layers not 45 |
| KDA | Gated DeltaNet (P.GDN: deltanet.c, k_gdn_heads, k_gdn_scan; the sigmoid gate of Qwen3.8, flag bit 2) | the decay for each channel (a flag: t x 8192 decays, dt_bias [8192], g = -5 sigmoid(exp(A_log) (f + dt_bias))); k_gdn_scan's shared memory (GDN_TC 16); the verify log per channel; the low-rank f/g pairs |
| MLA | DeepSeek V4.1's absorbed MQA over a 512 latent (k_ds_attn, k_ds_attn_g; 64 heads, keys from a sel list of up to 2176) | the grouped kv_b products (W_UK^T q and W_UV o: 64 groups, f16), a GLM mode of k_ds_attn (scale 1/16, no sink, no RoPE, no window) |
| Indexer | V4.1's k_ds_compress (softmax pooling with open-group state), k_ds_idx_score (H 32, D 128, relu), k_ds_topk; QSA's expand-blocks-and-tail | ratio 4 + ape, LayerNorm with bias on k, no post norm, k 512, expand x 4 + tail, a bf16 pooled-key cache |
| mHC | V4.1 identical math (ds_hc_coef, k_ds_hc_mix/pre/post, k_ds_stream_mean) | no lag (the coefficients of the current streams), layer 0 its own pre, the head a plain mean |
| Router | k_router_topk (HotCache hooks); k_ds_router has bias-for-choice and normalize x scale | a sigmoid + bias + scale mode in ROUTER_TOPK (GPU and interp.c) |
| SwiGLU clamp | k_ds_swiglu, moe_part.c flag 128 | the clamp in the EXL3 act sites (exl3_moe.c, moe_part.c, k_kqh_act, qmoe_tc) and the dense MLP; shared expert weight 1 (no sigmoid(slog)) |
| DSpark | V4.1's dspark.py loop, verify length, accept/restore, DS_MARKOV/DS_CONF; MiniCPM's reference math | a dense Qwen3 drafter program (64 x 64 heads, ring K/V of 2048, block 8), the taps, VERIFY_MAX 8 -> 9 |
| Tokenizer, template, tools | QwenTokenizer (split_pattern, ignore_merges) | GLM's template and its XML tool calls (`<tool_call>name<arg_key>..</arg_key><arg_value>..</arg_value>`) |

## 5. Phases

Each phase ends with a check; the numbers are KL against the reference on
decode rows (prompt rows are noisy: routing chaos), as for Qwen3.8.

**P0. The reference** (scripts/glm53_reference.py). transformers' glm5_next
in float32 on the CPU (dsref-venv), the weights of the FP8 checkpoint
dequantized as they are read, the routed experts through an LRU (as
qwen38_flash_reference.py). Golden dumps (~/reclank-golden/glm53): ids, gen,
the top 64 and lse of each row, the full logits of the last prompt row and
the first 16 steps, and the hidden states after each layer of a short prompt
(for the layer checks); --forced for teacher-forced texts. Run while no
weightkeeper holds memory (the LRU wants about 150 GB).
Check: tokenizer against HF (scripts/check_qwen_tok.py), template rendering.

**P1. The weights** (reclank/formats/exl3_glm53.py, GLM53Source: the GGUF view
of the EXL3 directory, as EXL3Source for Qwen). Names for the KDA, MLA,
indexer, hc and MoE tensors; the 288-expert stacks on huge pages; kv_b split
into W_UK (transposed at load) and W_UV; no "+1" on norms, no -exp(A_log)
rewrite. Verify the fused qkv_proj row order (q | k | v) and kv_b's per-head
layout against the original checkpoint's q/k/v_proj and kv_b (decode both,
compare). Check: every tensor of the index mapped, shapes; EXL3 products of
a few matrices against the FP8 originals (dequantized), dB.

**P2. The NumPy model** (reclank/models/glm53/model.py, GLM53CPU): the layers
in NumPy and the existing C products (EXL3 linear, the EXL3 MoE of the CPU),
the slow reference of the runtime: KDA (recurrent form, the decay for each
channel), MLA absorbed (and the plain form, to compare), the indexer with its
pools, mHC (from deepseek41/model.py without the lag), the router and the
clamped MoE. Check: hidden states after each layer against P0's dumps (a
short prompt), then decode-row KL of the golden prompts (the EXL3 error:
expect about 1e-2 at 4 bits).

**P3. The CPU ops** that the GPU runtime needs on the host: the router mode in
interp.c, the clamp in exl3_moe.c and moe_part.c (EXL3 branches), shared
weight 1. Check: check_moe_parts.py --exl3 extended to GLM's experts (the
parts against the whole, bit for bit), the MoE of P2 with the ops.

**P4. The GPU kernels**, each with a check script against NumPy:
1. KDA: the channel decay in k_gdn_heads (steps) and k_gdn_scan (prompts),
   the verify log and its commit; check_gdn-style against P2 (one layer,
   steps and a prompt of 300 tokens, state after).
2. MLA: grouped kv_b products (f16, GEMV and tensor cores), k_ds_attn GLM
   mode (steps, verify groups, prompt groups via k_ds_attn_g).
3. The indexer: pooling with ape, scores, top 512, expansion + tail; against
   P2 at 1K, 3K and 10K tokens (the selected sets; ties: see 7).
4. mHC without lag, the stream mean of the head.
5. The router mode, the clamp in the EXL3 act kernels (k_kqh_act, qmoe_tc,
   the dense MLP), shared weight 1.

**P5. The GPU model** (reclank/models/glm53/gpu.py, GLM53GPU on the QwenGPU
machinery of gpu35.py: programs for a step, verify groups and prompt
groups, HotCache with 288 experts, the mixed groups with the CPU, parts from
the weightkeeper). Check: decode-row KL against P0 (target: the EXL3 error of
P2, about the same); prefill 2K/8K and decode at 8K (pre.py, prof.py).

**P6. The weightkeeper** (reclank/models/glm53/keep_exl3.py, from Qwen's
keep_exl3.py: the layout and the fill of EXL3 modules; GLM's tensor set).
--copies both. Check: the server from the keeper in seconds, the segments
page-locked, the parts reading each node's copy in place.

**P7. DSpark** (reclank/models/glm53/dspark.py + its GPU program). First a
reference replay (scripts/glm53_dspark_reference.py, as
minicpm_dspark_reference.py) on P0's dumps to settle what the code leaves
open: the taps (input or output of the layer; the 4 streams collapsed by the
mean, as fc's 6 x 4096 says, with or without a norm), the positions, the
non-causal block. Then the GPU drafter: 5 dense layers, ring K/V of 2048,
the fc of the taps, the Markov and confidence heads, its matrices in BF12
(the bf12_tc products of Qwen3.8's dense part); the verify loop of V4.1
(VERIFY_MAX 9). Check: the BF12 drafter against the bfloat16 replay
(acceptance the same within noise), acceptance on real chat text, greedy
output equal to plain decode.

**P8. The server** (scripts/serve_glm53.py or the Qwen engine with a GLM
model): the original template (or the EXL3 repo's, chosen deliberately),
reasoning effort (default Max as the template; a server option), the XML
tool calls, stop ids; start_server.sh M=glm. Check: the dsh client end to
end, tools, long context.

**Later:** the vision tower (564M params, 0.6 GB: deferred); the EXL3 MTP
layer as a drafter (mtp.safetensors: only if asked); a chunked KDA prompt
kernel; an FP8 latent cache.

## 6. Speed

Measured (the weightkeeper with both copies, 19 hot experts a layer, OMP_NUM_THREADS=1,
RECLANK_GPU_PARTS=1, RECLANK_GPU_CPU_THREADS=36 on taskset -c 0-19,24-43; without the CPU team
the mixed groups ran their CPU experts on one thread: 5 tok/s):

| | 2048 | 8192 |
|---|---|---|
| prefill | 192 tok/s | 151 tok/s |
| decode | 12.9 tok/s | 10.6 tok/s |

Decode is the CPU's EXL3 products: about 315 cold experts a step (7.9 G weights) at about 105 G
weights/s over both nodes (21 part threads a node: 13.2). KL against the reference on the 256
decode rows of the trace 9.7e-3 (argmax 98.0%).

After the first optimizations (the attention of a step split over its keys, the mHC of a step
in GLM_HC_MIX / GLM_HC_POST): decode 16.9 tok/s at 2048, 16.4 at 8192; prefill 161 tok/s at 8192.

The CPU experts (scripts/bench_glm53_cpu_moe.py, real rows, every expert cold): a step 54.8 ms
for the 42 layers (1.30 ms a layer, 154 G weights/s), the time follows the unique experts (a verify
group of 9 tokens: 47 of them a layer, 443 ms). The EXL3 kernel is bound by port 5 (the vpermw of
the codes: 5.7 of its 10 vector uops a group of 16 weights). Measured levers, decode at 2048:

| | 18 part threads a node | 21 |
|---|---|---|
| int16 x (RECLANK_MOE_X16=1, the default) | 16.8 tok/s | 18.0 |
| int8 x (RECLANK_MOE_X16=0) | 18.4 | 19.4 |

int8 x: decode-row KL 1.04e-2 (int16: 9.7e-3 to 1.27e-2 over runs), prompt rows 9.8e-2 against
9.4e-2. Huge pages for the experts (the weightkeeper under sudo, or shmem THP): 4.5% faster in
the parts. GLM53GPU sets RECLANK_PART_DYN_MIN=3 (the dynamic split of the parts from 3 tokens:
verify groups of 3, 5, 9 tokens 4%, 11%, 16% faster).

DSpark on the GPU (models/glm53/dspark.py, scripts/check_glm53_dspark.py): a draft 8.5 ms, 2.3 to
2.6 of 8 drafts accepted (the FP8 replay: 3.0); the verify groups cost about 40 to 50 ms a token
(each token adds about 8 experts a layer on the CPU), so the chosen lengths give 16.9 to 17.7
tok/s against 18.7 plain: no gain on this machine.

DSpark replay on the trace (scripts/glm53_dspark_reference.py): the taps are the streams entering
the target layers (the output of layers 19, 27, ...), collapsed by their mean: 4.02 tokens a verify
step, the first draft 91% (the output of the target layers: 1.60).

The estimates before:

- Decode: 8.45 G expert weights a token; the CPU MUL1 product about
  100-114 G weights/s on 24 cores, two nodes in parts: about 40 ms of cold
  experts a step before the hot hits (14-17 of 288 a layer on the GPU), so
  about 20-25 tok/s plain; DSpark at Qwen-like acceptance (about 80%) maybe
  1.4-1.5x.
- Prefill: a group touches nearly all 288 experts of each layer, 153 GB a
  pass (2x Qwen3.8), bound by the PCIe copies of the cold experts and the
  CPU; KDA prompt work 1.26x Qwen3.8's GDN.

Then: the MLA of a prompt group on the tensor cores (GLM_ATTN, k_glm_attn_tc: q and P float16,
-78 to -82 dB against NumPy; k_ds_attn_g took 510 ms a layer of 4096 rows at 8K), the mHC
coefficients with a lane for each entry of comb, and the small float16 products of a step fused (KDA:
f_a, g_a, b_proj in one product, f_b and g_b in one grouped product; the indexer's wk, gate, weights
in one): prefill 287 tok/s at 2048 and 332 at 8192 (197 and 226 before), decode 21.3 at 2048 and
19.2 at 8192; decode-row KL 8.3e-3.

The routing profile (scripts/glm53_expert_profile.py: the decode steps of 12 files of the repo,
<model>/expert_profile.npy, which GLM53GPU reads): on a held-out text the hot experts hit 38.8% of
the picks in the first steps (30.9% without), decode 19.7 tok/s against 18.1 there; after 128 steps
HotCache has the same set either way (21.4 / 21.5 tok/s). More hot slots (the program reserve 3 or
2 GB: 22, 23 a layer) and RECLANK_GPU_WARM=free: no gain.

The launch layout with only the system's cores held out (taskset -c 0-46, 22 part threads a node,
RECLANK_GPU_CPU_THREADS=44; the helpers on 22, 23): decode 22.3 / 22.2 tok/s at 2048 against 21.6 /
21.5 with 21 part threads on 0-21,24-45 (+3.5%).

## 6b. RQ8_0 experts from the FP8 checkpoint (a feasibility study)

scripts/study_glm53_rq8.py, on real rows of the trace (the experts their routing uses):

| | RQ8_0 | Q8_0 | EXL3 4.05 (K=4) |
|---|---|---|---|
| expert matrices against FP8 | -45.4 dB | -45.1 | -22.6 to -23.0 |
| the routed experts' output, layer 3 / 20 / 44 | -46.2 / -43.1 / -44.1 dB | -45.1 / -42.5 / -43.3 | -28.0 / -21.5 / -24.1 |
| the CPU parts, a step (1 token), a layer | 1287 us | (the same kernels) | 1047 us |
| a verify group of 9 tokens, a layer | 9630 us | | 7890 us |

(int8 x, 21 part threads a node; Q8_0 runs at 166-171 GB/s over the two nodes: the memory
bandwidth, so the gap stays.)

- Quality: the experts about 20 dB closer to the reference (the dense matrices stay EXL3, 6 bits
  for the attention: -33 dB). The KL of the decode rows would fall from 1e-2 by an amount only a
  build measures (the dense error remains).
- Decode: the CPU part of a step 54 ms against 44 (EXL3, the new defaults): about 16.5 tok/s
  against 19.8. DSpark gains nothing more (the cost still follows the unique experts).
- Prefill: the copies of the mixed groups carry 2.1 times the bytes (26.7 MB an expert against
  12.6); Qwen3.8 lost 30% at 8192 (RQ8_0 633 tok/s against NVFP4 910): about 150-170 tok/s
  against 226. Hot experts: about 9 a layer in place of 19 (they barely change decode: 1 slot
  17.5 tok/s, 13 slots 17.7-18.0).
- Memory: 324 GB of experts (12096 x 25.2 M weights x 34 / 32 bytes): one copy, each node the rows
  of its part (162 GB a node, the reads local, as DeepSeek V4.1's weightkeeper; no --copies both),
  about 26 GB a node left for the server, the pinned buffers and the system.
- Conversion: the FP8 shards (306 GB on the HDD, about 220 MB/s: 23 min of reads), each matrix
  E4M3 x its scales, the TQ6 rotation, Q8_0; a GGUF of the experts on the NVMe (324 GB of its ~413
  free) for the weightkeeper (a start in 2-3 min), or the weightkeeper converting at each start from
  the HDD (25 min or more).
- Work: a converter (scripts/convert_q8_gguf.py's form for the FP8 shards), the weightkeeper's GLM
  layout of RQ8_0 rows by part, GLM53Prog with RQ8_0 stacks (rot_experts: the program rotates x and
  the kernels the act, as Qwen4CPU), the GPU hot experts in Q8 (KQ_HOT_MOE takes Q8_0).

Verdict: feasible (memory tight, conversion half an hour), the experts much nearer the
reference, at about 17% of the decode speed and 25-35% of the prefill.

## 7. Risks and open questions

1. The channel decay touches every GDN kernel (steps, scan, verify log); the
   prompt path stays a sequential scan.
2. The grouped kv_b kernel is new; kv_b at f16 is 352 MB of reads a token.
3. Top-512 ties against torch.topk past about 2052 tokens (relu gives many
   zero scores): compare selected sets by score, not by index.
4. Host memory: 305 GB pinned, 39 GB a node left; hot-expert hit rates with
   only 14-17 experts a layer on the GPU (DSpark's 3.4 GB is the trade).
5. The qkv_proj row order and the DSpark taps are unverified until P1/P7.
6. The DSpark checkpoint is a preview; acceptance unknown.
7. The template: Reasoning Effort Max by default, no thinking-off switch.
