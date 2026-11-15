# DeepSeek V4.1 prefill: what the Nemotron Super work offers

Nemotron Super's prompt groups went from ~377 to ~772 tok/s (docs/NEMOTRON_SUPER_PLAN.md section 10): the
experts of a group staged in GPU memory, a tiled expert kernel reading fragment-ordered weights, the dense
groups in one bfloat16 plane, attention on the tensor cores. This plan says what carries over to DeepSeek
V4.1 Flash (models/deepseek41) and in what order.

## Where DeepSeek's prefill stands

- A prompt runs in groups of up to 2048 rows (`GROUP`), each as two halves A and B so the CPU's experts of
  one half overlap the GPU's work on the other (gpu.py, `ab`).
- The routed experts: 15 hot slots a layer on the GPU (`k_ds_hot_tc`: int8 tensor cores, the E2M1 weights
  decoded straight from global memory), every cold pair on the CPU (the NUMA parts, `GP_CPU_TASK`).
- Measured: 4414 tokens in 12.1-12.9 s (342-365 tok/s) for groups of 1024 to 4096; the CPU parts are the
  bound (DEEPSEEK_V41_PLAN.md: compute-bound decode and plane sums, every cold expert read again by each
  group, up to ~7 GB a layer a pass).
- An expert is 18.8 MB (KQ_MXX), 384 a layer, 40 layers: 289 GB. The link moves ~12 GB/s (the pull kernel):
  1.6 ms an expert.

## What does not carry over

- **Staging every cold expert** (Super's path): a group of 2048 would move ~7 GB a layer, ~0.6 s a layer,
  far beyond the CPU's ~0.14 s. Zero-copy was rejected for the same reason (OVERLAP_PLAN.md).
- **Tensor-core attention**: DeepSeek's sparse attention reads a fixed window and a fixed count of selected
  entries per query, so its cost does not grow with the context as Super's did. Phase 0 checks its share;
  likely not worth a kernel.
- **The x1 fix of the dense groups**: DeepSeek's FP8 and BF12 matrices take float32 x (21 and 16 bits kept in
  two planes), not x already rounded, so no plane is zero. Only the `x8_prefill` measure mode has a zero plane
  (`k_quant_x2r` writes `ql = 0` and `k_ds_hot_tc` still runs its `cl` mma): off by default, a one-line skip.

## Phase 0: measure

The DeepSeek weightkeeper must hold the model (one keeper at a time: the Super keeper and server stop).
- Nsight Systems on a prompt of ~4K real tokens: the GPU kernels' sum per group against the wall, the CPU
  tasks' busy time (`rc_gpu_task_stats`), the GPU's idle time while the CPU runs a layer.
- Per layer of a group: the pairs per expert (the distribution: a group of 2048 has ~12K pairs over 384
  experts, mean ~32), and the CPU's time per expert against its pairs (the fixed decode of its 18.8 MB
  against the per-pair sums).

Exit: the GPU's idle share during the CPU's experts, and how much of the CPU's time the heaviest experts take.

## Phase 1: the GPU takes the heaviest cold experts (the main item)

Super's NM_PLAN in reverse. After a layer's router, a one-block GPU kernel counts the pairs of each cold
expert and picks the ones with the most pairs, as many as fit the time the CPU's share of the layer will take
(their copy at ~1.6 ms each plus their tiles, against the CPU's per-expert and per-pair costs, the constants
from Phase 0). A pull kernel in the program's stream copies those experts from the registered weightkeeper
segments into a staging buffer, chosen on the GPU, no host step; `k_ds_hot_tc` runs them through an address
table, as the hot slots; the CPU's task gets only the rest.

- In a layer of ~140 ms of CPU work the link can bring ~80 experts; if the heaviest quarter of the cold
  experts carries ~40% of the pairs, the CPU's work falls by a third or more: an estimate of +30-60% on
  prompts, to be confirmed by Phase 0.
- The staging buffer: ~80 x 18.8 MB = 1.5 GB (from the hot slots during a prompt; DSHot's grow_hot gives them
  back after, as Super's pool does).
- Correctness: the GPU's kernel and the CPU's give the same products to within their rounding (the two paths
  already share pairs today); a check on the logits of real prompts against the current split (KL, argmax),
  and the greedy text of check_dsv41_gpu_ref.py.
- Interplay with the halves A and B: each half plans its own layer (the CPU of one half, the GPU of the other).

## Phase 2: the hot-expert kernel (bit-equal)

Once the GPU takes more experts it is on the critical path more often. From Super's experience:
- the grid with an expert's row groups adjacent (its pairs' x from the L2 once; Super's down tiles 34 -> 24 ms);
- larger pair tiles per warp (32 rows x 16 pairs; the x of a stage through shared memory by cp.async) in
  place of 4 warps of 8 pairs over 16 rows;
- Nsight Compute first (occupancy, stall reasons), as for Super.

## Phase 3: the dense groups (needs the user's call on numerics)

`k_fp8b32_tc` runs tf32 mma with x as two tf32 planes (~21 bits). Two bfloat16 planes (~16 bits, the form of
the BF12 path) would halve the tensor-core time, at slightly less precise x (still far above the reference's
FP8 activations). A numeric change: only with the user's approval, measured by KL on real prompts. Bit-equal
alternative: larger tiles for `k_fp8b32_tc` (64 x 32 today), worth it only if Phase 0 shows the dense share
matters.

## Phase 4: small items

- `x8_prefill`: skip the `cl` mma when the low plane is zero (measure mode only).
- If Phase 0 shows attention above ~10% of a group: a tensor-core form of `k_ds_attn_g`.

## Order and checks

Phase 0, then 1 (the gain), then 2 (bit-equal), then 3 (with approval). Each phase: the logits of real
prompts against the step before (bit-equal where claimed), the greedy text check, the server's prompt speed
on 4K and 13K prompts.
