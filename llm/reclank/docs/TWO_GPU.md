# Two GPUs in one process: the RTX 3060 and the GTX 1070

The box: a Xeon W-2133 (6 cores, AVX-512 without VNNI, one NUMA node,
61 GB), an RTX 3060 12 GB (sm_86, power limited to 100 W because it sits
against the other card) and a GTX 1070 8 GB (sm_61, 150 W). The goal: run
one model on both cards at once, with the 3060 as the main GPU (the dense
weights, the cache, the hot experts) and the 1070 as more room for experts.

Status (branch `two-gpu`, October 2026): Gemma 4 26B, Nemotron 3.5 Bolt and
Qwen3.6 35B-A3B run on both cards. Nemotron gains the most, about 3x. Gemma
and Qwen gain little, because their steps are the 3060's own work.

## 1. Running it

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
    RECLANK_GPU_DEVICE=1 RECLANK_EXPERT_GPUS=0 ...

- `RECLANK_GPU_DEVICE=1`: the 3060 runs the steps (the primary).
- `RECLANK_EXPERT_GPUS=0`: the 1070 is an expert device (CUDA indices, as
  CUDA counts them under `CUDA_VISIBLE_DEVICES`; only the first is used).
- `RECLANK_EXPERT_GB`: the 1070's memory for experts (default: its free
  memory less 0.8 GB).
- `RECLANK_EXPERT_SKIP`: the experts the primary keeps, by rank (Gemma
  2000; Qwen: the primary's free memory at the load over the bytes of an
  expert; Nemotron: the hot set it picks itself).
- `RECLANK_EX_DEBUG=1` logs the device's group jobs and their waits;
  `=2` also synchronizes after each launch.
- The engines start the device on their own (Gemma `Model.load_all`,
  `NemotronGPU.__init__`, `QwenGPU.__init__`); the server needs only the
  variables.

Each card gets a build of its own arch (`libreclank_cuda_sm86_*`,
`libreclank_cuda_sm61_*`). The builds run in parallel with each other and
with the CPU library (`build.prebuild`, from `cops.py` when `RECLANK_GPU=1`
or `RECLANK_EXPERT_GPUS` is set): a load that builds all three went from
456 to 134 s.

## 2. The design: an expert device

The first try was an expert server (a second process for the 1070). It was
slow (a GPUProgram for each layer and count, a socket round trip a layer)
and ran out of memory. The design now is in-process (`backends/cuda/devices.py`,
`backends/cuda/csrc/runtime/expert_dev.cu`):

- A second copy of the CUDA library for the 1070's arch, loaded with
  `RTLD_LOCAL` (each copy links its own static cudart, so each has its
  own device, stream and buffers). When the primary already loaded the
  same file, the copy is a renamed file in `/dev/shm`.
- The driver keeps one current context for each thread, shared by both
  copies. Each entry point of the device (`rc_gpu_ex_*`) saves the
  thread's context and restores it (`ex_enter`, `ex_leave`). The
  primary's worker threads call `cudaSetDevice(gg_device)` first.
- The experts go up once (`rc_gpu_ex_store`, `_nm_store`, `_kq_store`).
  The CPU part of a step posts the cold experts that the device holds and
  waits for them through function pointers in an `er_ctx` (post, wait,
  post_t, the held mask): `GP_REMOTE_POST` (a step: the list form),
  `GP_NM_REMOTE_POST` (a group of up to 256 pairs: the pair form, -1 for
  the pairs it does not take) and `GP_REMOTE_JOIN` (adds the device's sum
  to the CPU's). The CPU computes its own cold experts in the meantime.
- GeForce cards have no peer access: a job crosses host memory (the rows
  of a layer each way, a few KB).
- Exclusive caching: the primary's HotCache leaves out the experts the
  device holds (`HotCache.exclude`). Shared caching lost speed at 4K
  (55.4 -> 52.6 tok/s): the two cards held the same experts.
- NUMA: each GPU's node comes from sysfs; the device's pinned buffers are
  made by a thread on its node.

On the device a job reuses the primary's kernels where it can: a record
(`gp_rec`) of the job through `gg_launch` (Nemotron's `GP_NM_HOT_MOE`;
Qwen's `GP_KQ_HOT_MOE`, with a program struct of zeros so that the float-x
kernels run, which read only the record). Gemma's jobs use their own small
kernels on `hot_rows_xr`.

## 3. Gemma 4 26B

The 3060 holds the dense weights, a cache of 256K positions and a HotCache
pool (about 8.7 GB); the 1070 holds 1840 experts (6.2 GB) after the first
2000. Decode (the 3060 at 100 W):

| | 4K | 32K | 64K | 128K |
|---|---|---|---|---|
| the 3060 alone | 55.4 | | | 26.4 |
| + the 1070, shared caching | 52.6 | | | 28.2 |
| + exclusive caching | 58.7 | | | 32.7 |
| + `k_attn_fdtc` at 2 blocks an SM | 59.1 | | | 34.5 |
| + `k_attn_fdtc` in tiles of 32 keys | 60.4 | 55.2 | 48.9 | 41.0 |

The global layers' attention (`k_attn_fdtc`, the keys-only cache) was the
largest part at long context: 2.31 ms a layer at 128K, now 1.53 ms (scripts
`bench_attn_fdtc.cu`, `KO=1`). Two causes: the turned (RoPE) values were
turned back in the value pass by warps 0 and 4 only, while the other six
waited; and the tiles of 64 keys of 56 blocks (3.5 MB) did not stay in the
3 MB L2 until the value pass read them again (126 MB from memory for 67 MB
of keys). Tiles of 32 keys fit; the turned values are made in the key pass
(each warp one block of pairs) into shared memory.

MTP (the q4_0 assistant, `RECLANK_MTP_STOP=0.9`, 256 tokens of a chat
prompt): plain 66.9 tok/s, 2 drafts 73.8 (the same tokens). The verify
group's cold pairs go to the 1070 too, which gave 73.1 -> 73.8 only: a
verify group of 3 is about 2x a step on the 3060's dense part, and the
CPU's int4 cold experts are cheap. A profile of the 3060 alone shows about
1 ms of a step waiting for the CPU.

The prompt (4K, both cards) runs at about 590 tok/s; the 1070 does not take
part in prompts.

## 4. Nemotron 3.5 Bolt

The routed experts are IQ2_XXS (about 17 GB), the hot set is fixed (a share
of each layer by the routing profile, `expert_profile.npz` beside the
model), and the CPU's IQ2 products run at only about 2.4 GB/s of weights.
So a step on the 3060 alone was nearly all the CPU's cold experts. The 1070
takes the next 45% of the experts (7.6 GB), most used for their bytes:

| decode | 4K | 32K |
|---|---|---|
| the 3060 alone (33% hot) | 7.9 | 6.5 |
| + the 1070 (45% more) | 26.1 | 20.9 |

Then zero copy: `NM_GROUP_MOE` runs every pair of a run on the GPU and reads
the cold experts straight from host memory the GPU maps (PCIe 3.0 x16,
about 8 GB/s here). It was the path of prompt groups of 16 rows and up; it
is now the path of steps and verify groups too (`RECLANK_NM_ZC_MIN`, default
1). With an expert device the 1070 runs its experts beside it: two
`NM_SPLIT` (its pairs, the rest), the CPU's task posts them, `NM_GROUP_MOE`
leaves them out (idx -1), `AWAIT` and `ADD` after.

| | split (CPU + 1070) | zero copy (+ 1070) |
|---|---|---|
| a step, the 3060 alone, 4K of the docs | 7.3 | 20.8 |
| a step, both, 4K of the docs | 26.1 | 24.0 |
| a step, both, a code prompt | 15.2 | 25.0 |

The split wins only on the text of the routing profile (the docs), where
the CPU's share is small; on other text zero copy wins by far, so it is the
default.

The LiLiCorr drafter (NVIDIA's DFlash drafter, a separate 1.1B checkpoint in
`~/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr`; 7 drafts, a
verify group of 8): its verify group was all CPU work, 8 tokens of cold
experts.

| 256 tokens, both cards | verify | drafter | over plain (15.2) |
|---|---|---|---|
| split (CPU + 1070) | 305 ms | 15.2 | 1.01x |
| zero copy, the 1070 idle | 148.5 ms | 29.7 | 1.95x |
| zero copy + the 1070 | 82.5 ms | 49.4 | 3.25x |

The 3060 alone (128 tokens): the verify group 1031 -> 144 ms; plain 5.3 ->
19.9 tok/s (zero-copy steps), the drafter 28.3. About 4 of 7 drafts are
taken a round.

Memory: the hot set leaves the memory of the prompt programs
(`NemotronGPU.prompt_bytes`, from the config: 853 MB for Bolt; measured 824
MB, 880 with their graphs) and `RECLANK_NM_MARGIN_GB` (0.9) for the rest. A
fixed 1 GB for all of it ran the 3060 out of memory in its first prompt.
The drafter loads after the hot set is sized: give it room with
`RECLANK_NM_MARGIN_GB=2.0`.

## 5. Qwen3.6 35B-A3B

UD-Q4_K_M, K-quant experts. Qwen had no routing profile: `scratch
qwen_profile.py` writes `expert_profile.npz` beside the model (HotCache's
uses of 256 decode steps over the docs and the code; sparse, but it ranks
the experts). `QwenGPU` loads it by default (also its first hot set: the
3060 alone 47.6 -> 51.5 tok/s). The 1070 holds 3987 experts (7.6 GB) after
the first 2787:

| 4K | prompt | decode |
|---|---|---|
| the 3060 alone | 615 | 51.5 |
| both | 619 | 54.0 |

Like Gemma, Qwen's step is the 3060's own work (attention, the DeltaNet
layers, the shared experts, the head): the CPU part was about 1 ms of 19.
Qwen's own zero copy (`RECLANK_GPU_ZC`) works only for KQ_NVX experts.

## 6. Bugs found on the way

- Nemotron ran past its cache with no check: `NM_KV` wrote the rows of a
  run past `max_len` into the next buffers on the GPU, and a later prompt's
  `NM_GROUP_MOE` read a broken address (an illegal memory access). It was
  first blamed on the 1070's group kernels: a run that synchronized after
  each job (or logged) did not fail, and the 64-token tests stayed inside
  the cache. `NemotronGPU._run` and `NemotronGPUSpec.round` now refuse a run
  past the cache. `prefill(ids)` goes on from the cache's n; a bench that
  means a new prompt passes `pos=0`.
- `GP_NM_REMOTE_POST` wrote the pairs of a group into a stack array of 256:
  Qwen's prompt groups of up to 64 tokens smashed the stack. A larger group
  now goes to the CPU whole.
- Nemotron's default margin ran the 3060 out of memory (above).
- nsys deadlocks on the two-card setup (the device's start in a thread,
  or its CUDA calls from the CPU's task, under CUPTI); a normal run does
  not. `CUDA_LAUNCH_BLOCKING=1` and compute-sanitizer also stall it: the
  3060's `k_await` waits for a CPU task that waits for the 1070, and traps
  after 60 s.

## 7. Other measurements of this work

The 1070 with 4 GB of ballast (a 4 GB card), Gemma 26B in Maxwell mode
(`RECLANK_GPU_CC=52`): 16K prompt 265 tok/s, decode 23.4; 32K 213 and 21.2
(regular mode at 32K: 233 and 22.3).

## 8. Open

- Gemma and Qwen: the 1070 could take dense work (layers), not only experts.
- The prompts do not use the 1070 (Gemma stages its cold experts; Nemotron
  and Qwen read them by zero copy or the CPU).
- Nemotron's verify group: the CPU idles while the 3060 reads the cold
  experts by zero copy; a share of them on the CPU would balance it.
- nsys with two cards (a switch to start the device on the main thread).
- Qwen's profile from more text, the prompts too.
