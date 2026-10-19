# Two GPUs in one process: the RTX 3060 and the GTX 1070

The box: a Xeon W-2133 (6 cores, AVX-512 without VNNI, one NUMA node,
61 GB), an RTX 3060 12 GB (sm_86, power limited to 100 W because it sits
against the other card) and a GTX 1070 8 GB (sm_61, 150 W). The goal: run
one model on both cards at once, with the 3060 as the main GPU (the dense
weights, the cache, the hot experts) and the 1070 as more room for experts.

Status (branch `two-gpu`, October 2026): Gemma 4 26B, Nemotron 3.5 Bolt and
Qwen3.6 35B-A3B run on both cards. Nemotron gains the most, about 3x. Gemma
and Qwen gain little, because their steps are the 3060's own work. On branch
`nm-split` (section 7) Nemotron's steps run its zero-copy experts on the
step's kernels and its 1070 follows the conversation: decode 26.6 -> 41.2
tok/s on the docs.

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

## 6. The three models on the merged tree (c5a49e4)

The same scripts after the merge of upstream (the IQ2 kernels on AVX2, the
signal that waits for the last CPU task, DeepSeek V4.1): decode in windows
of 32 steps after the prompt, the RTX 3060 at 100 W, a cache of 256K
positions for Gemma.

Gemma 4 26B (`scratch gemma_decode_after.py`):

| | prompt | steps 0-31 | 32-63 | 64-95 |
|---|---|---|---|---|
| 4K, the 3060 alone | (292: the build) | 37.7 | 50.9 | 59.8 |
| 4K, both | 669 | 43.7 | 64.4 | 70.0 |
| 32K, the 3060 alone | 663 | 35.0 | 44.2 | 50.4 |
| 32K, both | 761 | 39.5 | 58.4 | 63.7 |

Qwen3.6 35B-A3B (`scratch qwen_decode_after.py`):

| | prompt | steps 0-31 | 32-63 | 64-95 |
|---|---|---|---|---|
| 4K, the 3060 alone | 621 | 49.0 | 52.8 | 53.5 |
| 4K, both | 625 | 51.1 | 56.0 | 57.0 |
| 32K, the 3060 alone | 749 | 35.8 | 41.6 | 40.9 |
| 32K, both | 750 | 39.5 | 45.5 | 45.6 |

Nemotron 3.5 Bolt, before (c1edb6e) and after the merge:

| | before | after |
|---|---|---|
| both, plain decode (code prompt, 128 tokens) | 25.8 | 26.7 |
| both, the LiLiCorr drafter | 39.2 (1.52x) | 48.3 (1.81x) |
| the 3060 alone, prompt 4K | 276 | 287 |
| the 3060 alone, decode after 4K | 21.9 | 21.9 |

The drafter's draft program varies from run to run (12 to 32 ms a round,
on either tree), so its speedup swings with it; the verify group is 75-77
ms on both.

## 7. Nemotron: where the second card's work goes (branch `nm-split`)

A step on both cards (nsys, a 1K prompt of the docs, 39.0 ms a step): the
3060 was busy 37.4 ms of it, the 1070 2.5 ms. 15.9 ms of the 3060's was
`k_nm_gtile_tc`, the tiles of `NM_GROUP_MOE` on a single token. The zero
copy itself was small: the routing profile has 4% of the pairs on no GPU.

### The routing of real chats

`RECLANK_NM_ROUTES=1` makes every program count each layer's choices
(`GP_COUNT`; `NemotronGPU.route_counts()`). Four chats of two turns of 256
tokens each (`scratch nemotron_routes.py`), the share of a step's pairs:

| chat | the 3060 (34% of the experts) | the 1070 (45%) | no GPU | zero copy a step |
|---|---|---|---|---|
| code | 0.45-0.48 | 0.42-0.43 | 0.11-0.13 | 49-58 MB |
| a story | 0.31 | 0.49-0.51 | 0.19 | 85-89 MB |
| a summary of this doc | 0.55 | 0.39 | 0.06 | 27-29 MB |
| a hash map, explained | 0.42-0.43 | 0.46 | 0.12 | 54-56 MB |

The profile predicted 63% on the 3060, 33% on the 1070 and 4% on no GPU;
the fixed sets hold less of real chats. Offline, on these counts: a 1070
set picked by the chat's own first turn (the downtime after it, the next
prompt not known yet) takes the second turn's no-GPU pairs from 12.5, 18.5,
5.9 and 12.1% to 6.3, 4.6, 3.2 and 5.0%. Another chat's routing does not
help (13.2, 18.2, 7.6, 10.5%). A pick within each layer's own count of
slots does as well as one across layers, so the 1070's stacks keep their
sizes; re-picking the 3060's set too adds little.

### The step's kernels on the zero-copy tables

`NM_GROUP_MOE`'s grid is every expert by every tile of 32 rows (240-496
experts by 32-80 tiles): for one token, 7,680 to 35,712 blocks of 256
threads for about 6 live experts, 100-380 us a matrix whatever the bytes.
`NM_HOT_MOE` now takes the tables instead of slots (operand 22): each
expert's address, a slot on the GPU or the host memory the GPU maps, and
every pair of idx >= 0 runs (the 1070's pairs are -1). So the step's
kernels (`k_nm_hot_up8`/`down8`: a warp on 4 rows of a pair;
`k_nm_hg_*` for 2-8 rows) read the cold experts straight from the host.
`RECLANK_NM_ZC_ROWS` (default 8; 0: the tiles for all).

| | the tiles | the tables |
|---|---|---|
| both cards, decode after 1K of the docs | 26.6 | 39.3 |
| the 3060 alone, decode after 4K of the docs | 21.9 | 24.9 |
| both, a run of 4 rows | 56.5 ms | 45 ms |
| both, a verify group of 8 rows | 71.8 ms | 62.4 ms |
| the 3060's routed-expert kernels a step | 15.9 ms | 3.8 ms |

The logits: KL at most 1.2e-3 from the tiles (the same products, summed in
another order), the same argmax.

### The shared expert during the wait

The 3060 then waited 2.47 ms a step for the 1070 (`k_await`): a layer's
round trip (the signal, the CPU's task, the post, the 1070's kernels, the
copy back, the flag; about 150 us) outlasts the 3060's own experts.
`NmEmit.moe` now returns the wait and the sum, and `compile_nemotron_step`
emits them after the shared expert, which does not need the routed ones:
39.3 -> 41.2 tok/s, the same logits bit for bit.

### The 1070's experts by the conversation (the downtime between prompts)

With an expert device the programs count the routing (`GP_COUNT`).
`NemotronGPU.refresh_experts()` re-picks the 1070's experts of each layer
by the routing since its last call (the routing before weighs half for
each 128 tokens since, `RECLANK_EXPERT_REFRESH_HALF`; the profile breaks the
ties), copies those that come into the slots of those that leave
(`rc_gpu_ex_nm_put`, from the pinned host stacks), and updates the CPU's
held table and the 3060's `NM_SPLIT` tables. The server calls it when its
job queue is empty (`ModelWorker` -> `engine.idle()`), so it runs after an
answer is out, not before its end. `RECLANK_EXPERT_REFRESH=0` keeps the
profile's set.

Three chats of two turns (`scratch nemotron_refresh.py`), the second turns,
decode tok/s (the pairs no GPU holds); before the shared expert's overlap:

| chat | no refresh | old routing at half a turn | at a quarter | none (only the last turn) |
|---|---|---|---|---|
| code | 34.3 (12.4%) | 37.9 (6.3%) | 38.0 (6.3%) | 37.9 (6.4%) |
| a story | 30.5 (18.7%) | 36.8 (7.4%) | 36.9 (7.3%) | 38.6 (3.7%) |
| a hash map | 34.6 (11.1%) | 36.6 (7.5%) | 37.6 (5.9%) | 37.7 (6.2%) |

The decay by tokens (half a turn of 128 tokens) gives about the last
column on full turns, without letting a short answer undo the turns before
it. A refresh copies 400-1300 experts (0.5-1.6 GB) in 0.07-0.22 s. The
first turn of a new topic gains nothing.

Through the server (both cards, the story chat, 200 tokens a turn, with the
shared expert's overlap): the first turn 31.2 tok/s, then the refresh in
the idle (1390 experts, 1.79 GB, 0.24 s), the second 39.9 tok/s.

The refresh is the expert device's own now (`devices._refresh`,
`rc_gpu_ex_put` for the three stores), so Qwen and Gemma have it too
(`devices.refresh_hot`): their HotCache counts every selection of the
steps, the expert device's too (`HotCache.routes`), and the prompts'
`GP_COUNT`; the idle re-picks the expert device's experts among those the
3060's slots do not hold or take, and `HotCache.exclude` follows.
`RECLANK_EXPERT_REFRESH=0` turns it off for all three. Through the server,
the same three chats of two turns (256 tokens, decode tok/s; the pairs no
GPU held):

| second turn | Qwen, no refresh | Qwen, refresh | Gemma, no refresh | Gemma, refresh |
|---|---|---|---|---|
| code | 52.3 (10.1%) | 52.1 (8.5%) | 72.3 (0.0%) | 72.9 (0.0%) |
| a story | 52.1 (4.0%) | 55.3 (4.4%) | 68.7 (0.0%) | 70.4 (0.0%) |
| a hash map | 53.9 (7.7%) | 54.3 (5.4%) | 73.1 (0.0%) | 74.2 (0.0%) |

MTP on the same tree (256 tokens, `RECLANK_MTP_STOP=0.9`, tok/s):

| | plain | 1 draft | 2 drafts | 3 drafts |
|---|---|---|---|---|
| Gemma 26B, both cards (the q4_0 assistant) | 76.5 | 84.7 | 88.3 | 85.1 |
| Qwen3.6, both cards (its MTP layer) | 59.5 | 56.6 | 63.7 | 64.0 |
| Qwen3.6, the 3060 alone | 61.7 | 64.7 | 63.3 | 61.7 |

On a short prompt the 1070 costs Qwen a little (plain and one draft): its
round trip outlasts the CPU's few cold experts; after 4K it gains (57.0
against 53.5, section 6).

Little for either: their 3060 slots already follow the text step by step
(HotCache), so the 1070 holds a tail its first set covered about as well,
and Gemma's experts fit the two cards (0.0-0.2% of the pairs on the CPU).
Nor do they need the overlap: Qwen's shared expert is part of
`KQ_HOT_MOE`, the GPU's part between the signal and the wait, and
Gemma's step (`forms.py`) emits the router and the experts before the
dense MLP.

### The drafter

The LiLiCorr drafter, 256 tokens of the code prompt (both cards; three
runs): plain decode 33.0 tok/s, the drafter 51.7-51.9 (1.57x; draft 12.3
ms and verify 79 ms a round, 3.9 of 7 drafts taken). The tokens part from plain greedy at token 185 (at 99-120 before:
a verify group's dense kernels are not a step's).

A verify group of 8 rows (62 ms on the docs) is another balance than a
step: the 3060 is busy 61.4 ms (the 8-row dense products 19.8, the routed
experts 21.9 with their zero copy, `k_await` 7.9), the 1070 24.4 ms (its
`k_nm_hg_up` 14.1, `k_nm_hg_down` 9.0: the IQ2_XXS decode is compute bound
on Pascal). Here the 1070 limits, not the 3060.

A step now (nsys, both cards, 26.2 ms under the profiler): the 3060 is
busy 25.7 ms: the NVFP4 matrices 8.3 (`k_nv4_gemv2`), the Q8_0 ones 4.4,
the shared experts 2.2, the routed experts 3.8 (about 3.2 of it their zero
copy), `k_await` 1.5; the 1070 2.5 ms. A step is the 3060's dense work
now; the 1070 has room for more experts (more memory), not more work.

### Ways not taken

- The 1070 reads the cold experts too (zero copy on its own link): its
  round trip would outlast the 3060's layer in a step, and in a verify
  group the 1070 is already the slower card.
- The 1070 runs the shared experts: they would lengthen its round trip by
  more than they save the 3060, which now runs them in the wait.
- The 1070 takes dense work (half the rows of the Mamba projections): a
  round trip for each matrix, 80 a step.
- The cold experts split across both links (each card half the rows of
  each): at most half the 3060's zero copy, about 1.6 ms of a step.

## 8. Bugs found on the way

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

## 9. Other measurements of this work

The 1070 with 4 GB of ballast (a 4 GB card), Gemma 26B in Maxwell mode
(`RECLANK_GPU_CC=52`): 16K prompt 265 tok/s, decode 23.4; 32K 213 and 21.2
(regular mode at 32K: 233 and 22.3).

## 10. Open

- Gemma and Qwen: the 1070 could take dense work (layers), not only experts.
- The prompts do not use the 1070 (Gemma stages its cold experts; Nemotron
  and Qwen read them by zero copy or the CPU).
- Nemotron's verify group: the 1070's IQ2_XXS group kernels (`k_nm_hg_*`,
  compute bound on sm_61) are its limit; the 3060 waits 7.9 ms of 62.
- Nemotron's refresh only between prompts: one after the first 64 steps of
  a long answer would halve the rest's zero copy (offline: 10.8 -> 5.1%).
- nsys with two cards (a switch to start the device on the main thread).
- Qwen's profile from more text, the prompts too.
