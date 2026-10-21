# The GPU and the CPU at the same time

The question: how can the GPU and the CPU both be busy, not take turns?
This plan names where they wait for each other today, what already overlaps,
and the ways to more overlap, best first. DeepSeek V4.1 Flash is the example
(the numbers of DEEPSEEK_V41_PLAN.md); the same holds for the Qwen3.8 MoE models.

Status: a plan. A read of the code (below) leaves one new lever for overlap: lanes (2.1). The
halves of a verify group (2.2) are already merged where it pays, and the next draft cannot start
during a verify (2.3).

## 1. Why they take turns

A token goes through the layers in order. In each layer:

```
GPU: attention, dense, router ─┐                  ┌─ combine -> the next layer's attention
                               ├─ experts ────────┤
CPU:              (waits)      └─ cold experts ───┘     (waits)
```

The GPU's hot experts and the CPU's cold experts run beside each other: the
only overlap of one token. The next layer's attention needs the combined output,
so the CPU waits on the GPU, then the GPU on the CPU. DeepSeek, a step of about
37 ms: the CPU's tasks about 20 ms (4.7 cold experts a layer, 18.7 MB each, at
about 195 GB/s, the DRAM's rate), the 3090 about 17-20 ms, mostly one after the
other. One stream cannot change this: token t + 1 needs token t, and layer j + 1
needs layer j.

## 2. What overlaps: a second stream of work

Two pieces of work that do not depend on each other, half a layer apart, keep
both busy:

```
GPU: A attention | B attention | A attention | B attention ...
CPU:             | A experts   | B experts   | A experts   ...
```

The ways to a second stream, best first.

### 2.1 Two conversations (lanes)

Done for Qwen3.8 EXL3 (LANES_PLAN.md, commit cac9544): two plain decodes 1.37
times the tokens of one (62.6 against 45.8 tok/s); one stream alone about 6%
slower (fewer hot experts a layer). Not done for DeepSeek.

- Work: DeepSeek lanes as LANES_PLAN.md: a program of each lane, the mailbox
  and the lane thread of rc_gpu_lane_serve, the cache and the state of each
  lane, the server's routing of two requests.
- Gain: about 1.3-1.4 times the total, only while two requests run (dsh
  subagents in parallel, two users). One conversation gains nothing.
- Written, not yet run on the GPU (reclank/models/deepseek41/lanes.py,
  scripts/check_dsv41_lanes.py; `--lanes 2` of serve_dsv41.py). DeepSeek's
  programs hold the addresses of their state from the time they are made, so
  a lane cannot bind a cache as Qwen's do: lane B has a sequence's state of its
  own (DSV41GPU.lane_state: the caches, the compressor states, the attention
  keys of the steps, the CPU tasks' scratch; its engram hashes; DSpark's rings)
  and its own programs, made under DSLane.view and keyed by the lane (freeing
  the programs, for room or a new cap or the hot cache's size, frees both
  lanes'). Its steps and verify groups get C scratches of their own; its prompt
  runs alone. Any two runs of up to VERIFY_MAX rows pair (run_both: a step
  beside a verify group too). The server: the engine hands its runs to
  LaneScheduler (DeviceOp, DSpark.stream with coop), a prompt in parts of
  RECLANK_LANE_PART with pauses; lane B's caches as long as the model's (the
  lanes share the contexts in host memory); relax keeps the cap of the other
  lane. Checked without the GPU: the view's swap of the state, and the coop
  loop of DSpark against the direct one (the same tokens and restores).

### 2.2 The halves of a verify group (DSpark)

DeepSeek splits a verify group into halves A and B that overlap: A's CPU task
runs while the GPU runs B (gpu.py, "A's task runs on the CPU while the GPU runs
B"). The halves gain less than they could: each reads all the FP8 dense weights
again, so the GPU bounds the group (41 ms of kernels for 2 rows; a verify group
of 2 58 ms, 3 70, 4 89).

- The dense products of both halves in one pass of the weights would undo the
  overlap: half B's attention of layer i needs B's FFN of layer i - 1, which is
  on the CPU while A's attention runs. Only the experts can go together: the
  hot and shared experts of A and B after B's signal (gpu.py, merge, groups of
  2: 51.0 -> 49.4 ms). For more rows it was slower (6 rows 103 -> 106 ms: the
  CPU bounds them, and A's experts filled the GPU's wait for B's task). So
  done where it pays; nothing more here.

### 2.3 The next draft during this verify

A DSpark round: the draft (12.6 ms on the GPU), then the verify (GPU and CPU
in turn). The draft of round n + 1 needs the verify of round n. But while the
verify's CPU tasks run, the GPU can draft round n + 1 on the guess that every
draft of round n is taken; when one is refused, that draft goes.

- Not possible: the draft reads the main rows of DSpark's target layers, 37, 38 and 39 of 40,
  and its first token is the one sampled from the verify's last logits. Both
  exist only when the verify ends, so a draft cannot start while the verify's
  CPU tasks run. (The drafter's own guess of that token would still need
  layers 37-39.) The chain draft -> verify -> draft has no gap to fill.

### 2.4 A shorter CPU half

Less to overlap when the CPU's part is smaller:

- A second GPU with a share of the cold experts (TWO_GPU.md's expert device,
  not done for DeepSeek): an RTX A2000 12 GB holds about 590 experts more,
  about a quarter of today's cold pairs; the CPU about 20 -> 15 ms a step, about
  10-15% decode.
- The hit rate of the hot experts: the best possible 0.63 against DSHot's 0.43
  on recorded routes, but a copy over this PCIe link (8 GB/s, 2.3 ms an expert)
  costs more than a layer: no lever on this machine.
- Fewer bytes a cold expert: a change of the weights; not taken.

## 3. What does not work

- Splitting one token's layers between the GPU and the CPU: the chain above.
- The GPU computing cold experts over PCIe while it waits: one expert by DMA
  (18.7 MB at 8 GB/s, 2.3 ms) is longer than a whole layer.
- Prefetching the next layer's cold experts into the caches: 4.7 x 18.7 MB a
  layer against 35.75 MB of L3 a socket. (For Qwen3.8 EXL3 the predicted
  experts could be pre-decoded: qfe's notes, about 7-9% decode, not built.)

## 4. Phases

| phase | work | check |
|---|---|---|
| O1 | 2.3: not possible (the draft needs the verify's last layers and its sampled token) | |
| O2 | 2.2: done where it pays (the experts of the halves of 2-row groups) | |
| O3 | 2.1: DeepSeek lanes (written; the GPU checks to run) | scripts/check_dsv41_lanes.py: each lane's tokens those of its prompt alone (--fixed-hot: bit for bit), the pair's tokens a second against one; the server's warm pair at the load (--lanes 2) |
| O4 | 2.4: the expert device for DeepSeek's KQ_MXX experts (with a second GPU) | the device on and off give the same tokens; a step's time |

For one dsh conversation there is no lever of overlap left (a shorter CPU half, 2.4, is the one);
for the total of several requests O3 (about 1.3-1.4 times).
