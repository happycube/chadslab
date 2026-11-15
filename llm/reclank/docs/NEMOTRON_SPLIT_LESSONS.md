# Lessons from splitting Super's prompts across two GPUs

These notes come from running Nemotron 3.5 Super (NVFP4 experts) on an RTX 3060 (12 GB, sm_86) with a GTX 1070
(8 GB, sm_61, dp4a only) beside it, each on its own PCIe link, on a W-2133 (11 threads of AVX-512, ~53 GB/s of
DRAM). The 3060 runs the model; the 1070 is an expert device. In a prompt group the 1070 takes a share of each MoE
layer's experts (NEMOTRON_SUPER_PLAN.md has the data of each step). Most lessons apply to any work split between
two unequal GPUs that read the same host memory.

| A prompt on both cards | 1K tok/s | 2K tok/s |
|---|---|---|
| The 3060 alone (the 1070 holds decode experts only) | 172 | 334 |
| The split, the 1070 reading its cold share through the map | 209 | 317 |
| Its share copied by DMA into its memory | 254 | 352 |
| The next layer's first chunks copied ahead, the work reordered, the cost model re-tuned | 274 | 401 |

## 1. Find where the time goes, at the right grain

- **Totals point at the device; a timeline points at the cause.** The kernel totals (nsys, per device) said the
  3060 waited ~24 ms a layer for the 1070 and that the 1070's jobs took 55 ms. They did not say why the
  prefetch (2 of 5 chunks already in memory) shortened the job by only 2 ms. One layer's timeline did: the
  job's kernels with their start and end, the copy engine's busy intervals by stream, and the other GPU's wait.
  It showed the held experts' 20 ms of work running first while the copy engine sat idle.
- **Compare a change against its own baseline in the same session.** A run-to-run spread of 1-2 tok/s is normal
  here. A change that also changes something else (here: the order of the cold experts, and copies of experts
  with no pairs) can hide its gain. Two cases differing in one switch, run back to back, separate the effects.
- **Do not edit a CUDA source while a measurement starts.** The runtime rebuilds the library when the source's
  hash changes, and a run that starts after an edit spends minutes compiling.

## 2. Moving the bytes

- **Prefer DMA to mapped reads for bulk weights.** Beside the 3060's copies, the 1070's `cudaMemcpyAsync` from
  the registered host pages ran at ~7.8 GB/s. Its kernels' mapped reads ran at ~5-6.5 GB/s, and they also force a
  kernel shape that tolerates PCIe latency. Staging the share by DMA took the split from 209 to 254 tok/s.
- **Choose the kernel by where the data is.** The columns-outer tile kernel (k_nm_gtile_nv4pc) is 30% faster
  than the rows-in-shared-memory one on experts in VRAM, and ~8 times slower on mapped memory (its small reads
  per step cross the link). Run each kernel only on the experts it suits (a slot table says which are resident).
- **Copy only what is used.** A share chosen from the routing profile includes experts that a given prompt never
  routes to (8% of the 1070's share at 1K). Copying them anyway cost 1.2 GB and 3% at 1K. Copy those only if
  the copy is free (for example ahead of time, in a gap).
- **All copies share the DRAM.** The CPU's expert reads, the 3060's copies and the 1070's DMA all read the same
  ~53 GB/s. Adding a copy somewhere slows the others a little, so a split's gain is less than the sum of the
  links.

## 3. Keeping the copy engine busy

- **Copy into the gaps.** Between two MoE layers the 3060 runs a Mamba or attention layer (~30 ms at 1K rows)
  while the 1070 has nothing to do. Copying the next layer's first two chunks there removed part of each
  job's copies from the critical path.
- **Order the work so that the buffers free up early.** With two stage buffers, both full of prefetched chunks,
  the copy of the third chunk can start only when the first chunk's work ends. Running the long held-expert
  work first kept both buffers full and the copy engine idle for 20 ms a layer. Run the prefetched chunks first,
  then the work that needs no copies (the held experts) while the next chunks copy, then the rest. The same
  prefetch went from +4% to +16% by the order alone.
- **Validate prefetched data by identity.** The next job compares the expert ids at each prefetched slot with
  its own chunk layout and copies again any chunk that does not match. A changed share or a changed held set
  then costs a copy, never a wrong answer. A generation counter would miss the second case.
- **Size the chunks against the gap.** Stage buffers of 32, 48 and 64 experts gave 274, 274 and 242 tok/s at 1K,
  and 391, 403 and 402 at 2K. Larger chunks hide more per copy, but at 1K two chunks of 64 take longer to copy
  than the gap between layers lasts.

## 4. Balancing the two GPUs

- **Re-tune the split after speeding up one side.** The cost model gives the 1070 experts while its estimated
  time (the larger of its link's and its pairs') stays under the 3060's. Before the reorder, every setting that
  gave the 1070 more experts was slower. After it, the best rates moved (the 1070's link 5 -> 6 GB/s, its time a
  pair 10 -> 8.5 us) and gave 261 -> 274 tok/s.
- **Know when a split stops paying.** A group of 4096 rows is compute-bound on the 3060, not link-bound, and the
  split gave 452 -> 458 tok/s. The split is on for groups of 1024-2048 rows only.

## 5. Correctness of a reordered job

- **Find the implicit contracts first.** The group record does its setup (x's int8 planes, the sort of the pairs
  by expert) on the record whose chunk starts at local expert 0, and the final sum on the one that ends at the
  last expert. Launching the chunks in another order broke both silently. The fix was to number the local ids in
  launch order.
- **Check against a reference with KL, not only argmax.** The broken order still gave argmax 9 of 9 at 1K; the KL
  against the unsplit run was 4.2e-2 instead of 8e-3.
- **Tell a bug from ordinary divergence.** Dumps of each layer's sums from two configurations differ anyway:
  bfloat16 rounding in the next layers turns a 1e-9 difference into 1e-3 within two layers. To test a job, run
  it twice in one process on the same inputs, once each way, and compare the outputs (here max |d| 2.4e-7 at
  |out| 1-3). Then look at the prompt's logits (row 0) apart from the decode steps after it.
- **Look at the top-2 margin before chasing a decode difference.** A "broken" 2K case (KL 6.9, argmax 3 of 9)
  was an exact tie in the reference: two tokens at the same logit (65.612, a bfloat16 step is 0.31) at the third
  step. Any change of summation order flips such a tie, and greedy decoding then follows a different text.

## 6. Applying them to decode

A decode step on both cards takes ~71 ms. In each MoE layer the 3060 runs its hot experts, the 1070 its held ones,
and the CPU the ~12 cold experts a token that neither holds (~1 ms a layer at ~37 GB/s), while the GPUs wait.
Between MoE layers the 3060 runs the Mamba and attention layers (~0.7 ms) and the CPU waits. A step's experts are
not known until each layer's router runs, so the copy-ahead lesson needs a prediction.

| Decode after a prompt of 1K, both cards | tok/s |
|---|---|
| Before (the 3060 pulls 2 cold experts a token across its link) | 14.2 |
| The 1070's experts refreshed from the prompt's routing | 14.1 |
| 4 predicted experts copied ahead, the job's uploads queued behind them | 11.5 |
| ... the job's inputs read by a kernel, late copies left to the CPU | 14.0 |
| ... 3 copied ahead, the 3060's pulls off | 14.6-14.7 |
| The server's plain chats (the same change) | 14.12 -> 14.47 |
| The server's chats with MTP 3 drafts | 15.97 -> 15.88 (verify groups gain nothing) |
| The 3060 copies 1 more predicted expert in the gap, issued after the CPU's task | 14.5 -> 14.8-15.0 |
| The server's plain chats (the 3060's copy too) | 14.43 -> 14.74 |

- **Measure whether the future is predictable before building for it.** On the NumPy reference (a real prompt,
  13 minutes on the CPU), the next MoE layer's router applied to the residual entering the MoE layer before picks
  92% of its top 4 cold experts right, 88% of 6, 84% even two MoE layers ahead. The previous token's experts at the
  same layer predict only 38%, and refreshing the 1070's held experts from the prompt's routing moved its share of
  the pairs by 2 points: the prompt's routing does not predict the decode's at this cache size.
- **One copy engine serves every stream.** The first version was slower (14.2 -> 11.5). The 1070's job uploads its
  record, pairs and x (~20 KB) by DMA, and those uploads queued behind the prefetch's 12 MB: their start after issue
  went from 28 us (p90) to ~2 ms. The job now reads its small inputs from pinned memory by a kernel (16-byte loads,
  ~10 us), and the copy engine carries only the prefetch.
- **Never wait for a prefetch.** Each copied expert records its own event. Before the layer's pairs are split, the
  CPU's task drops the experts whose copies have not ended, and runs their pairs itself. A late copy then costs
  bandwidth, never time on the critical path.
- **The host's DRAM is the shared resource.** The CPU's pairs read ~37 GB/s and the 3060's pulls ~12 GB/s at the same
  time, near the ~53 GB/s the DRAM gives. The 1070's copies only paid off with the 3060's pulls off (14.1 -> 14.7):
  the bytes come from DRAM either way, so moving work between devices pays only where the DRAM has room. Copying in
  the gap after the CPU's pairs (DRAM idle) did not work either: the gap (~0.7 ms) is too short for the copies
  (13.9 tok/s, a third of them late).
- **A change that pays for one shape may not for another.** Verify groups (2-4 rows of the drafter) have ~30 cold
  experts a layer, and there the 3060's own pulls of the most-routed ones beat the 1070's prefetch: with the pulls
  off the server's MTP chats fell 15.97 -> 14.82. The group prediction stays off by default.
- **Find the slack before filling it, and size the work to it.** The gap between MoE layers (~0.7 ms, the CPU
  idle and the DRAM free) fits one 3.1 MB copy across the 3060's link (~7.4 GB/s beside the 1070's), not two: one
  expert 94% used and 14.5 -> 14.8-15.0 tok/s, two 45% and no gain, three 13.6.
- **Keep the issue of a copy off the critical path too.** The 3060's copies were first issued at the end of the CPU's
  task, before its completion flag, so the GPU waited for the API calls (~30 us a layer). The runner now calls a
  hook after the flag (rc_gpu_set_cpu_after), and the copies leave from there.
- **Signal completion from the GPU, not a host thread.** A host function after each copy set its flag on the CUDA
  driver's thread, which shares the CPU with an 11-thread team spinning on 12 hardware threads: the flags came so
  late that 15 of 7566 copies were used. A one-thread kernel after each copy on the same stream writes the flag:
  46% at once (94% with one expert).
- **Read mapped memory once.** The kernel that takes the staged experts first had all 1024 threads read the stage's
  words across PCIe (65 us a layer). One read a word into shared memory brought it under 10 us.
- **Count the copies that were used.** The device counts the experts copied, those a job read, and those that were
  late. That ratio (79% read, 14% late at 3 experts; 32% read at 4) found the right count faster than the rate did.

The ceiling here is the DRAM and the gaps, not the prediction. A step's ~1.7 GB of cold experts at ~53 GB/s is ~32
ms, and the CPU's window (~40 ms) and the 3060's dense layers (~31 ms) still run one after the other. The gaps between
MoE layers are too short to move much: the 1070 copies its 3 experts during the CPU's window (in the DRAM's headroom
once the 3060's pulls are off) and the 3060 one in the gap: ~3.3 of a token's ~13.5 cold experts a layer used.
NEMOTRON_SUPER_PLAN.md has the data.
