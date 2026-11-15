# Program memory on the GPU

Notes for keeping the device memory of the program generators small (a `compile_*` with a GPU emitter:
Nemotron's `NmEmit`, and the same shape in Gemma 4, Qwen and DeepSeek). They come from Nemotron 3.5 Super
on the RTX 3060 (12 GB) beside the GTX 1070, where every GB that the programs do not hold goes to the hot
experts (decode) or the stage of a prompt's experts (prefill).

## 1. Measure it first

A program's memory is the device copies of the host arrays its records name (the `Mirror`'s, added at
its first load) and its own named buffers (`GPUProgram.named`). Diff `set(g.mirror.arrays)` around the
making of one program (`NemotronGPU._prog(t)`), map the arrays to names by their addresses through
`prog.names` and the emitter's scratch (`NmEmit._scr`), and add the program's `named` buffers. Two traps:

- The estimate a generator sizes things by (`NemotronGPU.group_bytes`) was within 5% of the arrays, but
  the first run adds more: the cache and the Mamba states (`cache_dev.attach`, 0.4 GB at 8K for Super),
  the step's programs, the graphs. Measure free memory (`gpumm.mem_info`) and the pool's size around a
  whole prompt and the `relax` after it, not around a program.
- Aggregate by name and by row: a prompt group's memory grows with its rows, so the useful number is
  bytes a row.

Super, a prompt group (about 600 KB a row: 0.66 GB at 1024 rows, 2.48 GB at 4096):

| | KB a row |
|---|---|
| the routed experts' pairs: the outputs of each pair before the sum (`de`, 90), the act in two int8 planes (`qa`, `qal`, 118), their scales (7) | 215 |
| Mamba: in_proj's output (`zxd`, 74), the conv's (41), y (33) | 150 |
| rows of the hidden width: the residual (`H`), the norms (`hn`, `xn`), the attention's | ~120 |
| the shared expert, the latent and routing rows, the int8 x of the products | ~115 |

Top-k 22 makes the per-pair scratch the largest item; any MoE with a large k is the same.

## 2. What was done (Super, the 3060)

1. **Big programs from the pool, freed after the prompt** (`NemotronGPU.is_big`, `BIG_FROM`). The programs of
   2048 and 4096 rows already took their memory from the hot experts at a prompt and gave it back after
   (`relax`). The 1024 one does too now: decode keeps 0.66 GB more of hot experts (3.1 -> 3.7 GB). The
   programs of 64 and 256 rows stay: a chat's every turn runs them, and remaking them costs a turn's latency.
2. **Give the memory back with the right margin** (`relax`). It kept `MARGIN_GB` (0.9) free before refilling
   the pool, but the margin had been spent on what was made after the pool was sized (the step's programs,
   the cache at the first run), so the pool came back 0.5 GB short after every prompt. It keeps
   `BIG_MARGIN_GB` (0.4), the reserve, and the room of the small programs not made yet: 3.53 of 3.73 GB.
   A program that finds no room still makes the pool give some back (`_load`).
3. **Count what a prompt holds at once, and hold no more.** A prompt in parts used groups of 2048 and 4096
   rows, both programs kept until the end, beside the stage: out of memory on the server's 5K prompts. One
   big group's program at a time now (`_free_big` before the next is made), and one of the MTP block's big
   prompt parts at a time. A rule that freed the others only when the pool could not give the next program
   kept two, and left the MTP block's part no room: the room a generator plans (`_fit_big`) must be what it
   then allows. The MTP block's parts of more than 256 rows are big programs too: `_fit_big` takes the
   largest part that leaves a big group room (`mtp_part`; `spec.NemotronMTPSpec.flush` keeps to it).
4. **Stage the experts in chunks** (`stage_c`, `NM_GROUP_MOE` operand 23). The staged prompt groups copy each
   layer's experts into two buffers of a layer (3.2 GB) while the layers before run; that did not fit, so
   the 3060 read them across PCIe in the tiles (`k_nm_gtile_nv4`: 4.2 GB/s, the SMs waiting on each load).
   Two buffers of a chunk of the experts do fit: a 4K prompt 200 -> 363 tok/s, 8K 369, 1K 133 -> 166.
5. **A window of the pairs' scratch** (`win_cap`, `WIN`; nm.cu `nm_window`, `k_nm_gsumw`). With the stage in
   chunks, a chunk's pairs are contiguous in the sorted order: their act and outputs go in the scratch by
   their place in a window of the chunk's pairs (twice a chunk's share), and each window's outputs are added
   to the tokens' rows after it; a chunk of more pairs runs in more windows (empty ones exit). The sort puts
   each expert's pairs in the order of their index (`k_nm_gsort`, with windows), so the windows, and the
   order of the sums, are the same from run to run. The 4096's program 2.5 -> 2.1 GB at the same speed;
   the logits KL 4e-7 (the order of the k slots' sum changes).
6. **Sub-batches of rows** (`sub_rows`, `SUB`; `kq._compile_nemotron_sub`). Only the residual stream, the
   rows after norm_f and the routed experts' rows (lat, idx, val, their output) are kept for every row of
   a big group; everything else is the size of a sub-batch (1024 rows) and reused by each. The Mamba
   states carry on through the cache's slots, the attention's keys and values go in at each sub-batch's
   position (`pos.s`, `nreal.s`, bound by `_prepare`), and the routed experts still run once for the whole
   group (its experts read once). A block's outputs go into the residual at once (the order of the adds
   changes). The 4096's program 2.1 -> 1.1 GB; the room let the stage take 2 chunks in place of 4: a 4K
   prompt 365 -> 395 tok/s.

After all of it, the 4096's program (from the arrays' sizes): every row's residual, rows after norm_f and routed
experts' rows 177 MB; a sub-batch's rows 314 MB (the hidden width 84, in_proj's output 76, the products' int8 x
52, the conv's 42, y 34, the shared expert's 22); and the window of the pairs, the largest part left: 884 MB in 2
chunks (twice a chunk's share is every pair), 442 in 4, 221 in 8. `_fit_big` takes the fewest chunks that stage
every size (fewer copies), so with room to spare the window is big; that costs nothing during a staged prompt (the
pool's hot experts wait for the decode, and relax gives the memory back), and more chunks or a smaller
`WIN_SLACK` are the levers when it must fit beside something else. The server on both cards then: prompts of 5.2K
and 15.7K tokens at 321 and 389 tok/s plain (236 and 329 before the work of this note), 235 and 312 with MTP
(197 and 223).

## 2b. Nemotron's prompt programs packed (Super on the GTX 1070, Bolt; 8 GB beside 7 GB of dense rows)

Super on the GTX 1070 had 0.18 GB free after its load: its prompt groups stopped at 256 rows (each pass of a group
reads every expert across the link, so a 1K prompt ran 4 of them: 56 tok/s). Measured as section 1 says, the
64-row program was 75 MB of which 27 MB its arrays, 8 MB its graphs (955 records in ~6 segments): the graphs and
the records were never the problem, the buffers were. In order:

1. **No scratch of the tensor cores' on a GPU without them.** rc_gpu_load made the bfloat16 rows of the BF12 and
   MXFP8 matrices (up to 32 MB) and the two bfloat16 planes of x (2 t cols bytes) for every program; only
   bf16_tc_on's kernels read them (0 without the tensor cores). api.cu skips both there; the same bits.
2. **A window of the pairs on the mapped path** (MAP_WIN_C chunks of the experts, k_nm_gtile_nv4p's operand 23 as
   k_nm_gtile_nv4t's): each chunk's pairs in a window of twice its share (passes enough for the worst chunk; the
   empty ones exit at once), k_nm_gsumw adding each window's outputs. Each expert still crosses PCIe once (the
   chunks split the experts). On the 3060 (k_nm_gtile_nv4p forced, the same chunks and windows as the stage) the
   prompt's logits are bit for bit those of the staged path.
3. **The sub-batch that fits** (_fit_big: for each count of chunks, each big group's sub-batch the largest of
   its rows whose program fits the room with that count's window; SUB still fixes it).
4. **Packing** (ir/alloc.pack, NemotronGPU._pack; WRITES has the Nemotron records): each layer's temporaries
   its own (kq.compile_nemotron_step's rows, NmEmit's scratch and int8 x, P.lazy_zeros: no memory until
   written), so those of layers that do not live at once share the arena's bytes; and the programs share one
   arena (the largest so far; the programs on a smaller one made again on it when used). A big group's program
   takes one of its own, freed after the prompt, unless the pool holds less than SHARE_BIG_GB (the 1070).
   Super's arenas: 15.8 MB at 64 rows, 45.0 at 256 (its buffers had been 171 MB), 1024 rows 180 MB whole or 98
   MB in sub-batches of 512 (0.62 GB before); Bolt's 12.9 and 184.6 MB at 64 and 1024. NemotronGPU.arena_bytes
   estimates them (each layer kind's temporaries live at once: within 1% for Super, 20% over for Bolt), and the
   margin of a big group is 0.12 GB where it shares the arena. The logits bit for bit those unpacked (Super and
   Bolt, both GPUs; RECLANK_PACK_POISON=1, the arena NaN at first, the same bits: no record reads a byte it did
   not write).

The 1070 then takes Super's prompts in groups of 1024 (sub-batches of 512) with the graphs: 1K and 4K 56 -> 72
tok/s, decode the same. Bolt's prompt programs 0.85 -> 0.40 GB of the estimate: its hot experts 32 -> 35% on the
3060 (decode 30.1 -> 30.4, the drafter 34.8 -> 36.5, 4K 538 -> 547), 8 -> 11% on the 1070; the server on both
cards 29 -> 32% hot, chats 37.9 -> 38.6, prompts of 5.2K / 15.7K 457 / 499 -> 462 / 504. Super's server on both
cards (the 3060's pool 0.1 GB larger): MTP with 3 drafts, chats 15.5 (15.5-16.0 before), prompts 236 / 311 -> 262 /
335; without MTP chats 14.1, prompts 321 / 390 -> 327 / 395.

## 2c. One set of graphs for the layers of a kind, chained by tail launch

What was left of a packed program was mostly its graphs, one exec a segment. Super on the GTX 1070, packed, the
device memory of each program's graphs (cudaMemGetInfo around prepare, with the programs made largest first, so
that no program is closed and no graph's freed memory is used again):

| Rows | Records | Segments | Graphs | Arena |
|---|---|---|---|---|
| 1 | 994 | ~7 | 6 MB | (not packed) |
| 64 | 955 | ~6 | 6-13 MB | 15.8 MB |
| 384 | 2195 | ~14 | 57 MB | ~67 MB (est.) |

The 256- and 1024-row programs have the same 2195 records: the windows' records, 8 chunks of the experts, each a
few passes, each several kernels. A toy (one process a measure) gives ~7.5 KB of device memory a kernel node, a
graph's memory in 2 MB chunks (14 graphs of 160 nodes: 27 MB; one of 2240: 17 MB), and the same for graphs made
for device launch. The records themselves are 224 bytes each (0.5 MB). (A first measure of the 256 rows'
graphs, 46 MB, was low: the bigger arena had closed the 64-row program, and its graphs' freed memory was used
again.) Super's 88 layers are three kinds (Mamba, attention, MoE), and no two layers of a kind come one after
the other (in Bolt either).

**What was built.**

1. **The layers marked.** A record's flags (ir/opcodes.py: F_LAYER with the layer's index + 1 from
   F_LAYER_SHIFT, F_LAYER_END on the first record after the last layer; Program.mark gives them to the next
   record emitted). kq.compile_nemotron_step and _compile_nemotron_sub mark the layers of programs of at least
   TAIL_ROWS (RECLANK_NM_TAIL_ROWS, 17) rows. The records' order and count do not change, so the fusions across
   records (gg_skip, gg_qx_src) see what they saw.
2. **The chains** (runtime/graphs.cu). rc_gpu_load's gg_find_links: each run of marked layers with no boundary
   record inside is one segment, a chain. At its capture (prepare or the first run) each of its layers' records
   is captured with its kernels reading their records in one area, dbody, not at the layer's place in dcode,
   and k_tail_next goes last. gg_graph_cmp compares it with the program's bodies so far: the same nodes
   (kernels, grids, blocks, shared memory, every argument's bytes by cudaFuncGetParamInfo, memsets, memcpys,
   edges). The same is that body; else a new body. The new bodies are instantiated for device launch and
   uploaded.

   The run: the host launches k_tail_start (the first layer's records into dbody) and its body. Each body's
   k_tail_next copies the next layer's records into dbody and tail-launches that layer's body; the stream counts
   the chain as the one launch, so what follows is queued as before. The host's records stay whole, so prepare,
   the CPU's tasks and the mirror walk them as before.

   Where the chain cannot run, its records are one ordinary graph instead:
   - a layer whose record fuses one past the layer's end (gg_skip still set at its end);
   - a body that cannot be made for device launch.

   A failed tail launch writes its link and error into mapped memory; the next run reports it.
3. **Kernels that took a layer's pointers as arguments** read them in the record when the launch gives it
   (GG_RP_KQ, GG_RP_SC in kernels/operands.cu; the pointer arguments then null):
   - the MXFP8 products k_mxf8_gemv(_s), k_mxf8_tc and k_mxf8_gemm_f, and their FP8 tensor scale (Super's
     Mamba in/out, the head);
   - the NVFP4 products k_nv4_gemv and k_nv4_gemv2 (Super's shared experts in a step);
   - GP_KQ_LINEAR's x to int8 (k_kq_quant_x_rec, k_quant_x2_rec).

   RECLANK_GPU_TAIL_LOG=1 prints, for each new body, what made it differ from the body most like it: that is how
   these were found (Super's chain of 88 layers: 49 bodies, then 4).

**What it gave.** Super on the GTX 1070, four bodies a program (layer 0, whose first record is a plain norm;
Mamba; MoE; attention). Its graphs, made largest first:

| Rows | Graphs, chains off | Graphs, chains on |
|---|---|---|
| 384 | 57 MB | 0 (under 2) |
| 320 | 59 MB | 2 MB |
| 256 | 59 MB | 2 MB |
| 64 | 13 MB | 0 |

Five programs made: 0.538 -> 0.727 GB free (189 MB). Bolt on the 1070, its prompt programs of 512, 256 and 64
rows (not staged): 13 / 10 / 10 MB -> 2 / 0 / 4, 27 MB in all; its MoE layers vary (E 240-496, inner
1536-2560), so the 64-row chain has 15 bodies.

The logits are bit for bit those of the ordinary graphs (the prompt's last row and 8 steps):
- Super on the 1070 (1K, and 4K with RECLANK_PACK_POISON=1), on the 3060 (4K), on both cards (1K);
- Bolt on the 1070 and the 3060 (4K; the CPU's share off).

check_nemotron_super_spec PASS on the 3060 and on both cards. The speed:

| | Chains off | Chains on |
|---|---|---|
| Super, 1070, 1K / 4K prompt | 74 / 72 | 74 / 73 |
| Super, 3060, 1K / 4K prompt | 172 / 402 | 172 / 403 |
| Super, both cards, 1K prompt | 172 | 172 |
| Bolt, 1070 / 3060, 4K prompt | 189 / 549 | 187 / 547 |

**What is not chained.**
- **The steps** (fewer than TAIL_ROWS rows). A chain costs a few us a layer: Super's step on the RTX 3060
  13.01 -> 12.89 tok/s (192 steps, twice each, alternated), the 1070's 8.6-8.7 -> 8.5, for ~6-10 MB a
  program. RECLANK_NM_TAIL_ROWS=1 chains them too.
- **A program that fetches weights into a stage** (GP_FETCH: the 3060's staged groups, Bolt's on both GPUs).
  Their MoE layers fetch inside, so only the Mamba and attention layers could be chains, of one or two layers.
  - Super on the 3060, 1K prompt: 172 -> 164 tok/s with them chained. Its pull kernels (k_nm_tile_pull, on the
    copies' stream beside the layers) took 5.42 -> 5.64 s; no toy's launches, events, priorities or overlaps
    differed.
  - Bolt on the 3060: 549 -> 537 tok/s.
  - Those layers get little to share: on the 3060, none (below); on the 1070 Bolt's Mamba layers share 3
    bodies, ~2-3 MB.

  RECLANK_GPU_TAIL_FETCH=1 chains them anyway.
- **The 3060's tensor-core products** (gg_gemm_bf16 and gg_gemm_bf12: k_to_bf16, k_gemm_bf16_tc,
  k_mxf8_to_bf16 on chunks of rows) still take their pointers. Its 64-row chain has 88 bodies (no gain), and its
  steps' chain 4. These would need the record form too, with the chunks' offsets.
- **Programmatic edges** (PDL, sm_90 and later): none in a body. k_tail_next must wait for the whole layer.

Switches: RECLANK_GPU_TAIL=0 (the ordinary graphs), RECLANK_GPU_TAIL_MIN (the fewest layers of a chain),
RECLANK_GPU_TAIL_PLAIN=1 (each chain an ordinary graph, the cuts at the layers kept), RECLANK_GPU_TAIL_LOG=1.

**The cost of the other ways**, a toy (2220 kernels of one block, 88 layers of 3 kinds, best of 8,
interleaved) against 14 graphs:

| | Tail launch | Host launch a layer + copy | WHILE + SWITCH |
|---|---|---|---|
| GTX 1070 | -0.04 to +0.33 ms | 0.27-0.63 ms less | +0.29-0.64 ms |
| RTX 3060 | +0.46 ms | +0.23 ms | +0.79 ms |

Two other ways were not chosen. Env slots per layer would need the compiler to emit them and the host's checks
to read per-layer tables. cudaGraphExecKernelNodeSetParams costs ~1 us of host time a node.

## 3. Pitfalls met on the way

- **numpy's huge pages against pinned memory.** numpy marks large arrays `MADV_HUGEPAGE`; their faults
  compacted memory directly, and the compaction tried to move the host pages the GPU had pinned (a
  weightkeeper of 4 KB pages, registered for zero copy). Each try failed and flushed the TLBs of every CPU:
  ~1M shootdowns a second, the OpenMP team waiting at its barriers (a 1K prompt 121 vs 172 tok/s). The
  runtime turns them off when it loads the CUDA library (`runtime._no_numpy_hugepages`). The same storm
  from proactive compaction: `vm.compaction_proactiveness=0`, or a keeper under `sudo --lock` (1 GB pages,
  which compaction never moves).
- **Mapped reads want few loads in flight.** Reading host memory from a kernel ran ~12 GB/s with one block
  an SM and one 16-byte load a thread, 6-7 GB/s with more in flight; a kernel made for the GPU's own memory
  (many blocks, every load first) read 5 GB/s. Copy the host's rows into a stage with a small pull kernel
  (`k_nm_pull`, `k_nm_tile_pull`) and compute from there.
- **Estimates of a size you then change.** A program's size depends on the chunk count (the windows) and the
  stage's on the same count: `_fit_big` tries the counts and keeps the fewest that stage the most sizes.

## 4. For the other generators

- Split a program's per-row memory into what lives across layers (the residual, what the head and a drafter
  read, the inputs and outputs of a pass that must see every row at once) and what one layer uses and drops.
  The second kind can be sub-batched in any family whose layers run in order; recurrent layers carry their
  state in the cache's slots already (Mamba here; Qwen3.6's DeltaNet layers the same way), attention through
  the cache's positions.
- An MoE's per-pair scratch (k of them a row) is the first thing to bound: by a window of the sorted pairs,
  summed into the outputs per window. Qwen's and Gemma's group MoE records have the same form (pairs by
  expert, a sum after).
- Make the large prompt programs temporary and sized from the cache or pool they borrow from, and check the
  margin you restore after them against what was allocated since the sizing.
- Count every program that can be alive in one prompt (the group sizes of its parts, a drafter's passes),
  and enforce the count you planned: free one before making the next.
- Keep the small sizes that every request runs.

Not done yet, and worth it: bfloat16 storage of the large intermediates (in_proj's output, the conv's, the
pairs' outputs: ~90 KB a row); the MTP block's parts packed too (Nemotron's main programs are: section 2b).

## 5. Gemma 4 (the 26B and the 12B on the RTX 3090)

Gemma's prompt groups were small already: ir/alloc.pack puts every temporary of a layer in one arena by
its liveness, and WarmPool frees the large groups' programs at the first decode step after a prompt when
the hot experts are in the pool (a card of 12 GB). What was left (KB a row, the peak of the arena):

| | before | after |
|---|---|---|
| the 26B: GP_MOE_GPU's pairs (de 90, act 45, act2 22.5), the experts' input and output (22.6) | 180 | 103 (1024 rows), 77 (2048: the global attention's q and output, the peak now) |
| the 12B: the gate, the up and the GELU rows of the dense part (61 each), the norm (15) | 200 | 154 |

1. **A window of the pairs** (GP_MOE_GPU operand 23, `gpu.MOE_WIN` 4096; kernels/moe/group.cu). The
   products of a window of 4096 of the sorted pairs write act, act2 and de at the pair's place less the
   window's first (the scratch holds a window and a tile of slack: an expert's last tile may run past the
   window's end, and the next window computes those pairs again), and `k_moe_sum_w` adds each window's
   outputs to the tokens' rows. h is quantized once: a window's act2 goes after it in the int8 scratch.
   `k_moe_order` (a block for each expert, after the atomic sort) puts each expert's pairs in the order of
   their tokens, so the windows are the same from run to run (the prompt programs' logits bit-identical
   between runs). The first form of that order, one thread scanning the pairs for each expert, cost 77 ms
   a prompt of 4K tokens (nsys: 1.3 ms a layer); prompts were 3.5% slower. The 26B's group of 2048 rows
   369 -> 157 MB, of 1024 185 -> 105.
2. **The GELU in place** (`GELU_IN_PLACE`): gelu(g) * u of a large group into g. The same bits. The 12B's
   group of 2048 rows 409 -> 315 MB.
3. **One arena of the large groups** (`ONE_ARENA`): ir/alloc.Arenas gives a new program the arena of the
   others when it is large enough, but a group made after smaller ones takes a larger one, and the smaller
   groups kept theirs (the 26B after prompts of 1024 then 4096 tokens: 0.64 GB of programs). They go when a
   larger arena comes, and a later use makes them again on it (0.2-0.3 s): 0.64 -> 0.22 GB (the 12B 0.73
   -> 0.39).

The logits of 195 decode rows of three real prompts (2.8K-5.1K tokens, a reply of 64 each) against
transformers in float32 on the GGUF's weights: KL 1.23e-4 before, 6.9e-5 after (the same rows through the
prompt programs 1.06e-4, 6.8e-5), the top token the same in every row; after against before 8.3e-5, the
order of the sums of a token's slots amplified by the routing of the layers after (Nemotron's window gave
4e-7). The speed, A and B in turn in one process: the prompt within 1% (1024 tokens 1392 -> 1392 tok/s,
4096 1168 -> 1170, 8192 941 -> 950), the decode the same; with 12 GB (the hot experts in the pool, which
keeps 0.2 GB more during a prompt) 1019 -> 1046, 1059 -> 1065, 907 -> 906, the decode 90 -> 96 at 1024. The
12B 2291 -> 2286 and 2234 -> 2256. bfloat16 storage was not needed: after the window the largest items are a few rows of the
hidden width and the attention's.

## 6. Qwen3.8-Flash-Next (EXL3 on the RTX 3090)

QWEN38_PLAN.md, "The memory of the programs", has the detail. The mixed group of 4096 rows took 3.47 GB
(1.81 now): the arena 406 KB a row (236), the C side's scratches 1.39 GB (0.59).

1. **The arena's peak was a dense layer**, not the MoE: compile_qwen4_step gave every product's output
   the widest row of the layer, and a DeltaNet layer's beta and alpha (48 columns) held 48 KB a row each.
   Each output is now a temporary of its own columns; the bits the same.
2. **A window of the sorted pairs** (`RECLANK_GPU_E3_WIN`, 12288 rows) in the EXL3 GP_KQ_GROUP_MOE: its
   workspace (the int16 x and the parts of every pair, 1.19 GB at 4096 rows) and act, act2, de of a
   window, `k_qmoe_place` putting each expert's pairs in the order of their index. The KL of decode rows
   to float32 the same (3.52e-3 before, 3.27e-3 with the window, 249 rows).
3. **What a prompt's programs hold after it**: a program loaded after a large one takes the newest C
   scratch of a kind, so the MTP layer's group of a prompt's rows kept the mixed group's scratches and the
   large arena (2.9 GB) through the decode. The programs of more than 16 rows load with scratches of a
   class of their own (`rc_gpu_scratch_own(2)`), and all of them, the arena too, go at the first decode
   step; until then they stay (they had been freed and made again between the parts of a prompt).

In Qwen all a prompt takes comes back to the pool at its first decode step, so the sub-batches of the dense
part (236 -> about 100 KB a row) and bfloat16 intermediates would help only while a prompt runs: not done.

## 7. DeepSeek V4.1 Flash (the 3090: the server with images, two lanes, DSpark)

Measured as section 1 says (the Mirror's arrays by the line that made them, free memory around each program, each
prompt and each relax). After the load: 0.44 GB free; the hot experts 7.5 GB (10 slots a layer), the dense weights
8.6, both lanes' caches 1.9, the prompt groups' scratch 2.5 (made for 2048 rows by the first group of any size, and
kept), their stage 0.6, the C side (outside GpuMem) 3.1. A group's scratch ~1.2 MB a row: o, oa, y 185 KB; x, xa, qa,
qr, q 182; the hot experts' act, act2 and int8 planes 180, their de 123; engram's kv 100; the index scores 98; the
streams 80. Every row's bits stay those of before (the logits of a 5018-token prompt and 32 steps: the same md5;
both lanes against each alone, check_dsv41_lanes):

1. **The hot experts' slots by a table** (`DSV41GPU.stores`: the address of each slot, kernels/dsv41/common.cuh
   `ds_slot_w`), in segments of one slot of every layer (`segs`). shrink_hot moves the best experts of the slots it
   frees into the slots that stay (on the GPU) and frees segments, grow_hot adds segments and fills them at once
   (`DSHot.plan(most=...)`); no program is made again (both freed every program before). A step 36.5 -> 36.7 ms.
2. **Big groups from the hot experts** (`is_big`, `BIG_FROM` 512, `big_room`, `relax_big`): their programs and
   their own scratch (`big_scratch`, its stage: `STAGE_FROM`) made at a prompt, one at a time of either lane, the
   slots they lack taken from the hot experts, all freed after the prompt's last group (the engine, before the
   answer) and the slots back. The small groups' scratch is made for the largest small group (0.32 GB).
3. **The margin** (`room_margin`): the image encoder's room (`reserve_gb`, 1.8 GB with images) and
   `GROW_MARGIN_GB` 0.5 (it was 4 for all of it); the engine grows the hot experts into the rest after the warm.
4. **Sub-batches** (`SUB` 512, `sub_ranges`, `loc_rows`): each half's attention, engram's rows, the hot and the
   shared experts one sub-batch after the other (the window through the ring, as half B after A), their arrays a
   sub-batch's and shared by the halves (`zl`, the scratch key "sub"); the CPU's task and the stage of the half.
   The big group's scratch 2.5 -> 1.3 GB.
5. **The attention parts of attn_launch only where they are used** (runtime/api.cu): every program made 12.6 MB of
   them, and DeepSeek's records never use them (GP_ATTN_F32, F32H, QC, Q8, V8, TQ and QSA do). A small program's
   own memory 45-53 -> 12-20 MB; the C side after the load 3.0 -> 1.95 GB.

After it: the hot experts 7.5 -> 9.0 GB at the load (10 -> 12 slots; plain decode 9.8 -> 10.5 before item 5) and
9.8 GB (13) after a long prompt, 2.8 GB free (the margin); a 4K prompt of lane B takes one slot back for its group
and gives it back. The KL to the float32 reference the same (the bits the same); the server's prompts and decode the
same within noise (its prompts are bound by the reads of engram's rows from the NVMe, one page fault at a time:
cold 70 tok/s against 560 warm on the same prompt; not of this work).
