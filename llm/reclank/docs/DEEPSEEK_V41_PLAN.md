# DeepSeek V4.1 Flash in reclank

The plan for DeepSeek-V4.1-Flash from its safetensors checkpoint (510 GB in
48 shards) with the reference code of DeepSeek (inference/model.py,
kernel.py, engram.py, vision.py, image_processor.py) and the chat encoding
(encoding/encoding.py). Every fact here comes from config.json, the headers
of the shards, or that code (model.py:N), unless it says it is a measure or
an estimate. DEEPSEEK_V4_PLAN.md (V4 Flash Vision) holds the parts that are
the same (the hyper-connections, the two weight types, DSpark, the ViT); this
file gives what V4.1 changes and how it fits this machine.

**No model crippling.** The weights stay in the form of the checkpoint:
the MXFP4 experts, the FP8 matrices, and the engram tables are read in place
from the shards (or copied byte for byte), never requantized to fewer bits
(EXL3, NVFP4 transcodes, int4 and the like are out). Two modes (section 3):

- **near (the default):** nearly the reference's values, for speed and
  memory: the bfloat16 matrices as BF12 (the bfloat16 values in 12.25 bits;
  on Qwen3.8 0.0055% of the values went to zero, the largest 1.65e-3 of the
  RMS of its matrix), and the activations of the expert products as int16
  in two int8 planes (the Qwen3.8 path: finer than the reference's FP8).
- **exact** (RECLANK_DSV41_EXACT=1, serve --dsv41-exact): the reference's
  compute: the bfloat16 matrices as bfloat16, the activations of every FP8
  and FP4 product rounded as the reference rounds them, float products.

The rounding that is part of the model in both modes, as the reference does
it: the window KV, the compressed KV, and the indexer's keys and queries.

The target: the 3090 (24 GB), the experts in the RAM of the two Xeon 8268
with **balanced memory channels** (the same channels and the same memory on
each node: the plan takes about 156 GB a node, 313 GB in all), the two
engram tables on the two Optane modules (one on each node), and the CPU
alone too.

## 1. The checkpoint

| part | tensors | form | GB |
|---|---|---|---|
| routed experts, 40 layers x 384 | w1, w3 (2304 x 5120), w2 (5120 x 2304) | MXFP4: E2M1 codes two a byte (stored as I8), an E8M0 scale for each 32 values of a row | 271.8 + 17.0 scales |
| routed experts of DSpark, 3 x 128 | the same | the same | 6.8 + 0.4 |
| engram tables, layers 1 and 14 | embed (384006168 and 384016682 rows of 256) | FP8 E4M3, an E8M0 scale for each 32 values of a row | 196.6 + 6.1 scales |
| attention | wq_a (1280 x 5120), wq_b (32768 x 1280), wkv (512 x 5120), wo_a (8192 x 4096), wo_b (5120 x 8192) | FP8 E4M3, an E8M0 scale for each 32 x 32 block | 5.15 |
| shared experts, engram wkv, indexer wq_b, DSpark main_proj | | FP8, 32 x 32 blocks | 1.9 |
| embed, head | 129280 x 5120 each (not tied) | bfloat16 | 1.32 each |
| router, compressors, indexer weights_proj and wk, vision, aligner | | bfloat16 | 1.4 |
| hyper-connections, sinks, biases, norms | hc_attn_fn, hc_ffn_fn (24 x 20480) | float32 / bfloat16 | 0.2 |

The shards hold the names of the reference already (layers.N.attn.*,
layers.N.ffn.experts.N.w1.weight, layers.N.engram.*, mtp.N.*). wo_a is FP8
here (the reference converts it to bfloat16 on load, model.py:784; this
runtime keeps it FP8, a grouped product).

## 2. What V4.1 changes from V4

### 2.1 The shape

- 40 layers (V4: 43), hidden 5120 (4096), q rank 1280, 64 heads of 512, one
  KV head, window 128, vocab 129280, context to 1M (YaRN factor 16 over
  65536).
- MoE: 384 routed experts (256), top 6, inner 2304 (2048), one shared
  expert; router sqrt(softplus), the bias for the choice only, weights
  normalized and times 1.5 (model.py:792-827); no hash-routed layers
  (V4's tid2eid is gone); the clamped SwiGLU at 10 (model.py:830-851).
- Hyper-connections: 4 streams, 24 mixes, Sinkhorn of 20 iterations
  (V4: 19); the mixes of a sublayer are used by the next one: the
  attention takes the pre mix of the FFN before it, the FFN that of the
  attention (model.py:968-994). The head takes the pre mix of the last FFN
  (no hc_head_fn).

### 2.2 The attention: shared KV and a two-level indexer (model.py:613-789)

compress_ratios (40 layers): 0 for layers 0, 1; 2 for 2-19; 1 for 20-39 (the
three DSpark blocks 0).

- **Shared compressed KV.** Only the kv_source_layers (2, 8, 14, 20) compress;
  the layers after a source, to the next one, read its cache: layers 2-7
  read layer 2's, 8-13 layer 8's, 14-19 layer 14's (ratio 2), 20-39 layer
  20's (ratio 1: no pooling, a projection of each token, model.py:461).
- **Ratio 2:** kv = wkv x, score = wgate x (float32 products of the bfloat16
  weights, model.py:446-485), an entry for each 2 tokens: the softmax over the
  2 rows of score, times kv, summed; then the norm. A decode step keeps the
  incomplete pair (kv_state, score_state). No overlap (V4's ratio 4 had one).
- **Indexers.** index_source_layer_ids 2, 8, 14, 20, 24, 28, 32, 36 run an
  indexer; the other layers reuse the top 512 of the source before them.
  An indexer: 32 heads of 128 (wq_b FP8 of qr), RoPE on the last 64, the FP4
  rounding (E2M1, an E8M0 scale for each 32); the keys (only at the KV
  sources: wk, bfloat16, of the latent before RoPE, k_norm, RoPE, the same
  FP4 rounding); score_t = sum_h relu(q_h . k_t) w_h, w = weights_proj(x)
  times 128^-0.5 32^-0.5; the top 512 (model.py:488-580). No Hadamard
  transform (V4 had one).
- **Candidates.** Layer 20 (candidate_source_layer) scores all the entries of
  the ratio-1 cache and keeps the best 2048 blocks of 8 entries (16384
  positions; the block of the newest entry always), model.py:583-610; the
  indexers of layers 24-36 score only inside those blocks.
- **The keys of a query:** the window (the last 128 rows of the layer, its
  own, FP8: E4M3 with an E8M0 scale for each 32 over the whole 512 values,
  RoPE part too, model.py:700-720), and the 512 selected entries of the
  shared cache (the layers with a ratio). The softmax with a sink for each
  head; the output loses its RoPE (the inverse rotation), then the grouped
  LoRA (wo_a FP8 block-diagonal over 8 groups of 8 heads, then wo_b).
- **The cache values:** a compressed entry is RoPE'd (the position of the
  first token of its group, compress_rope_theta 160000 with YaRN) and then
  rounded to FP4: E2M1 with an **E4M3 scale for each 16** (model.py:760), the
  form of NVFP4 without its global scale. The layers without a ratio use
  rope_theta 10000 and no YaRN.

### 2.3 Engram (model.py:296-365, engram.py)

At layers 1 and 14, before the block: a lookup of n-gram hashes added into
the 4 streams.

- **The hashes** (engram.py:129-184): each token id goes to a compressed id
  (a normalization of its text: NFKC, NFD, no accents, lower case, runs of
  white space to one; 99092 ids, checked against
  engram_compressed_vocab_size); for each position the 2-, 3-, and 4-grams
  ending there (pad id for no history; an image span stops them), each
  hashed for 8 heads: XOR of id times an odd multiplier (numpy
  default_rng(10007 layer)), modulo a prime of its own (the primes after
  16000000, in order, never reused), plus the offset of that prime's range:
  24 rows a token for each of the two layers.
- **The rows:** 24 rows of 256 FP8 values (an E8M0 scale for each 32) =
  6144 values; kv = wkv (FP8, 25600 x 6144) of them: a key for each stream
  (4 x 5120) and a value (5120).
- **The gate** of stream h: dot = sum(h . q_weight k_weight . key_h) times
  rsqrt(mean h^2) rsqrt(mean key_h^2) 5120^-0.5; gate = sigmoid(signed
  sqrt(dot)); h += gate value (model.py:350-365).
- **The cost:** 48 random rows (264 bytes each) a token from a 203 GB table;
  wkv is 157 M FP8 weights a layer.

### 2.4 DSpark (model.py:1020-1156)

Three blocks (window only), 128 routed experts top 3 (V4: the main
experts' shape), the shared expert, the embed and head of the model; the
input main_x from the attention inputs of layers 37, 38, 39 (the mean over
the streams, 15360 values, main_proj FP8); a block of 5 (the last token and
4 noise tokens 128799); the Markov head (rank 256) and the confidence head.
The reference implements the forward only (forward_spec); the loop is ours
(speculative.stream).

### 2.5 Vision

The ViT and aligner of V4 (32 blocks of 1024), up to 1024 tokens an image
(V4: 384), min 544 x 544 pixels; image_token_id 129264 on every position of
an image span; the span in one chunk; bias_vl for its routing; no engram
for its tokens.

### 2.6 The chat format (encoding/encoding.py)

DSML tags with a leading space (`<｜DSML｜ calls>`, ` invoke`, ` parameter`);
reasoning effort as a number 1-100 (low 50, high 75, max 100; the default
75), only in thinking mode at the start; mid-conversation system messages
(`<｜System｜>`). evaluation/dsh-minimal.patch is the DeepSeek harness of
the user's client.

## 3. The arithmetic: near and exact

The reference rounds the input of every FP8 and FP4 product to E4M3 with an
E8M0 scale for each 32 values (act_quant, kernel.py:98-126; scales rounded
up to a power of two, fast_round_scale); the weights are E2M1 or E4M3 with
E8M0 scales, and its bfloat16 matrices are products on bfloat16.

| | near (the default) | exact |
|---|---|---|
| MXFP4 experts | as stored; x as int16 in two int8 planes (kq_quant_part16 / k_quant_x2, a scale for each 32): the exact int32 sums 128 hi + lo of the codes times 2 (E2M1 times 2 is an integer), then the E8M0 scales: kq_nvx_rows2 / k_kqh_nvx_q with an E8M0 scale for each 32 in place of the E4M3 one for each 16 | x rounded as the reference (E4M3, E8M0 for each 32); products in float (3.1) |
| FP8 matrices (32 x 32 blocks) | as stored; float16 x on the GPU (E4M3 is exact in float16), float32 on the CPU | x rounded as the reference; the same kernels |
| bfloat16 matrices (embed, head, router, compressors, indexer weights_proj and wk, vision, aligner) | BF12 (its kernels: k_kq_bf12_tc and the rest) | bfloat16 |
| the cache, the indexer | as the reference (2.2): FP8 window rows, FP4 compressed rows, FP4 index keys and queries | the same |
| the check | KL against exact and against the reference at the floor of this MoE (Phase 2) | bit for bit with the reference's products on the same values; the float32 sums in another order the only difference |

### 3.1 The exact products

Every product of exact mode is a product of values exact in float16 (E2M1,
E4M3 times a power of two):

- **The GPU:** float16 tensor cores (sm_80) with float32 sums, a block of 32
  as two k-steps of 16, then times the two scales (x and w, the same 32
  columns); FP8 32 x 32 blocks the same with one scale a block. A step of
  one token on CUDA cores: the codes to float16 pairs, fma in float32.
- **The CPU (no AVX512-FP16 or BF16 on the 8268):** the codes to float32 by
  vpermps (E2M1) or a table of 256 (E4M3), fma with x in float32, the
  scales once for each 32. E4M3 x is not an integer of 16 bits (a block
  spans 2^17), so the integer kernels of near cannot take it.
- **The cost:** a decode step stays bound by the memory (4.5 GB of experts
  a token); a prompt on the CPU is several times slower than near (float
  FMA against VNNI), so exact runs its prompts on the GPU.

### 3.1a The BF12 check of the checkpoint

scripts/check_dsv41_bf12.py runs the BF12 converter (cops.kq_bf16_to_bf12)
on every matrix of the checkpoint, with the rule of the runtime
(convert_q8_gguf.py --dense bf12): cols a multiple of 32, more than one row,
no Inf or NaN, the largest zeroed value at most 2^-8 of the RMS of its
matrix.

- **The bfloat16 matrices: all BF12 but two small ones.** 202 matrices, 1.975
  G values: 81308 zeroed (0.0041%; Qwen3.8 0.0055%), 246 collisions, the
  worst largest zeroed 3.9e-4 of its RMS (layers.14.attn.compressor.wgate;
  Qwen3.8 1.65e-3). embed and head 0.0041% zeroed each, the largest 1.3e-4
  and 1.7e-4 of their RMS. Kept: vision.patch_embed.proj (1024 x 588: 588 is
  not a multiple of 32; bfloat16, 1.2 MB) and mtp.2.confidence_head.proj
  (1 x 5376: one row, a dot product; float32).
- **The FP8 matrices stay FP8, unchanged** (the bytes of the shards on the
  GPU; BF12 would take 12.25 bits for an 8-bit value). For the record, as
  their exact bfloat16 values BF12 would also pass the rule (355 matrices,
  0.0024% zeroed, the worst 1.35e-3 of its RMS), so nothing in them is
  outside its range; it gives them nothing.
- The vectors (norms, biases, sinks, hc bases and scales) stay float32 (the
  hc_attn_fn / hc_ffn_fn matrices are float32 in the checkpoint and stay so).

### 3.2 Near against exact

Near is finer than the reference in the activations (int16: about -84 dB a
value, against about -30 dB for E4M3) and coarser only in the BF12 values;
the model was trained with the FP8 rounding (the clamps of the SwiGLU keep
it in range), so near is a small shift either way. The measure of Phase 2
decides: KL of near against exact on real text, and the greedy tokens; if
near is not at the floor of this MoE (about 4e-3 on Qwen3.8), exact becomes
the default for that part.

## 4. The machine

### 4.1 Where each part lives

| part | GB | where | why |
|---|---|---|---|
| routed experts | 288.8 | RAM, the rows of each expert split between the nodes (the parts of Qwen3.8, moe_parts.py): 144.4 GB a node | a token reads 4.5 GB of them; both sockets read their half at their own bandwidth |
| DSpark experts | 7.2 | GPU (all 384) | a draft round reads up to 45 of them (0.85 GB): about 1 ms from the GPU, 5-8 ms from the CPU; and 3.6 GB a node less in the RAM |
| engram tables | 203.1 | the Optane modules, one table on each (DAX, read in place: 116 GB of ext4 a module, about 14.6 GB left on each) | 48 random rows a token: 12.7 KB, latency rather than bandwidth; not in the RAM, which the experts fill |
| dense, shared experts, engram wkv, router, hc, norms | 7.5 | GPU | every token |
| head, embed (BF12 in near: about 1.0 each) | 1.32 + 1.32 | GPU (both, BF12, streamed from the shards, 4.2) | the embed a row a token, decoded on the GPU; no host copy |
| cache (3.2) | 0.9 at 1M tokens | GPU | |
| hot experts (MXFP4 blocks of 18.8 MB) | about 5 | GPU: about 260 of 15360 (about 640 with the DSpark experts in the RAM) | HotCache / WarmPool |

- **The RAM of a node** (about 156 GB with balanced channels), measured on
  the Qwen3.8 GPU process of the same structure (the parts, the experts
  page-locked in place for the copies to the GPU: no staging copy, VmPin 0):

  | | node 0 (the GPU's) | node 1 |
  |---|---|---|
  | routed expert rows (the parts, 0.5 each) | 144.4 | 144.4 |
  | the runtime: CUDA context and driver, pinned buffers (the rows of the CPU part, hp, the flags), the CPU MoE scratch, the Python heap (Qwen3.8: 3.3 GB in all, 0.74 of it the heap) | about 3.5 | about 1 (the scratch of its part) |
  | host copies of weights: none (the dense weights and the embed streamed to the GPU, 4.2; the Qwen3.8 loader kept 7.9 GB of staged dense weights) | 0 (two staging buffers while loading) | 0 |
  | the system (the kernel's slab: 9.9 GB at idle on this machine, 1.6 of it unreclaimable; the other processes under 2 GB) | about 2-3 | about 2-3 |
  | left | about 5-7 | about 8-10 |

  The page tables of 290 GB of 4 KB pages are about 0.6 GB: the expert
  rows in large pages (1 GB or 2 MB, 4.2) make them negligible and spare
  the TLB. Room beyond that: the coldest experts (by the counts of
  HotCache on real text) on the about 14.6 GB left on each Optane module,
  each node's own rows on its own module, read in place by that node's part
  (about 6.5 GB/s of one module against about 100 GB/s of a node: only the
  experts used least), and a share a little under 0.5 for
  the GPU's node (its runtime). Mixed groups of 1024 rows keep the CPU
  scratch near 0.4 GB (4096 rows: about 1.6 GB: de and act of 28672 pairs).
- **Balanced channels:** the parts take equal shares (RECLANK_GPU_PART_SHARE
  0.5; moe_parts.AUTO_SHARE0 is 0.55 for the 6/4 channels of today: measure
  again and set it by the layout, or by a calibration of the nodes at the
  load).
- **The page cache:** the shards are on the NVMe (and a copy on the HDD).
  The load reads each expert's rows once into memory of its node
  (MPOL_BIND) and drops the pages of the shards (posix_fadvise DONTNEED);
  289 GB at about 3 GB/s of the NVMe: about 100 s a load, and a test round
  needs many loads: the weightkeeper (4.2).
- **The Optane modules:** two of 128 GB, one on each socket, now two
  regions (not interleaved), each an fsdax namespace with an ext4 of 116 GB
  mounted with dax=always: /mnt/pmem0 on node 0, /mnt/pmem1 on node 1. On
  this machine the firmware's memory map starts the first region 4 KB after
  the NFIT's start, so the kernel cannot reserve a namespace that begins at
  the region's start: that namespace begins 16 MB in (its label moved by
  hand; a re-created namespace would start at the region's start again, so
  fsdax is put on the existing namespace instead of ndctl's reconfigure).
  The Qwen3.8 files (249 GB) no longer fit one module and moved to the HDD.
  Each engram table (101.4 GB: the codes and the scales, a file as in the
  shards) is copied once to its module: layer 1's on /mnt/pmem0, layer 14's
  on /mnt/pmem1. A row is local to the node that reads it: the gather of a
  layer runs on the threads of its module's node (the hashes depend only on
  the token ids, so both gathers can start with the token, before layer 1,
  and wait in pinned rows until their layers).

### 4.2 The weightkeeper: the weights loaded once for many rounds

A process that holds the weights and outlives the test rounds, so a round
(a new build, a new script) starts in seconds, not the minutes of a load.
**Done:** reclank/runtime/weightkeeper.py (the weightkeeper, and attach() for a round),
scripts/weightkeeper.py (the command: `--lock`, `--status`, `--stop`,
`--check`), tests/unit/test_weightkeeper.py.

- **The segments.** The weightkeeper reads the shards once into memfd segments,
  one for each node's expert rows, each bound to its node with mbind before
  its pages are touched, in the largest pages it can get (below). The
  layout (dsv41-expert-planes-v1): for each layer, each expert, the planes
  w1.weight, w1.scale, w3.weight, w3.scale, w2.weight, w2.scale of the
  part's rows (moe_parts.split: the rows of inner for w1 and w3, of hidden
  for w2; `--share`, default 0.5), the bytes of the shard unchanged, each
  plane at a multiple of 64 bytes; strides in a small JSON index. The reads
  use O_DIRECT (no page cache), each expert matrix read once and its rows
  copied to both nodes.
- **The dense weights go straight to the GPU** (18.7 GB in the shards:
  attention, shared and DSpark experts, embed and head, vision). Next to
  144.4 GB of expert rows they do not fit the RAM of either node (4.1), and
  the GPU is where they run, so the weightkeeper does not hold them: a round
  streams them from the shards to the GPU, tensor by tensor (O_DIRECT reads,
  no page cache), through two pinned staging buffers of a few hundred MB on
  the GPU's node: the FP8 matrices and their scales as their bytes, the
  bfloat16 matrices as BF12 (cops.kq_bf16_to_bf12 on the CPU between the
  read and the copy, all but the two of 3.1a; exact mode: bfloat16), the
  vectors as float32. No host copy stays: the embed too goes to the GPU in
  BF12 (1.0 GB; a row a token decoded there), which takes 1.3 GB off the
  GPU's node, the tightest memory of the plan (4.1), for 1 GB less of hot
  experts. About 9 s a round at the 2.2 GB/s of the NVMe. `--dense` puts
  their bytes in a weightkeeper segment on the GPU's node instead (a subset of
  layers, a machine with room). The engram tables stay files on the Optane
  modules.
- **The large pages, whenever possible.** In order of preference, per
  segment and per node:
  1. 1 GB hugetlb pages (memfd_create with MFD_HUGETLB | MFD_HUGE_1GB, from
     the node's own pool: nodeN/hugepages/hugepages-1048576kB/nr_hugepages);
  2. 2 MB hugetlb pages, for what the 1 GB pool could not give;
  3. shmem transparent huge pages (2 MB, MADV_HUGEPAGE), when
     transparent_hugepage/shmem_enabled allows them (this machine: never,
     so root sets advise for the weightkeeper's life);
  4. 4 KB pages.

  Filling a pool needs root, and 1 GB pages need 1 GB of contiguous free
  memory on the node. So the weightkeeper fills the pools first: it drops the page
  cache, compacts each node (nodeN/compact), asks each node for the count it
  needs, and takes the rest in the next size down. The count is a node's
  segment rounded up: 145 pages for its 144.4 GB of expert rows. Without
  root, it uses pools that were filled beforehand (hugepages= on the kernel
  line, or a unit at boot); no privilege is needed to take their pages.
  Each segment starts at a multiple of its page size in the index, and a
  round maps the same offsets, as hugetlb requires. It prints the share of
  each page size per node at the start. The rounds gain too: 1 GB pages
  make cudaHostRegister of 290 GB pin about 300 pages instead of 70
  million, and the TLB covers the rows of a whole layer.
- **The lock (`--lock`, started as root).** With this parameter the weightkeeper
  fully locks down the memory of each node: nothing the kernel does moves,
  swaps, reclaims, or changes a page of the weights while it runs.
  1. **Not swapped or reclaimed:** RLIMIT_MEMLOCK raised to unlimited, and
     every segment mlock()ed once it is filled. hugetlb pages are never
     swapped anyway; this also covers the 2 MB and 4 KB fallbacks. This
     machine swaps to zram (157 GB) in its own RAM, so without the lock
     cold expert rows could be compressed and decompressed under pressure.
  2. **Not moved:** the MPOL_BIND policy of each segment (no
     MPOL_F_NUMA_BALANCING) makes NUMA balancing skip them; hugetlb VMAs are
     skipped as well. vm.compact_unevictable_allowed set to 0, so
     compaction leaves the locked fallback pages where they are; 1 GB pages
     are never compacted. After the fill, a check of /proc/self/numa_maps
     confirms that every page of a node's segment is on that node.
  3. **Not changed** (with or without `--lock`): after the fill, the weightkeeper
     drops its writable mapping and seals each memfd (F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW |
     F_SEAL_SEAL), keeping a locked read-only mapping. A round cannot write
     a weight or resize a segment, even by mistake.
  4. **Not root longer than needed:** before the fill, a small root parent
     forks the weightkeeper. The weightkeeper drops to the invoking user (SUDO_UID and
     SUDO_GID, every capability gone) once its pools, locks and seals are in
     place; locked pages stay locked. Its socket is 0600 and owned by that
     user. The parent does nothing but wait. When the weightkeeper exits, for any
     reason, the parent gives the pools back (nr_hugepages as before),
     restores shmem_enabled and compact_unevictable_allowed, and exits.
  5. **All or nothing:** `--lock` without root, or with a node that cannot
     be locked in full, refuses to start and names what failed: no partial
     lock. Without `--lock`, the weightkeeper takes what it can get and says so.

  The cost: locked memory is out of reach of the kernel's reclaim, so under
  pressure the OOM killer picks other processes. The 5-7 GB left on the GPU's
  node (4.1) must hold a round, and the hugetlb rounding (under 1 GB per
  segment and node) counts against it. The engram tables are DAX files on
  the Optane modules: not RAM, so nothing to lock (MAP_POPULATE maps them
  in 2 MB pieces at the start).
- **The handoff.** A round is a fresh process (exec: the new code and the
  new libraries): it connects to the weightkeeper's Unix socket, receives the
  segment fds (SCM_RIGHTS) and the index, and maps them read-only (MAP_SHARED);
  no copy, the pages stay where the weightkeeper bound them. The round page-locks
  what the GPU copies from (cudaHostRegister on its mapping: about 2 s for
  70 GB measured on Qwen3.8, so about 8 s for 290 GB, once a round) and
  uploads the dense part to the GPU. A fork of a loaded process would freeze
  its code and libraries in the child, and a CUDA context does not cross a
  fork: the weightkeeper never touches CUDA.
- **The memory.** The segments are the memory of the weights, counted once
  whatever the rounds (one at a time; two rounds at once share them but each
  needs its own GPU). The weightkeeper holds nothing else (no CUDA, no programs).
- **The life of the segments.** They go when the weightkeeper exits (memfd); a
  `--keep` mode (not done yet) puts them in files (the same mbind; on a hugetlbfs mount of
  each page size when the segments are in hugetlb pages, else /dev/shm) so
  even the weightkeeper can restart (with `--lock`, the parent keeps the pools
  until the files are removed); a checksum of each shard's header and size in the index
  so a round never maps weights of another checkpoint.
- **For the server too:** serve maps the segments in the same way when a
  weightkeeper runs (a restart of the server in seconds), else loads as before.
- **Measured** (as the user, so 4 KB shmem pages; 4 layers, 28.9 GB):
  populate 14.4 s, the reads 13.1 s at 2.2 GB/s (the NVMe gives 2.0 GB/s to
  dd with O_DIRECT); every page on its node (bind:0 / bind:1 in a round's
  numa_maps); a round maps both segments in 0.98 s, and 40 random experts
  of each part match the shards byte for byte. Faulting the pages in the
  fill instead of populating them first took 31.7 s. All 40 layers: about
  130 s of reads, plus populate (4 KB: about 145 s; 1 GB pages as root:
  seconds). As root with `--lock` (4 layers): the 1 GB pools filled in 8.6
  s (the page cache dropped, the nodes compacted), 15 pages a node populated
  in 7.3 s, the reads 13.1 s, every page on its node, ready in 20.7 s, then
  served as the user. A round maps both segments in under 10 ms (30 pages
  of 1 GB), the 40 experts match, and a writable map or a resize of a
  memfd is refused. The hugetlb memfds take F_SEAL_FUTURE_WRITE: this
  kernel refuses F_SEAL_WRITE on them once they hold pages (EBUSY). mlock
  leaves VmLck at 0 for hugetlb maps (the kernel does not mark them locked;
  their pages are never swapped, and 1 GB pages never compacted); it
  locks the shmem fallbacks.

### 4.3 The decode rate (an estimate)

A token: 6 x 40 experts x 18.8 MB = 4.5 GB of experts, about 98% cold at
first (260 hot of 15360, the DSpark experts on the GPU). Two balanced
sockets at about 100 GB/s each read their half in about 22 ms; the GPU part
(the dense 7.5 GB at about 700 GB/s: 11 ms) comes before the CPU part in
each layer (the experts wait for their input, as in Qwen3.8), so about 33
ms: 25-30 tok/s plain at best, the CPU part's bandwidth bound, more with the
hit rate of the hot experts and with DSpark (a round of 5 drafts reads its
experts from the GPU). The measure of Phase 7 replaces this.

## 5. Into reclank

| V4.1 part | Nearest code | New |
|---|---|---|
| MXFP4 experts | the NVX paths (kq_nvx_rows2, k_kqh_nvx_q, qmoe_tc) | type KQ_MXFP4: E8M0 for each 32; int16 x (near), float products (exact) (3) |
| FP8 dense (32 x 32 blocks) | GP_KQ_LINEAR, KQ_MULTI | type KQ_FP8B32; the grouped wo_a |
| the parts on balanced nodes | moe_parts.py, GP_KQ_MOE_PART, GP_MOE_COMBINE (the rows form) | MXFP4 rows; the share by the layout |
| hot / cold experts, the pool | HotCache, WarmPool, GP_HOT_SPLIT | blocks of 18.8 MB; 384 experts a layer |
| the router | ROUTER_TOPK | sqrt-softplus, the bias for the choice, bias_vl |
| hyper-connections | HC_PRE / HC_ADD of Qwen3.8 | the Sinkhorn of 20; the mixes used one sublayer later |
| the window, the sinks | the sliding layers of Gemma 4 | the FP8 window rows |
| the shared compressed KV, the indexers, the candidates | QSA_SELECT (qsa.c) | the sources and their readers; the pooling of 2; the FP4 cache (E4M3 for each 16: the NVFP4 group form); the two-level top-k |
| engram | the n-gram layer of Qwen3.8 (PLE_CONV, the per-layer embeddings on the module) | the hashes (a C port of engram.py; the compressed ids once at the load, checked against 99092), the gather from the module, the gate |
| DSpark | the MTP of Qwen3.8 (speculative.stream) | the block draft, the Markov head, the confidence as the stop |
| ViT, aligner | encoders.py, kernels/enc | 1024 tokens an image |
| the chat format | a port of encoding.py | the V4.1 tags, effort, system messages |

## 6. The steps

### Phase 0: a reference

The reference needs CUDA, tilelang, and the whole model on its GPUs. Its
ops have plain torch forms on the CPU (act_quant and fp4_act_quant: the
rounding of fast_round_scale; sparse_attn: a gather and a softmax with the
sink; hc_split_sinkhorn; the GEMMs on dequantized weights of each layer),
so the model runs in torch on the CPU, layer by layer from the shards (in
its own virtual environment; torch on the CPU needs no NVML): the logits
and each layer's output for a few short prompts at position 0 in one
chunk, then steps of one token, and one image (scripts/dsv41_reference.py).

**Done (text):** scripts/dsv41_reference.py runs the checkpoint's model.py
as it is, with scripts/dsv41_ref/kernel.py in place of its tilelang kernels
(the kernels' roundings: the power-of-two scales of fast_round_scale, E4M3
and E2M1 to nearest even, the sink of sparse_attn, the probabilities in
bfloat16 before V), layer by layer from the shards (a few GB beside the
weightkeeper): 160 s for a prompt of 16 tokens, about 30 s a step. "What is the
capital of France? Answer in one word." gives "Paris". It writes for each
prompt the ids, the logits, each layer's output, the experts of each token,
and the greedy steps; `--long-file` adds a prompt of 1400 tokens (the
compressed KV and the indexers past their top 512); `--dump-layers` the
inputs and intermediates of chosen layers. Vision and DSpark: later.

### Phase 1: the reader, the weightkeeper, and the types

1. formats/dsv41_st.py: the view of the checkpoint (names, the config, the
   types), the address tables of the experts in the shards, the engram
   tables; nothing written but the byte copies of 4.1, and in near mode the
   BF12 forms of the bfloat16 matrices, made at the load (in the weightkeeper's
   segment, once).
2. The weightkeeper (4.2, done) and the loader that maps its segments, or loads
   alone when none runs; the large pages and `--lock`; the measure of a round's
   start, and of a decode step in 1 GB pages against 4 KB.
3. KQ_MXFP4 and KQ_FP8B32 on the CPU and the GPU, both modes: near (int16 x
   for MXFP4: the int8 kernels of Qwen3.8 with the E8M0 scale; BF12 for the
   bfloat16 matrices) and exact (the act rounding of the reference as a
   record, float products: AVX-512 FMA, sm_86 float16 tensor cores; Pascal
   float32): GP_KQ_LINEAR, the MoE kernels, the pool blocks.
4. Check (scripts/check_dsv41_types.py): the dequantized values against the
   reference's, bit for bit; the products against NumPy on those values.

**Done on the CPU** (csrc/kernels/kq/mxx.c, formats/dsv41_st.py,
scripts/check_dsv41_types.py): KQ_MXX (64), the MXFP4 codes in the groups
of 16 rows of KQ_NVX with the 16 E8M0 scales of each block (272 bytes for
32 columns; a scale 0 or 255 is counted at the pack: none in the experts),
and KQ_FP8B32 (65), the FP8 rows as stored with their 32 x 32 scales. On
real experts: the packed values equal the checkpoint's, bit for bit; the
near product (int16 x, the exact int32 sums, p = a 2^(e - 127), an fma a
block) equals a NumPy form of the same operations on every output; the
act rounding of exact mode equals a NumPy form picking among the 256 E4M3
values. Against products of the unrounded x: near -82 dB, exact (the
reference's E4M3 x) -31 dB. The weightkeeper's rows equal the shards'. The
weightkeeper's layout v1 (planes) is packed to KQ_MXX at each use for now; a v2
layout in KQ_MXX at the next start of the weightkeeper. The GPU forms: Phase 3.

### Phase 2: the model on the CPU

1. models/deepseek41/: the config, the cache (window rings, the source
   caches, the compressor states, the index keys, the engram hash state),
   the layer in NumPy, the head.
2. Check against Phase 0: exact mode each layer (dB), the logits (KL), the
   greedy tokens; near against exact (3.2); a prompt in chunks gives the
   bits of one chunk (the pooling state of 2.2, the hash state of 2.3).

**Done (1 and 2):** models/deepseek41/model.py (DSV41CPU: the reference's
forward op for op in NumPy with the KQ_MXX and KQ_FP8B32 products, the
experts from the weightkeeper) and engram.py (the hashes without torch,
tokenizers or sympy: the token map equals the reference's for all 129280
ids, 99092 compressed ids; HF's StripAccents drops every combining mark,
Mn, Mc and Me); scripts/check_dsv41_cpu.py. Each layer fed the reference's
own inputs (`--dump-layers`), exact mode: the window KV, the compressed KV,
the index keys and the top-k bit for bit, the compressor -142 to -176 dB,
the attention output -59 dB, the MoE -62 to -64 dB, the layer's output -73
to -79 dB. A whole prompt ("Paris", 16 tokens): the same last token and
decode step in both modes; exact: layers -65 dB at first, then the coarse
roundings of the model amplify the small differences (a value at the edge
of an FP4 step of the compressed KV, a router near-tie: the routes of 3 of
16 tokens differ from layer 3 on), so the layers sit at -35 to -52 dB, KL
3.6e-2 (forced to the reference's experts: 1.1e-2). Near on the same
inputs as the reference: the attention -25 dB, almost all of it the
compressed KV (-28 dB: from float32 latents in place of bfloat16 ones its
FP4 rounding picks the next step for many values; both are draws of the
model's own FP4 noise). The measure of near against exact on real text:
Phase 7.
3. The CPU program: the records of the hyper-connections, the attention
   (window, shared cache, indexers, candidates), engram, the router; the
   experts in the parts.

### Phase 3: the GPU

1. The dense part, the cache, the attention, the indexers on the GPU.
2. The experts: hot on the GPU, cold in the parts (the rows form, the same
   bits wherever an expert runs); mixed groups for a prompt.
3. Engram: the 24 rows of each layer read from its module by the threads
   of that module's node, both layers at the start of the token (the hashes
   need only the ids), into pinned rows for the GPU; the wkv product and the
   gate on the GPU. A copy of the rows to the GPU by the GPU itself (a
   mapped DAX range registered with CUDA) is the other form to measure.

**Done so far:** GP_DS_MXX_LINEAR (kernels/dsv41/types.cu: the KQ_MXX rows
on int16 x with the CPU's operations; on real experts, 1 and 4 tokens, the
same bits as the CPU) and GP_DS_FP8_LINEAR (a warp a row, -136 to -140 dB
against float64); scripts/check_dsv41_gpu_types.py. The cold experts in
parts take KQ_MXX (GP_KQ_MOE_PART flag 128: the act silu(min(g, 10))
clamp(u, -10, 10) times the weight of the pair before the down rows, as the
reference's Expert, and the down rows summed as they are): -127 to -143 dB
against each expert's products (scripts/check_dsv41_parts.py). The weightkeeper's
layout v2 holds the experts in KQ_MXX (the parts read it in place).
**The step (done):** kernels/dsv41/step.cu has the records of a decode step
(GP_DS_HC_MIX, _HC_PRE, _HC_POST, _ROPE, _FP4, _WIN_WRITE, _COMPRESS,
_CMP_WRITE, _IDX_SCORE, _CAND, _TOPK, _ATTN, _ROUTER, _SWIGLU, _ENGRAM; the
position is read at run time, so one graph serves every step; the
compressor's records act only on the step that ends a group); wo_a is
GP_DS_FP8_LINEAR with 8 groups. models/deepseek41/gpu.py (DSV41GPU) builds
the program: the dense weights to the GPU once (FP8 as they are, the bfloat16
matrices and the head as BF12 through GP_KQ_LINEAR; the host keeps only
untouched placeholders), the routed experts split: GP_HOT_SPLIT, the hot ones
on the GPU (GP_DS_HOT_MOE: whole experts in KQ_MXX in slots, the operations of
the CPU parts, rc_dexp), the cold ones by GP_CPU_TASK of GP_KQ_MOE_PART on the
weightkeeper's segments (ExpertParts.from_desc; the rows form, flag 64), and
GP_DS_COMBINE adds the de rows in the order of the router. The engram rows
and the embedding come from the host for now.

scripts/check_dsv41_gpu_step.py, against the CPU model from the same state
(near): the prompt of 16 tokens, 3 steps: the same experts in every layer,
the same top token, logits -40 to -43 dB (KL 1e-8 to 4e-4), layer 0 at -110
dB, the last at about -43; the window rows and the index keys mostly the same
bits (an FP4 rounding flip now and then). The prompt of 1414 tokens (the
top 512 of 707 and 1414 entries): the same top tokens; one step had a near-tie
of the router in layer 11 (one expert of 6 differs, the layers after it at
-17 to -30 dB, the logits KL 9e-9). --hot-check: a step with all its experts
cold and the same step with all of them hot give the same bits (logits and
every layer).

**The time of a step** (the 3090, the experts all cold but 6 slots a layer;
GPUProgram.profile): about 73 ms a step in the graph; the CPU parts 77 ms of
the profile's 104 (1.9 ms a layer), the FP8 products 11.4 ms (the 7.5 GB at
the bandwidth), the rest of the GPU about 15 ms. The parts alone: 0.21 ms an
expert, linear in the experts, the same with 12 threads as 18: bandwidth.
**The slow DRAM:** a read of each 1 GB page of the weightkeeper (18 threads of its
node): the first 25 GB of node 0's segment and the first 16 GB of node 1's
read at about 18 GB/s, all the other pages at about 71 GB/s; the slow pages
are local (from the other socket 13-16 GB/s). The likely cause: 5 DRAM
modules and the Optane module in the 6 slots of each socket, so a 4-way
interleave set and a 1-way set of about 32 GB (to check: sudo dmidecode -t
memory). A token reads 6/384 of every byte, so the 25 GB cost node 0 about 22
ms a token whatever the layout (its parts at least about 48 ms); with all
pages fast about 32 ms. Moving the weightkeeper out of the slow pages saves at
most about 8 GB of node 0 (it needs 145 of the 128 fast); the fix is the
modules (6 DRAM a socket: about 105 GB/s a node), else the experts that the
slow pages hold kept hot on the GPU first, and DSpark (a verify group reads
an expert once for its tokens).

**The groups of a prompt (done):** the records take t tokens (the window rows
of a group in kvq and GP_DS_RING_PUT after the attention, the compressor over
the groups that end in the chunk with the state for the rest, the indexer, the
candidates and the top k for each query, the attention over the ring, the
group's rows and the selected entries, GP_HOT_SPLIT_MT and the parts' rows by
pair); DSV41GPU.prefill runs a prompt in groups (RECLANK_DS_GROUP, 128; then
the powers of two below). The FP8 products of a group run on the tensor cores
(k_fp8b32_tc: mma m16n8k8 tf32, x in two parts; -127 to -129 dB). The parts
split their units by the pairs of each expert (kq_part_units: a real prompt's
skewed choice gave one thread twice the work).

scripts/check_dsv41_gpu_ref.py against the reference (the prompt on the GPU
in groups of 512, the steps teacher forced), the same as the CPU model in
near mode: prompt1 steps mean KL 8e-2 (CPU 7e-2), prompt2 6e-8 (CPU 3e-7),
prompt3 (1414 tokens) 2.1e-2 (CPU 1.3e-2), the same top tokens (17, 17, 15
of 17).

**The times now** (the 3090, real prompts): the prompt of 1414 tokens in 18 s
(the CPU model: 403 s): the parts 4.2 s of a group of 512, the FP8 products
0.8 s. A step: 63-70 ms (14-16 tokens/s) with 11 GB of hot experts (14 slots
a layer); its profile at 1.5K of context: the parts 38 ms (about 4 cold
experts a layer at the bandwidth of the slow pages), the FP8 products 11 ms,
the hot experts 4 ms, the attention 3 ms (its keys split in 8 ranges and a
join: it was 16 ms), the rest about 9 ms.

**The hot experts (done):** DSHot: a score for each expert (decay 0.98 a token
plus its uses), after each run at most 4 copies a token into the slots (the
copy worker, rc_gpu_cache_copy, from the weightkeeper's segments; the map entry of
a slot cleared before, set at the first run after the copy). Exact: the KL is
the cold one's to the last digit. Held: 30-34% of the pairs on prompt3's
steps, 14-22% on the short prompts (a simulation on prompt3's real routes:
44% with 16 slots and 4 copies a token). CUDA does not page-lock the sealed
memfd (cudaHostRegisterReadOnly: operation not supported), so the copies
stage through the CPU; about 1 ms a step.

**The caches (measured on the real prompts):** cache="min", the default,
holds each cache in the smallest form that keeps its values exactly: the
window rows in FP8 (E4M3 codes, an E8M0 scale for each 32), the compressed
rows in FP4 (E2M1 codes, an E4M3 scale for each 16: the form of the model),
the index keys in FP4 (an E8M0 scale for each 32); the sign of a zero kept.
About 0.9 KB a position (0.9 GB at 1M; float32 6.4 KB, bfloat16 3.2 KB). The
logits of the prompts and of every step the same bits as float32 and as
cache="bf16" (the other form; float32 is gone). The decode of the codes is
bit operations (no tables in local memory), the indexer decodes a key once
for the warp, and the attention of a group of a prompt (k_ds_attn_g: 16 heads
a block, one pass with an online softmax, each chunk of keys decoded once)
makes the two forms the same speed: a group of 512 6.1 s in both (its
attention 1357 / 878 ms before, now 769 / 747), a step about 60 ms. The group of a prompt (group, RECLANK_DS_GROUP): 512 by
default (the prompt of 1414 tokens: 128 55 tokens/s, 256 69, 512 80, 1024 80
with 1.4 GB more of the GPU; the same logits). rq8: the bfloat16 matrices in
RQ8_0 in place of BF12 (8.5 bits a value, not exact): the KL to the
reference within the noise, but the weights save only 0.37 GB (the FP8
matrices are 8 bits already) and the GPU's use went up 0.84 GB (the int8
scratch of the Q8 products), so it stays off.

**1M positions by default:** DSV41CPU's max_len is the model's (1M) unless
given; the GPU's caches then take 0.93 GB from the start (the GPU's use 24.3
GB with 13 GB of hot experts, 0.92 more than at 4096). The kernels read only
the entries below each query's position, so a step costs the same. What grew
with max_len is the scratch of the programs: the index scores [t, entries]
float32 (4 GB a layer for a group of 1024 at 1M) and the rope tables (0.5 GB
at 1M). The programs now serve the positions below cap (RECLANK_DS_CAP,
8192 at first): the scores of every index layer in one array [t, cap] (each
layer reads its own in that layer), the tables of cap rows; a run past cap
(with room for DSpark's drafts) doubles it and the programs, their scratch
and the tables go (DSpark's programs too), made again when run. When a
program's scores do not fit RECLANK_DS_SCORE_MB (512; t x cap x 4.5 bytes
with the blocks' max) they go in chunks of entries (a power of two; a group
of 1024 at cap 128K: 64K): each chunk's scores (DS_IDX_SCORE, DS_CAND from
an entry c0), then DS_TOPK_MERGE: the k best so far and the chunk's, ties to
the lower index; the ones so far are all below the chunk, so the choice is
DS_TOPK's over the whole row (the candidates' blocks the same way, their
mask on the last chunk). A program whose scores fit (the steps, the verify
groups, the groups below about 100K positions) keeps the records of before;
the group no longer shrinks at a long context (the candidates' mask stays
[t, cap / 8] bytes: 128 MB for 1024 at 1M). The logits of the real prompts
the same bits at cap 8192, with cap growing (1024 to 2048 in prompt3), with
the scores in chunks of 1024 and of 512 (RECLANK_DS_SCORE_CHUNK, a check);
DSpark the same tokens (accepted 0.70-0.71) with cap growing in its loop and
with chunks of 512. With 13 GB of hot experts and DSpark the GPU had no room
for a graph (cudaGraphInstantiate): 12 GB with 1M of cache. reset zeroes the
caches on the GPU (the host copies are the last load_state's or
save_state's).

**The prompt (prefill), measured on prompt3 (1414 tokens), OMP_NUM_THREADS=1,
the parts' team of 21 a node, 13 GB of hot experts:**

- *The decoder's bounded replay* (replay=True, the default; V4.1 2.2, 3.2.2):
  a prompt of more than the window runs the encoder (layers 0..19, then the
  compressed rows and index keys of layer 20, the decoder's global KV) on
  every token, and the decoder (layers 20..39) on the last 128 tokens only,
  its window from the first of them (operand 18 of GP_DS_ATTN, mode 2); the
  encoder's outputs of those tokens go to the decoder's input with
  GP_DS_ROWS_AT. This is how DeepSeek serves the model, simulated in its
  post-training; it is not the bits of model.py's full forward
  (--full-prefill keeps that). Against the full forward: KL 4.8e-3 at the
  prompt, 8.7e-4 on the steps, the top tokens 17 of 17; to the reference
  6.2e-2 (the full forward 7.6e-2). 15.1 s -> 9.3 s. A prompt of at most
  128 tokens runs as before (the same bits). DSpark: its rings take the
  decoder's 128 rows; the same tokens as plain steps, acceptance as with the
  full forward (0.69 and 0.66 over 48 tokens).
- *Two halves* (ab=True, the default, a group of 64 tokens or more): the
  records of the halves A, B of a group interleave so that the CPU's experts
  of one half run while the GPU runs the attention and the FFN of the other
  (the runner already runs the CPU tasks in order while the GPU waits only at
  each GP_AWAIT; no runtime change). The same bits (each half a group).
  The tasks of the CPU now wait 0.1 s for the GPU in all (2.6 s before).
  The CPU's work of a half re-reads the experts' weights: with groups of
  512 its busy time went from 3.9 to 5.0 s; with groups of 1024 (halves of
  512) it is 4.5-4.9 s.
- *The groups:* the largest multiple of 128 tokens up to the group (2048 by
  default, RECLANK_DS_GROUP), the powers of two below 128 (GROUP_STEP): each
  group reads every cold expert again (up to 7 GB a layer a pass, whatever
  the tokens), so prompt3 runs as 1408 + 4 + 2 (with 1024: 1024 + 384 + 4 +
  2). Each doubling of the group takes 1.2 GB more of the GPU (fewer hot
  experts).
- *The GPU's experts of a large group:* k_ds_hot_tc (and the grouped kernels
  without tensor cores) listed the pairs of an expert in shared memory, at
  most 3072; a lane of more than 512 tokens (a half of a group over 1024)
  fell back to a block for each pair (the rows read again for every pair):
  DS_HOT_MOE of a group of 2048 3.8 s against 0.21 s for 1024 (rc_gpu_profile
  of a group). The pairs now go in chunks of 3072 tokens (an expert at most
  once a token), the same bits. This was the GPU "the longer of the two"
  with groups of 2048 (4.5-4.9 s before).
- *The scratch shared:* the programs of the groups run one at a time, so the
  arrays of each (its intermediates, the pinned rows to and from the parts)
  are views of one set of arrays made for the largest group (_scratch), and
  the runner's scratch of each program (rc_gpu_load: xh, kqx, w16, btp, btc,
  e3) is shared the same way with a count of its holders. 0.5 GB less of the
  GPU with groups of 1024 (22.1 GB with 13 GB of hot experts).
- *The CPU's experts* (moe_part.c, mxx.c): the units of the gate, up and
  down rows taken dynamically by the threads of each part (chunks of 16, a
  counter of the part) in place of a static split by the pairs: the waits at
  the barriers went from 0.65 s to 0.05 s; the weights of a step of a block
  decoded in 3 operations of ports 0 and 5 (a broadcast load, a masked
  shift, an and, the table; 5 before); the tokens past the fours in split
  chains. perf in a group of 1024: 2.0 GHz (the AVX-512 license 92% of the
  cycles), IPC 2.25, ports 0 and 5 52% busy, 16% of the cycles stalled on
  memory: the kernel's own operations (the decode, the sums of the planes
  and scales of each block for 6-9 tokens an expert) are the bound, not the
  DRAM (software prefetch: no change).
- *The times:* prompt3 (1414 tokens): the first prompt of a process 6.5 s;
  then 4.0-4.3 s (332-351 tokens/s) in groups of 2048 with 12 or 13 GB of
  hot experts (groups of 1024: 4.4-4.65 s). The CPU's tasks are busy 3.6 s
  of it. A real prompt of 4414 tokens (DEEPSEEK_V4_PLAN.md cut, then "Summarize
  the text above in two sentences."): groups of 1024 (13 GB) 12.9 s (342
  tokens/s), 2048 (11 GB) 12.3 s (360), 4096 (9 GB) 12.1-12.5 s (354-365);
  the logits the same bits for all.
- *8-bit inputs of the experts* (x8_prefill, an option; the steps keep 16):
  KL to the 16-bit run 6e-3 (prompt3's steps), the top tokens 15 of 17; to
  the reference 1.5e-1 at the prompt (6.2e-2 with 16 bits). Their low planes
  are 0, so the KQ_MXX kernels take the high planes alone (kq_mxx_rows1:
  half the dot products, the same bits as both planes): the first prompt
  5.5 s against 6.2 s.
- *DeepSeek's own precision* (inference/model.py, kernel.py; V4 5.2.1): the
  activations in bfloat16; the input of every FP8 and FP4 product in E4M3
  with an E8M0 scale for each 32 (act_quant); the products FP8 x FP8 on the
  tensor cores (the experts' FP4 widened to FP8, lossless), float32 sums of
  each block of 32, the outputs in bfloat16; the head, the router, the
  compressors and weights_proj in bfloat16; the window KV in E4M3, the main
  KV in FP4 (E4M3 scales for each 16), the indexer's q and k in FP4, the
  index scores in bfloat16. The same kernels for the prompt and the steps;
  the prompt adds the bounded replay. Ours is finer everywhere (float32
  activations, int16 inputs of the experts, tf32 hi + lo for the FP8
  products).

**Next:** the engine and the server (Phase 4); the small records of a step
fused (about 9 ms); the top k of a long context (one block a query now); for
the prompt: the CPU's experts (the parts are the bound now), the GPU's
scratch shared by the programs (groups of 1024 by default then).

### Phase 4: the engine and the server

The engine, the tokenizer, the port of encoding.py, the parser of the
output (DSML blocks with the V4.1 tags); dsh as the client.

**Done:** models/deepseek41/engine.py, scripts/serve_dsv41.py (python -m server
--model DIR: a checkpoint directory, its config.json model_type
deepseek_v41). The tokenizer: DSTokenizer, the BPE of QwenTokenizer with the
pre-tokenizer of tokenizer.json (a Sequence of Isolated splits in the regex
module, no normalizer): the ids of the tokenizers library on the real
prompts (11K tokens) and on CJK, kana, digits, emoji, code, CRLF. The chat:
the checkpoint's own encoding.py (encode_messages; the tools on the system
message; the reasoning effort of a request: reasoning_effort, a level or
1-100, chat_template_kwargs thinking; default high, 75); a plain completion
after <｜begin▁of▁sentence｜>. The output: the reasoning (</think>), the content,
the DSML calls by the checkpoint's parse_tool_calls (a block not closed is
pending). The decode: DSpark.stream with the server's Sampler (greedy, or
the main model's distribution: a draft taken with its probability). Between
the turns: snapshots after the last <｜Assistant｜> of a prompt (the window rings
of every layer, the compressors' open groups, DSpark's rings: about 3 MB);
the compressed rows and the index keys before a point stay, so a prompt that
starts with a snapshot runs only its new tokens (as steps run, not the
bounded replay of a fresh prompt). Checked by HTTP: a short answer, a stream
with the reasoning apart, a second turn (40 of 74 tokens cached), a tool call
and its result (330 of 465 cached), /v1/completions, /v1/models. Images:
Phase 6. The snapshots also keep the engram hashes' ids of their last 8
positions (a request between two turns, a client's title request, wrote
others there: the first tokens of the next turn had other hashes).

### Phase 5: DSpark

The draft of 5 with the 3 blocks (their 128 experts, top 3), the Markov
head, the confidence as the stop of a draft, the verify group (the pooling
and hash states across the group).

**Done:** models/deepseek41/dspark.py. The drafter on the GPU: main_x from
the mean of the streams at the inputs of layers 37-39 (GP_DS_STREAM_MEAN in
every main program), each block's ring of main rows (the window rope), the
block of 5 over its ring and all its rows (GP_DS_ATTN mode 1), the 384
experts on the GPU (7.2 GB, KQ_MXX, GP_DS_HOT_MOE with the identity map), the
main head on the norm of the last hc_pre, the Markov head one position after
the other (GP_DS_GATHER, BF12 product, GP_DS_MARKOV: the bias, then the
argmax), the confidence (GP_DS_CONF). The loop: the verify length that gives
the most expected tokens a millisecond (the survival of each prefix from the
sigmoids of the confidences; the times of the draft and of each verify
length measured as it runs); the verify group on the main model
(DSV41GPU.verify: programs of t <= 8 give the logits of every row and keep
backups); the refused rows undone (DSV41GPU.restore: the ring rows,
GP_DS_RING_RESTORE; the compressor's open group, GP_DS_COMPRESS_FIX; their
compressed entries and index keys stay: no query reads an entry past its
position, and the true tokens write them again). A row of a verify group has
the bits of a step (the attention splits its keys for t <= 8), so greedy
DSpark gives the tokens of plain decoding: checked on the three real prompts
(scripts/check_dsv41_dspark.py, 160 tokens each).

**Measured** (the 3090, 2 GB of hot experts: 2 slots a layer, the DSpark
experts take the rest): 77-79% of the verified drafts accepted, 2.2-3.1
tokens a round; the confidence holds (above 0.95: 89-99% accepted; 0.5-0.8:
73-84%). The draft 9 ms. The verify group: 61-67 ms for 1 token, 106-110 for
2, 148-154 for 3, 207-264 for 6: each more token about 40 ms, the CPU parts
reading its new experts (6 tokens in a row choose 22.6 distinct experts a
layer, 3.8 a token: the real routes of prompt3). Plain 13.6-14.7 tokens/s,
DSpark 14.8-15.7 (1.00-1.14x).

**The DSpark experts in the RAM of node 1** (DSpark(where="cpu1"), then the
default: a part of node 1's CPUs runs them, GP_CPU_TASK in the draft): the
GPU's memory goes to the main model's hot experts (10 GB: 13 slots a layer),
and plain steps go from 13.6-14.7 to 17.4-17.8 tokens/s, the best now. The
draft takes 20-25 ms there (9 on the GPU), and a verify group still costs
about 30-40 ms a token more than a step, so a draft every round loses
(0.81-0.93x). generate(spec="auto") drafts only while the measured rate of
the rounds beats plain steps (a plain step keeps the rings whole; every 8th
step tries the other): 0.97-1.02x of plain on the three prompts (on prompt3
its rounds gave 3.9 tokens, 94% of the drafts accepted). On this machine the
best is plain steps with the GPU full of hot experts; DSpark pays when a cold
expert costs less: the DRAM at its full bandwidth (6 modules a socket), or
several sequences at once.

**Why DSpark was not faster, and the fixes** (12 x 32 GB of DRAM now, 1M of
cache, groups of 2048; the real prompt of 5018 tokens, its 424 tokens of
plain decoding as the drafts; rc_gpu_profile of each verify length):

| ms | step | verify 2 | verify 4 | verify 6 |
|---|---|---|---|---|
| the CPU's experts | 21 | 36 | 64 | 81 |
| cold experts a layer | 3.8 | 7.8 | 13.7 | 16.8 |
| DS_FP8_LINEAR | 12.2 | 18.8 | 34.3 | 50.7 |
| DS_ATTN | 5.3 | 8.7 | 16.0 | 21.5 |
| DS_HOT_MOE | 4.5 | 6.6 | 9.6 | 19.0 |
| the wall | 47 | 77 | 127 | 178 |

- *The FP8 products* of a verify group went token by token (each word of the
  rows read and decoded again for every token). k_fp8b32_gemv<R, NT>: R rows
  a warp (4, 2 or 1: at least 128 blocks), the tokens 2, 4 or 8 at a time,
  the words of 4 steps loaded first; the operations of each row and token in
  the order of a step (the same bits). 12.2 / 14.4 / 20.0 / 29.1 ms.
- *The CPU and the GPU took turns:* a verify group ran as one lane (the
  attention while the CPU waited, the CPU's experts while the GPU waited).
  Groups of 2 to 8 tokens now run as two halves like a group of a prompt
  (the CPU's experts of one half while the GPU runs the other): verify 2 /
  4 / 6 59 / 91 / 121 ms (were 77 / 127 / 178). The ring backups go by lane,
  the compressor's state of the start of the group from the first lane; a
  lane of one token combines the CPU's rows by pair (GP_DS_COMBINE operand 8:
  it took them for the packed rows of a step, wrong bits with hot experts).
- *auto* chose plain steps: the first use of each verify length (its program
  and graphs, about 300 ms) entered its measured rate; DSpark.warm makes them
  (and the drafts of 1 to 8 rows) before the loop.
- *The memory:* with DSpark 2 GB less of hot experts (HOT_GB_DSPARK, 10: the
  programs of the verify groups and drafts, DSpark's weights and its graphs);
  DSpark's seeds share their scratch (made for the window with the bounded
  replay: only the decoder's group seeds), so do the groups' main_h and the
  two engram layers' keys and values; the decoder's arrays are made for the
  window.
- *What stays:* the CPU's experts. The tokens of a verify group share few
  experts (6 tokens: 21.6 distinct a layer of 36), and each cold one is 18.8
  MB read at 130-160 GB/s, near the DRAM's bound; more of them on the GPU
  (hot) is the lever. Then DS_HOT_MOE (a block a pair).
- *The attention of a step and a verify group* (nsys, ncu): k_ds_attn took
  about 95 us a layer for a query (5.3 ms a step), neither compute (15%) nor
  memory (15%) bound: 60% of its instructions decoded the keys (the source of
  each key, the forms, FP8 / FP4, the scales), once in each of the 16 blocks
  of a query and twice in each (the scores, then the outputs a value at a
  time). Now k_ds_attn_keys decodes each query's keys once a layer into
  float32 rows (DSV41GPU.akeys, 8 x 640 x 512, the programs of t <= 8 share
  it; ds_rld4: the same values), then k_ds_attn_s (4 heads a block, the keys
  split in 8 as before) reads them from L2: the scores of each key and the
  outputs of each column in the same order, the same bits. 32 + 4.5 + 3.2 us
  (the join) a layer: 2.0 ms a step, 7.1 ms for a verify group of 6 (24.1).
  Tried: tiles in shared memory, 512 threads, the next tile in registers (no
  gain: occupancy), 32 splits (no gain, other bits).
- *The hot and the cold experts of a step* (the prompt of 5018 tokens, its
  plain tokens; RECLANK_PART_PROF, rc_gpu_task_stats): a step 42 ms, the
  CPU's tasks busy 22.3 ms and idle 16.2 (the GPU's own work: the layers wait
  on each other). The cold experts (4.7 a layer, 3.55 GB a step) read at
  about 195 GB/s in their two products (gate and up 11.9 ms, down 5.9), near
  the DRAM's bound; the rest is about 4.5 ms of barriers and small phases.
  Splitting the small phases (the act, its int8, the rows of the pairs: about
  5 cold pairs for 21 threads) by groups balanced them but the step did not
  change (the products got slower by as much), so it stays as it was; the
  dynamic split of the products for one token is slower (43.2 ms).
  The 12 GB of hot experts (15 slots of the 384 a layer) save only 6% (45.1
  ms with none): 40% of the pairs held over the 423 steps. Offline on those
  routes: the best possible (Belady) 0.63, the best static set 0.44, LRU 0.42,
  the scores of DSHot with 16 copies a step 0.43; 30 slots 0.60 (no room). In
  the program, 1, 2, 4 or 16 copies a step (RECLANK_DS_HOT_BUDGET, a measure)
  give 38.2-40.3 ms a step: each copy goes through a staging buffer (the
  weightkeeper's sealed read-only map takes no cudaHostRegister here), its DRAM
  traffic costs about what its hits save. The hot experts are not a lever on
  this machine; the cold experts are at the DRAM's rate.
- *The FP8 products of a step* (k_fp8b32_gemv<1, 1>, a warp a row; ncu of a
  layer): wq_a 17 us (381 GB/s), wq_b 60 (705), wkv 15 (177), wo_a 53 (636),
  wo_b 68 (615), the shared expert's three 23-24 each (about 510), the
  engram wkv 191 (827: the kernel can stream); 10 ms a step, the attention's
  8.5 on the path (the shared expert runs while the CPU computes the cold
  experts). The stalls are the loads (long scoreboard 10-19 a issue), then
  the table of e4m3f in shared memory. Each row's sum keeps its order (the
  bits of a step and of the verify rows), so a row stays with one warp; the
  small matrices fill a part of the GPU. The codes now go through a half
  (their bits at 15, 13..7 of a half are the value times 2^-8, the
  subnormals too; the half to float and the 256 exact): 193 ms in 20 steps
  against 202-205 with the table, the same bits. Tried: 8 or 16 steps of the
  columns loaded ahead (no gain, 16 slower), 8 blocks an SM (2 steps ahead:
  2.6% faster with the table, none with the half; 4 steps ahead: spills).

The result on the prompt of 5018 tokens (424 tokens, the tokens of plain
decoding): plain 23.5-23.8 tokens/s (12 GB of hot experts); DSpark (10 GB)
a draft every round 28.8 tokens/s (0.76 of the drafts accepted, 2.9 tokens
a round), auto 26.5; it was 22.0 and 21.4.

**Sampling with DSpark** (generate(temperature, top_p, seed, stop)): the
drafts are the drafter's argmax (one token each), so a draft d is taken with
the main model's probability p(d) of its row (after the temperature and top_p)
and, refused, the row's token is drawn from p without d (renormalised); all
taken, the token after them from the last row: the tokens are draws of the
main model alone. stop ends it (no cap with n None). The recommended
settings (README: temperature 1.0, top_p 0.95), thinking mode with reasoning
effort 50, "Generate an SVG of a pelican riding a bicycle": 10731 tokens at
28.7 tokens/s, 0.75 of the drafts taken, 3.9 tokens a round (plain greedy:
24.8). Greedy stays the tokens of plain decoding.

### Phase 6: vision

The image processor (up to 1024 tokens), the ViT and aligner on the GPU,
the span in one chunk, bias_vl, no engram in the span.

**Done:** models/deepseek41/vision.py. The pixels as image_processor.py
(ImageOps.pad, gray 127; the values rounded to bfloat16: the patches equal
the reference's bit for bit). The ViT as a program of the encoder records
(the ViT of Qwen: ENC_LINEAR on the tensor cores, ENC_ROPE2D with the rows of
q and k of each head in its order, ENC_ATTN, and ENC_SILU_MUL, new), the
matrices BF12 (below); the aligner a second program, the host putting
the rows of each 3 x 3 block together (the columns of the aligner's w1 in
that order). Against the reference's ViT in float32: 2.3e-3 (relative, the
aligner's rows; the reference's own bfloat16 ViT is 3.6e-2 from it). The
span: image_start, the rows in reading order with image_newline after each
row, image_end (bfloat16, the reference's h). In the model (DSV41GPU): the
programs of a run that meets a span (the key vis, made when needed: the
programs of text keep their bits) take vmask and vrows: GP_DS_EMBED_ROWS
puts the rows in the streams in place of the embedding, GP_DS_ROUTER takes
bias_vl for them (two operands more, NULL for text), the decoder of a
bounded replay the mask; their engram n-grams stop at them (Hashes dead) and
their engram rows are zero (so wkv gives zero: the gate's term is zero, as
the reference's shut gate). A span may cross groups. The server: the image
parts of the messages (and of tool results) into encoding.py's blocks, each
placeholder expanded to its span, the rows of the last 8 images kept by the
hash of their bytes, the snapshots keyed by the image of each span; the
encoder's GPU memory freed after each image (it takes 1.78 GB at 1024
tokens: the hot experts leave VISION_GB, 1.8 GB, so 10 slots a layer with
DSpark; the steps 41.8 ms against 41.6 with 13 slots).

**Checked** with the reference (scripts/dsv41_reference.py --image, the
corn of the checkpoint's examples): its greedy answer "This image shows
three ears of fresh corn, with one partially husked..." is the server's,
word for word; the layer-0 experts of the image tokens differ from its
choice in 2.6% (text 4.2%; without bias_vl 45%); the prompt's KL 0.26, the
steps' 6.5e-3 with the same aligner rows (--image-rows), as text prompts
(0.14 and 6e-2 at their prompts). The text logits are those of before, bit
for bit (prompts 1-3 and 16 steps), DSpark the tokens of plain steps. A
chart of 3801 x 2640 (990 tokens): its numbers read right; the encoder 8.7
s (the attention of 8547 patches on the CUDA cores), 2.2 s for 444 tokens.

**The tensor cores and BF12.** ENC_ATTN on the tensor cores
(k_enc_attn_tc: q, k, v to float16, mma m16n8k16, float32 scores, softmax
and sums, the weights p float16; hd a multiple of 8 up to 128, padded to 16
in the kernel, so the hd 72 of Qwen's ViT too; the float kernel with
RECLANK_GPU_ENC_TC=0, which also turns off the tensor cores of
ENC_LINEAR). On the real q, k, v of the ViT regrouped as hd 48, 64, 72, 80
and 128: about 1e-4 from float64 (the float kernel about 4e-6). The ViT's
and the aligner's matrices BF12 (encoders.runtime.to_bf12 under the
runtime's rule; enc_linear takes GP_KQ_LINEAR and ENC_BIAS_CLAMP, x in two
bfloat16 planes; the patch linear stays bfloat16, 588 cols;
RECLANK_DS_VISION_BF12=0 keeps bfloat16, x float16). Against the
reference's ViT in float32: 6.0e-4 (corn) and 8.1e-4 (carrots), against
2.3e-3 and 2.9e-3 before. The encoder of the chart 0.74 s (8547 patches;
0.50 s with bfloat16 weights and float16 x), from 8.4 s; in the server
with the pixels and the upload of the weights 1.87 s, the corn 0.59 s. The
encoder holds 1.70 GB at 1024 tokens (1.79 before). The prompt's KL 0.22
(0.26 before), top tokens 13 of 13; the corn's answer still the
reference's word for word; the text logits the same bits.

### Phase 7: the measures

KL and greedy tokens against Phase 0; the decode and prompt rates on the
3090 with balanced nodes (the share 0.5 against others); the hit rate of the
hot experts; the modules' engram reads (latency of a token, each node on
its own module against both from one node); the context at
24 GB.

## 7. Open questions and risks

- **The RAM.** 144 GB of expert rows on each node of about 156 GB leaves 5-7
  GB on the GPU's node (4.1, measured overheads); the DSpark experts on the
  GPU, the dense weights freed after their upload, huge pages, and the
  coldest experts on the Optane modules (about 14.6 GB a node) are the
  margin. Two copies (as
  Qwen3.8 kept) do not fit.
- **The modules.** A table of 101.4 GB on each 116 GB module. Their random
  reads: 24 a token on each module at about 0.3 us each, in parallel
  threads: well under 1 ms; a prompt of 8K tokens reads 197K rows (52 MB)
  from each. One module alone (not interleaved) has half the bandwidth of
  the pair of before: a prompt is still far from it. The first region's
  namespace depends on its moved label: a firmware update or a re-created
  namespace can undo it (the label area is backed up off the module).
- **Near against exact.** Near must sit at the floor of this MoE against
  exact (3.2); a part that does not (the BF12 of the router, say) takes the
  exact form by default. Exact's CPU products are float FMA: bound by the
  memory at one token (measure), slower for a prompt (the GPU takes those).
- **The indexer on long contexts.** Layer 20 scores every entry of the
  ratio-1 cache (32 heads x 128 a token): at 1M tokens about 4 G products a
  token on the GPU; the candidates bound the other indexers to 16384
  positions.
- **Ties.** The top 512 and the top 2048 blocks: torch.topk of the
  reference against the selection of this runtime on equal scores.
- **Chunks.** The reference compresses a prompt in one chunk at position 0;
  chunks and groups need the states (2.2, 2.3) carried: a prompt in chunks
  of every size gives the bits of one chunk.
- **The engram token map.** It comes from the normalizers of the tokenizers
  library on the decoded text of each id; the port must give the same
  99092 ids (the count is checked; a table of the 129280 ids from the
  reference's code is the test).

**The host and the small records of a step (1017 checkpoint, the weightkeeper, 12 GB of hot experts):**
a step 43.5-44.3 -> 38.6 ms. nsys (cuda-graph-trace node): 1524 kernels, 10 graphs a step; the GPU
waits 23 ms for the CPU's experts (k_await). (1) The head (129280 x 5120 BF12, one token) on the
CUDA-core kernel (k_kq_linear_bf12<1>) in place of k_kq_bf12_tc<8>: 1.81 -> 1.26 ms (a rule of
shape: one token, 4096 columns or more, 16384 rows or more); the logits pinned: their copy 0.22 ->
0.11 ms. (2) k_ds_hc_post made its coefficients in one thread of each block (the Sinkhorn of 4 x 4:
20 iterations of 32 divisions, 25 us a record, 80 a step): now a warp (ds_hc_coef_warp: lane 4 j + k
an entry, each sum gathered by shuffles in ds_hc_coef's order: the same bits). (3) The hot experts'
observe and plan of a step (0.84 ms) run while the GPU ends the next step (rc_gpu_run returns with
the last CPU task, before the head); their copies start a token later (the step's time the same).
The logits the same bits (prompt and steps; RECLANK_GPU_BF12_TC=0 on both sides for the head);
prompt3 against the reference as before (steps KL 3.61e-2, 16 of 17). Not a gain: more copies a
token (RECLANK_DS_HOT_BUDGET 8, 12: the 15 slots a layer bound the hold, not the copies); the link
of this 3090 gives 8 GB/s (Gen3 x16), so a cold expert by DMA (18.7 MB) costs 2.3 ms against the
CPU's ~0.1 ms of a step: no gain from moving the cold experts over PCIe.

**DSpark pays (1017 checkpoint, the weightkeeper):** why it did not, measured (the true tokens as the
drafts, rc_gpu_profile, nsys): a step 39.8 ms (CPU tasks 19.9, the GPU's ~20 serial with them); a
verify group of 2 58 ms, 3 70, 4 89 (the CPU's tasks 39.5 for 2, 69.4 for 4: each row brings ~3.8 new
cold experts, about 5 ms each, the products at the DRAM's rate; the halves A and B overlap, but each
half reads all the FP8 weights again: 41 ms of kernels for 2 rows, so the GPU is the bound there); the
draft 12.6 ms. (1) The DSpark experts in the RAM of both nodes (DSpark(where="cpu"), the default):
each block's rows of gate and up and of down split as the weightkeeper's, a part a node, run by the
teams of the main parts (the draft runs between the main model's programs); its CPU task 5.6 -> 3.1
ms, the draft 12.6 -> 9.5 ms, the same drafts and confidences (where="cpu1" keeps node 1 alone). (2)
The parts (moe_part.c, fuse): a unit of B is 16 rows of gate and the same 16 of up, whose act its
thread makes at once (phase C took a thread for each pair, 27 us a layer, then a barrier): a step
38.8 -> 37.6 ms, the same bits (prompt and steps; RECLANK_PART_FUSE=0 the phases of before); not for
the groups of the dynamic split (a prompt 3% slower: the prompt keeps its phases). Tried, slower: D
(the int16 act) in units of a pair and a group of 32 over all the threads (E 0.4 ms slower a step:
its rows written by many cores), and for a token alone its int16 row read by every pair (no rows of
the pairs nor their barrier: B 2.3 ms slower). The three real prompts, 160 tokens greedy, the same
tokens as plain decoding: plain 28.5 / 29.8 / 26.8 tokens/s, DSpark 31.7 / 28.9 / 32.5 (1.11x,
0.97x, 1.21x; drafts accepted 0.74 / 0.47 / 0.74); before, with the DSpark experts on node 1, 1.00x,
0.99x, 1.02x. Not a gain: the FP8 products of a step (k_fp8b32_gemv) read at 630-800 GB/s on the
large matrices where a plain read of the same size reaches 768 (585 for 12 MB); 16-byte loads, a
block a row (wo_b alone 68.6 -> 59.4 us) and wq_a with wkv in one launch would give about 0.6 ms a
pass. The ceiling on this machine: a verify group of 2 at max(CPU, GPU) (~42 ms) and a draft of ~8 ms
give about 1.3x; prompt2's drafts (accepted 0.47-0.53) are the case where the choice of the verify
length still loses.
The verify length by the cutoff of the MTP drafters (DSpark(stop), dspark.STOP 0.9, RECLANK_MTP_STOP
wins): the drafts while the survival of their prefix (the product of the sigmoids of the confidences)
is at least 0.9, and the first under it when its survival is at least 0.3 (STOP_FLOOR,
RECLANK_DSPARK_STOP_FLOOR). On the three prompts: the rates' choice of before 1.11x / 0.97x / 1.21x;
0.9 with the first under it always 1.08 / 0.98 / 1.21; never 1.09 / 1.04 / 1.08 (rounds of no draft
waste the draft, and "auto" then takes plain steps); floor 0.5 1.14 / 1.04 / 1.18; floor 0.3 1.12 /
1.03 / 1.22 (DSpark 31.5 / 30.6 / 32.8 tokens/s against plain 28.1 / 29.6 / 27.0), the default.
**A verify group's rows the bits of a step again; the small records of a step:** the head of one
token on the CUDA-core kernel (above) had left a verify group's rows on the tensor cores: each row
1.5e-5 from the step's logits (the greedy tokens the same, the bits not). Now the head of 1 to 8
tokens is on the CUDA cores (runner.cu: long rows, 16384 rows or more), 2 to 8 by
k_kq_linear_bf12_r (4 rows a warp, each token's x loaded once for them: k_kq_linear_bf12<NT> read
x again for each row, 160 KB for 8 tokens, more than an SM's L1: 0.75 ms more a token); each row
and token the operations of a token alone: the head 1.39 ms for 2 tokens (tensor cores 1.80), 1.76
for 3 (1.81), 2.61 for 8 (1.90). The rows of verify groups of 2 to 8 (68 rows of the real tokens)
against plain steps: the same bits. A verify group of 2: the hot and shared experts of both rows
after B's signal in one record each (A's are read only by its ffn_post): 51.0 -> 49.4 ms, the GPU
its bound; not for more rows (the CPU the bound: A's experts filled the GPU's wait for B's task, 6
rows 103 -> 106). The step's records on the way to each CPU task (nsys: the gaps between kernels
near 0, so a fused kernel saves its own fixed time only): GP_DS_ROUTER, GP_HOT_SPLIT and GP_SIGNAL
of a step in one kernel (k_ds_route_signal, the rows of x to the host first: 22.4 -> 20.0 us);
k_ds_combine loads the CPU's rows before it sums them (16.9 -> 15.5 us: the PCIe reads of ~94 KB
a layer the rest); the post's q and comb made by GP_DS_HC_PRE's extra blocks from the sublayer's
mixes (k_ds_hc_post 5.7 -> 2.8 us, the pre 6.5 -> 6.9). A step 37.8 -> 37.3 ms, the same bits. The
three prompts: plain 28.7 / 30.1 / 26.4 tokens/s, DSpark 31.9 / 31.1 / 33.3 (1.11x, 1.03x, 1.26x),
the same tokens.
**Sums in another order (GP_DS_HC_STEP):** the records on the way between two CPU tasks that only
the order of their sums kept apart: k_ds_combine (15.4 us: the CPU's de row of each cold pair over
PCIe, ~94 KB a layer) and the post, mixes and pre of each sublayer (2.8 + 8.9 + 6.9 us, twice a
layer). (1) The CPU sums its cold pairs of each token (GP_KQ_MOE_PART flag 2048: float64 rows, 40 KB
a token), the GPU its hot pairs while the CPU's task runs (GP_DS_COMBINE operand 9, float64), and the
post adds the two in float64 and rounds once: a float sum of the CPU's and the GPU's parts would
depend on which experts are hot (the hot set changes from token to token: a verify group's rows were
then not a step's, up to 5 in the logits); six floats add exactly in float64 in practice, so the
float of the total does not depend on where an expert ran, as before. (2) GP_DS_HC_STEP: a sublayer's
post with the next one's mixes and pre in one kernel (64 columns a block, 8 warps; the last block of
a token adds the blocks' sums in a fixed order, scales x, and makes the next post's q and comb), the
attention's post at each FFN, the FFN's at the next layer's attention (not before the engram layers,
nor the first layer of a program, nor the last FFN). 14.7 us a record against 18.6 for the three
(and the combine at the FFN's); the step's records on the way 15.2 -> 14.3 ms (nsys). The pinned
operands of the CPU's sums keep their host address (runtime.SKIP). Other bits: against the
reference the same (prompt2 steps KL 4.0e-6 against 3.8e-6, top tokens 17 of 17; prompt2's prompt
2.3e-3 against 1.7e-4, prompt1 and prompt3 within their noise); against the bits of before KL mean
1e-4 (prompt2) to 1e-2 (a prompt of 1024: the routes of a few tokens change). The rows of verify
groups of 2 to 8 against plain steps: the same bits. A step 37.1 -> 36.6 ms; prompt3 3.80 -> 3.72 s
(warm), the prompt of 5018 tokens 12.93 -> 12.48 s; verify groups within their noise; DSpark the
same tokens (1.13x, 1.01x, 1.28x). RECLANK_DS_HC_STEP=0: the records of before (2: the CPU's sums
with the three records, a check).
