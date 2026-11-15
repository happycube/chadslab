# GPU weight swapping between models, in one server

Goal: a server that hosts several models switches the GPU between them by
evicting and reloading weights, with no restart and no reload from disk, and a
model's outputs the same bits before and after a swap.

## 1. What there is now (server/generate.py)

- Every GPU part of every generation model runs through encoders.runtime._Runner:
  - FLUX.2's text encoder, DiT and VAE (the VAE's encoder of reference images: vae_enc);
  - SD's CLIP, UNet and VAE;
  - Qwen-Image;
  - Wan's DiT and VAE (umT5 is CPU only);
  - Higgs (qwen3.decode, its codec);
  - ACE-Step.

  _Runner.release(keep_host=True) frees the device copies of the weights (the
  Mirror's Buffers) and every program's scratch, and keeps the host programs.
  The next run uploads the weights again from the host arrays.
- The swap mechanism therefore exists; the policy around it is ad hoc:
  - An image request releases the previous image model, unless --keep.
  - A video job releases the last image model, and Wan releases itself after
    each video.
  - **Speech (Higgs) and music (ACE-Step) never release.** Once used, they stay
    on the GPU beside whatever comes next.
  - Nothing releases speech or music for an image model or a video. Higgs at
    about 9 GB, plus an image model, plus Wan's 12.5 GB DiT can run out of
    memory.
- Every model named on the command line is loaded into host RAM at start:
  - read from disk;
  - converted (F20/BF12);
  - its programs built.

  The memory each model holds is resident for the life of the process. On
  this machine that is fine (384 GB); on a 32-64 GB machine it decides which
  combinations can start at all.

## 2. Measured on the 3090 (PCIe 3 x16, the GPU on NUMA node 0)

| | |
|---|---|
| Wan DiT (12.48 GB F20) from disk at start | 34 s (reading 19.7 GB float32 plus the F20 conversion) |
| A Wan DiT call, warm | 2.0-2.1 s |
| release(keep_host=True) | 0.33 s |
| The first call after a release | 5.05 s, plus the text prefix 0.64 s: about **3.6 s of reload**, about 3.5 GB/s |
| Host to device, pageable memory | 4.5 GB/s |
| Host to device, cudaHostRegister'd | 7.5 GB/s (registering 0.034 s a GB) |
| Host to device, cudaMallocHost | 8.2 GB/s (allocating 0.63 s a GB) |
| cudaMalloc + cudaFree of 40 MB | 1 ms |

A swap from host RAM therefore costs seconds, not the tens of seconds of a
load from disk. A pinned or staged upload would bring Wan's 3.6 s to about
1.6-1.8 s. Against a 4-minute video or a 5-second image, the bandwidth is
the smaller win; the policy (what is evicted, and when) is the larger.

## 3. Design

### 3.1 One residency manager in the process (encoders/runtime.py)

- `Residency`: a process-wide registry of every GPU _Runner, each with:
  - a label (model id and part);
  - its resident bytes (the Mirror's Buffers plus its programs' scratch, summed
    as they are made, not guessed);
  - its last use;
  - a pin count.
- `_Runner.get(n, gpu)`, before it builds a GPUProgram or lets the Mirror
  upload, calls `Residency.make_room(self, need)`:
  - `need`: the bytes of the host arrays this program reads that are not on
    the device yet, plus the scratch of its last build at this size (the first
    time, an estimate from the program's buffers);
  - make_room evicts unpinned runners, least recently used first, with
    release(keep_host=True), until cudaMemGetInfo's free memory covers
    `need` plus a reserve (default 512 MB).
- If a cudaMalloc still fails during a build or an upload: evict every
  unpinned runner, then retry once; after that, the error goes back to the
  request as now.
- **Pins:** `with Residency.use(*parts):` around a pipeline's call keeps its
  own parts from being evicted by each other mid-request:
  - FLUX's text encoder is not evicted for its own DiT;
  - Wan's DiT is still released before its VAE, as now: a part may unpin
    itself early.

  A request runs under server.lock, so the pinned set is that one request's.
- **Budget:**
  - `--gpu-budget GB` (default: the free memory at start, less the reserve);
  - `--keep` becomes "pin these models' weights" (a list of ids), not a
    global flag.
- The ad-hoc release calls in server/generate.py (last_image, Wan's release
  after each video) go away. The manager decides; the pipelines say only what
  they use.
- KV caches and saved voices:
  - Higgs's prefix snapshots (save_prefix) are host copies already and survive
    an eviction.
  - Its float32 caches are in the Mirror and are rebuilt (zeros) after one,
    which is correct: every request writes its own rows. Check this with a
    voice-clone request after an eviction.

### 3.2 A faster reload (backends/cuda)

- Mirror.device uploads each array with a pageable cudaMemcpy. Instead,
  reuse the staging of experts.cu (GG_STAGE: two pinned buffers, a memcpy
  into one while the other's DMA runs) for any array over a few MB. That
  gives about 7.5-8 GB/s, with no pinned copy of the model kept in RAM.
- Optional: cudaHostRegister the host weights of models pinned by --keep.
  This costs the same RAM but locks it; it is not the default.
- Prefetch:
  - when a queued request names a model that is not resident and the budget
    has room beside the running one, its upload starts on a second stream
    while the current request computes;
  - when there is no room, nothing is prefetched (no extra eviction).

**Done (the staged upload):** `Buffer.upload` of 8 MB or more goes through
`rc_gpu_h2d_big` (runtime/experts.cu): the caller and a helper thread (on the
copy workers' CPU: the model thread is pinned to its own, and a helper sharing
it gave 5.15 GB/s) memcpy 4 MB chunks into a ring of 8 pinned buffers, each
chunk's DMA queued in order on the caller's stream, the next upload going on in
the ring where the last stopped: 8.1 GB/s for arrays of 8 MB to 1 GB against
4.5 for pageable memory (16 MB chunks reach it only from 32 MB arrays; the ring
starting again at its first buffer at each upload serialized small arrays). The
generation checks with every runner evicted and the context reset before each
run (RECLANK_EVICT_ALWAYS=2) print the same numbers; the loads: Wan's DiT
11.69 GB 2.66 -> 1.55 s, its 720p VAE 11.57 GB 4.32 -> 2.37 s, Qwen-Image's
DiT 13.79 GB 3.73 -> 2.05 s, Z-Image's 14.82 GB 3.30 -> 1.89 s, FLUX.2's 7.81
GB 2.09 -> 1.16 s, Higgs's LM 8.40 GB 2.30 -> 1.50 s, SD's UNet 4.49 GB 1.26 ->
0.91 s; an LLM server's reload too (DiffusionGemma's 23.5 GB 5.43 -> 3.37 s,
the same chat). The prefetch is not built: it pays only when the queued request's model
fits beside the running one, which on the 24 GB card these models seldom do.

### 3.3 Host RAM as a tier too (lazy loading)

- Models named on the command line are registered, not loaded: the first
  request that needs one loads it, as Wan already does with umT5.
- `--host-budget GB`: over it, the least recently used model's host programs
  and weights are dropped as well. Its next use loads from disk again.
- **A converted-weights cache**, so that a load from disk does not repeat the
  conversion:
  - the F20/BF12/bf16 rows of a model written once to a cache directory,
    keyed by the checkpoint's file sizes and mtimes plus the conversion's
    version;
  - mapped read-only on the next load;
  - after a load, posix_fadvise DONTNEED on the checkpoint's files, which
    keeps a load from filling a node's page cache (see the page-cache
    stalls).

  Wan's load would go from 34 s to about the time to read 12.5 GB. It needs
  disk space equal to the converted weights and is optional
  (`--convert-cache DIR`).

### 3.4 Models added and removed at run time (no restart)

- `POST /v1/admin/models {"id", "kind": "flux2|sd|qwen-image|wan|higgs|acestep", "path", options}`
  registers a model; `DELETE /v1/admin/models/{id}` drops it from all tiers.
- Both require the API key.
- `/v1/models` gains each model's residency (`gpu`, `host`, `disk`) and its
  bytes.
- `POST /v1/admin/release` evicts everything unpinned now. Phase 4 uses it.

### 3.5 Sharing the GPU with the other servers (later)

The embedding server (8082) and the LLM servers (8081) are other processes
holding GPU memory. Wan at 720p needs the whole card and cannot run beside the
embedding server today.

- First step: each server gets `POST /v1/admin/release` and a `--gpu-budget`.
  A small arbiter script, or the generation server itself before a job that
  needs more than is free, asks the others to release, and they reload on
  their next request.
- The LLM servers' own weight handling (the weightkeeper, KQ, EXL3) is
  another design and not part of this plan. Switching which LLM is served
  without a restart is a separate question.

### 3.6 Phase 1.5: every server swaps (the LLM servers, embeddings, DeepSeek too)

The generation server swaps its own parts; the other servers hold the GPU from
start to exit. Phase 1.5 makes each of them able to give back all its device
memory, its CUDA context included, and take it back on its next request, and
makes the servers ask each other for room.

**Measured (3090, the library loaded, nothing else):** a bare context is 258 MiB
by nvidia-smi (284 with CUDA_MODULE_LOADING=EAGER); a server's is more: the
driver's local-memory reserve grows to the largest stack frame of the kernels
run, and the C side keeps buffers of its own (the h2d_rows stage, the copy
worker's tables). What the servers hold:

| Server | Held besides the context | Its device-only state |
|---|---|---|
| DiffusionGemma (8081) | 22 GB (experts by count, a cache of 32K) | the K/V rows in use |
| Gemma 4 26B (8081) | weights, hot experts, KV (11.6 KB a token) | KV rows, the hot set |
| Qwen3.8 / GLM / LFM (8081) | weights, pool, KV + recurrent state, lanes | KV/state rows of each lane, GLM's DSpark rings |
| Nemotron Super/Bolt (8081) | 7.2 GB dense, ~12 GB pool, KV 9 KB a token | KV/SSM rows, pool counts, routes, LiLiCorr's rings |
| DeepSeek V4.1 (8081) | 8.6 GB dense (placeholders: the FP8 codes no host copy, the BF12/RQ8 forms a 1.2 GB stash), 7.5-9.8 GB hot, 1.9 GB state | the dense weights, each lane's state |
| Embeddings (8082) | _Runners (RESIDENCY counts them, off) | none |

**A release undoes the device half of a load.** Each engine is split in two:
the host half (weights in RAM or the weightkeeper's segments, the host
programs, every host array the programs read) stays; the device half (the
mirror, every GPUProgram and its graphs, the pools, the hot slots, the device
caches, the head and stage buffers) is made by one `_device_init()` that the
load calls and a reload calls again. A release frees it all and the context; a
reload is the load's device half with no disk read. Releasing the live objects
in place was surveyed engine by engine and has traps that give wrong numbers
with no error:

- a new GPUProgram's seq starts at 0 while the CPU-task flags (pinned host
  memory) keep their last values, so the GPU's waits (`*flag < v`) pass early;
- the fixed hot stores' host copies hold the first hot set, the device slots
  the current one: a re-upload makes the slot tables lie;
- device addresses kept in host tables (`dslots`, `_commit_args`, `zc_tables`,
  `ko_table`'s module cache, `stage_tab`, `self.last`, GLM's `commit_g`);
- DeepSeek's dense weights are untouched placeholder maps: a re-upload uploads
  zeros;
- idle hooks (`refresh_experts`, `route_counts`, `observe`) that touch the
  mirror take memory back while evicted.

The device half made as at load avoids them by construction; only the host
state carries over, and the check is phase 1's: the same bits as a fresh
server. The rules of a reload:

- **The plan of the load.** The memory-based choices (the hot set's size, the
  pool target, `stage_cap`, `mix_short`, the groups, `head_cpu`, the warm pool)
  change which experts run where, and so the numbers. The load records them; a
  reload uses the record and asks for the same bytes (it waits or fails; it
  never makes a smaller plan).
- **The hot experts of the load** (the first placement, by counts), not the
  set of the moment before: a fresh server's bits. The decode steps and verify
  groups (the "rows" form) do not depend on the hot set; the mixed prompt groups
  do.
- **The cache stays:** before the release, the rows in use [0, n) of each lane
  (KV, recurrent and SSM state, MTP rows, the DSpark and LiLiCorr rings, the
  pool counts, routes) go to their host arrays; the reload uploads [0, n) only.
  A chat's next turn reads only its new part, as before the release.
- **The flags start again:** the CPU-task flag arrays are zeroed with the new
  programs (a fresh server's seq).
- **Evicted means no device calls:** the engine's entry points (generate,
  forward) reload first; the idle hooks skip.
- **DeepSeek's dense weights** are made again from the weightkeeper's segments
  (the FP8 codes as they are) and a host stash of their converted forms
  (BF12/RQ8: 1.2 GB, the head 1.0 of it, kept from the load: the conversion
  took 5.8 s of a reload). DSpark's experts stay in RAM (a reload made them
  again: 7 s); the first hot experts copy straight from the segments.

**The context.** With nothing left, `rc_gpu_shutdown()` in C: sync, stop the
copy worker and the helper threads, free the C side's statics (stages, scratch
pools, streams, events), `cudaDeviceReset`, and the context's epoch + 1. The
next CUDA call makes a new context; the 14 `static int attr` guards of
`cudaFuncSetAttribute` become epoch checks, and the streams, events and
workers start lazily as at the first call. Host memory that outlives a context:

- `gpumm.pinned()` (cudaMallocHost: a reset frees it under live NumPy arrays)
  becomes plain host memory (mmap) + cudaHostRegister, kept in a registry and
  registered again after a reset. Measured against cudaMallocHost, the same
  copies at every size (16 KB to 1 GB: 8.23 GB/s to the GPU, 8.45 back, within
  1%), and pinned() of 1 GB in 0.48 s against 0.78. The servers on the new
  library against the old (a real prompt, greedy, 5 requests, old/new/old):
  DiffusionGemma each request's time the same (6.1, 4.4, 5.4, 6.1, 4.2 s; the
  same passes and text), LFM2.5 rq4all (every expert on the GPU: the launches)
  142.4 tok/s against 142.5 and 143.6, Gemma 4 26B (6 GB of hot experts, the
  rest on the CPU) 103.9 against 104.5 and 104.2;
- the weightkeeper's registered segments are registered again (Qwen, Nemotron,
  DeepSeek: about 2 s per 70 GB, DeepSeek's 290 GB about 8 s); a server may keep
  its context instead (`--keep-context`) when that costs more than the
  context's memory is worth;
- the C side's cudaMallocHost buffers (the program envs, the stages) go with
  their programs and are made again by the rebuild.

The second GPU's expert device (devices.py, expert_dev.cu) has no teardown and
is not part of this: a server with one keeps its context.

**The servers ask each other** (in place of 3.5's first step):

- Each GPU server writes `work/gpu/peers/<port>.json` (pid, url, its device
  bytes, its last use, busy) and keeps it current; it is removed at exit, and a
  file whose pid is gone is ignored.
- `POST /v1/admin/release` (the API key): an idle server releases everything
  unpinned (an LLM engine: all of it, the context too) and answers the bytes
  freed; a server in a request answers `busy` at once (no wait inside a
  request). `GET /v1/admin/gpu`: its residency (bytes, evicted, last use).
- Residency.make_room, when its own victims are not enough, asks the peers that
  are not busy, the least recently used first, until the free memory covers the
  need; busy ones are asked again every second up to `--gpu-wait` (default
  600 s), then the request fails with the out-of-memory error as now. An LLM
  engine's reload goes through the same make_room.
- Against thrashing: a server is not asked within `--gpu-hold` seconds
  (default 30) of its last request unless nothing else is left; two servers
  short at once take `work/gpu/arbiter.lock` (flock) around asking and loading,
  so one finishes its load first.
- With peers, every server's `--gpu-budget` defaults to the card less the
  reserve.

**Locked models** (never freed for another's room):

- `--gpu-lock` on any server locks all its models; `--gpu-lock ID,...` on the
  generation and embedding servers locks those models (all their parts: phase
  1's `--pin`, which stays as its other name). A locked model is a pin of
  RESIDENCY for good: never a victim of make_room, of a peer's release, nor of
  RECLANK_SWAP_ALWAYS; an LLM engine locked never releases its device half or
  its context.
- `POST /v1/admin/lock {"model": ID, "on": true|false}` (the API key; no model:
  all of the server's) locks or unlocks while the server runs; unlocking a model
  makes it a victim again, it frees nothing by itself.
- `POST /v1/admin/release` releases only what is not locked and answers what
  stayed (`"locked": [ids]`); a server whose models are all locked answers
  `locked` with 0 bytes, and the asker goes on to the next peer. The peer file
  carries the locked bytes, so an asker knows what it can never get: when the
  card less the locked bytes of all servers cannot hold its need, it fails at
  once (the out-of-memory error names the locks) and does not wait.
- `GET /v1/admin/gpu` and `/v1/models` show each model's lock.

**The order of the work, and its checks:**

1. The C side (shutdown, epoch, registered `pinned()`) and Residency's
   `release_all(context=True)`. Check: the generation server's check scripts
   with RECLANK_EVICT_ALWAYS=1 and a context reset before every call: the same
   numbers. **Done:** RECLANK_EVICT_ALWAYS=2 (every runner evicted and the
   context reset before each GPU run, 2 to 45 resets a script, each 0.07-0.09 s,
   none refused): SD, FLUX.2, Qwen-Image, Z-Image, Wan (dit, vae, a second
   prompt, all), Higgs and a cloned voice, ACE-Step, Whisper (and its decoder
   and the long form) print the same numbers as without.
2. The peers, the locks and the admin routes (the generation, embedding and
   LLM servers). Check: a generation job with the embedding server holding the
   GPU: the embedding server releases, the job runs, the embedding server serves
   again with the same vectors (bytes); with the embedding server locked, it
   keeps its memory and the job fits in the rest or fails at once naming the
   lock; unlocked by the route, the next job gets its room. **Done:** the
   embedding server (EmbeddingGemma 2, its three towers on the GPU) and the
   generation server (Wan 2.2 5B), a 720p video (its VAE 22.3 GB): the
   embedding server released 1.91 GB and its context in 0.20 s, the video the
   bytes of a fresh server's; the 18 inputs' vectors (texts, audio, images)
   the same bytes after; locked by the route it kept its 1.59 GB and the video
   fit in the rest; unlocked, the next video's room came from it again, the same
   bytes. The asking is best effort: a part's need may be an estimate (the 720p
   VAE's 24.6 GB, 22.3 real), so with nothing more to give the build tries and
   only the locks (on the known part of the need) fail at once; a build that
   fails after asking names the peers' locks.
3. The engines, each with `RECLANK_SWAP_ALWAYS=1` (a release and a context
   reset before every request) against without (greedy or a fixed seed) and
   against a fresh server: DiffusionGemma, Gemma 4 (26B, E4B), Qwen3.8 (and
   GLM-5.3, LFM), Nemotron (Super, Bolt, MTP on), DeepSeek V4.1 (lanes 2,
   DSpark greedy). Each reload's time printed; a chat's second turn after a
   release reading only its new part. **So far** (5 greedy requests and a chat
   of two turns, RECLANK_SWAP_ALWAYS=1 against without): DiffusionGemma the
   same texts and passes (a release 0.56 s, a reload of 23.5 GB 5.4 s); LFM2.5
   rq4all (every expert on the GPU) the same texts (0.83 s, 24.4 GB in 4.2 s);
   Gemma 4 26B (6 GB of hot experts) each answer the bytes of a fresh server's
   for its prompt (0.3-0.4 s, 8.0 GB in 7.4 s; without the swap 2 of 7 answers
   part at a near tie: the server's hot experts follow the earlier requests,
   and a reload starts from the load's); Nemotron Bolt the same texts (0.97 s,
   23.1 GB in 19 s: the pool's experts and the zero-copy tables made again). The
   chats' second turns go on from the context in host memory (LFM: 25 of 130
   tokens; Bolt 98 of 127; DiffusionGemma: 25). A GPU program dropped unclosed
   stays loaded on the C side (it has no finalizer): `runtime._HANDLES` keeps
   every loaded handle and close_all unloads those whose objects are gone
   (NemotronGPU dropped one at each load: the context then stayed). DeepSeek
   V4.1 (lanes 2, DSpark, 3 requests and the chat) the same texts; a release
   6 s, a reload 47 s (the GPU half 35) against 508 s for a fresh server; with
   the stash of the converted dense weights, DSpark's experts kept and the hot
   experts copied straight from the segments 22.4-23.1 s (the GPU half 11: 7 of
   it the host memory registered again in the new context; the warm prompt
   5.5; lane B 5.9), the same texts as without the swap. A reload's lanes: `models.base.reload_lanes` runs the warm-up with
   the contexts in host memory set aside (they gave the warm prompts other
   starts than the load's) and keeps lane B's engine object (a request is routed
   to a lane before its job reloads: a new object left it on the old state).
   Nemotron Super (MTP, images; a release 1.9 s, 22.4 GB in 6.2 s) and
   Qwen3.8-Flash-Next (MTP; 5.3 s, 23.8 GB in 18-21 s): the answers part from
   the server's without the swap at near ties (the expert pool and HotCache
   follow the earlier requests there); against a fresh server for its first
   request, Super's bench request and chat the same bytes, Flash-Next's bench
   request the same and its chat's first turn the same too when the chat is
   the server's first request (two such swap servers alike, and with the mixed
   groups' calibration fixed the swap the same as without); after four
   requests in the same server it parted at a tie, the history dependence the
   servers without the swap show too.
   The chats' second turns go on from host memory (Super 64 of 93 tokens,
   Flash-Next 219 of 250). Gemma 4 E4B with its drafter on the GPU (MTP):
   the same texts and chat (a release 0.19 s, 3.28 GB in 0.9 s; the second
   turn 129 of 155 tokens from host memory); the drafter's own device array
   (the draft chain's tokens) is forgotten with its program
   (`GPUDrafter.release`: it was kept, and after a reload every request but
   the first drafted into the old context's address and answered nothing).
   GLM-5.3 has no engine in the server yet (the model and its checks only):
   its release and reload come with its engine.
4. A session across the servers: an LLM chat, an image, an embedding, music, a
   video, the LLM chat again (its cache kept), each output the same as alone;
   the card never over (nvidia-smi sampled). **Done:** the embedding server
   (EmbeddingGemma 2 on the GPU), the generation server (SD 1.5, ACE-Step,
   Wan 2.2 5B) and DiffusionGemma (23.5 GB) on one 3090: a chat, an image, the
   18 embeddings, music, a video, the chat's second turn. Each output the
   bytes of its server alone (the chat's answers too; its second turn 25
   tokens from the cache after DiffusionGemma gave its 23.55 GB to SD's UNet in
   0.57 s and came back in 6.2 s); the card at most 23187 of 24576 MiB. The
   embedding server asks its peers to release before its towers' warm-up (as an
   LLM server before its load).

**Testing note: the large models.** A model the size of Qwen3.8-Flash-Next or
larger (by its weights: Flash-Next, GLM-5.3-Flash, Nemotron Super, DeepSeek
V4.1) counts as locked (`--gpu-lock`, as if set for it) in the session across
the servers: it keeps the card and is never asked to release, and the session
of check 4 runs with the smaller LLMs (DiffusionGemma, Gemma 4, Nemotron Bolt,
LFM). Each large model is tested alone: check 3 (its release, the context
reset and the reload before every request, against without and against a fresh
server), its reload's time, and a peer's release request answered `locked`
with its memory kept. **The lock, checked** on an LLM server (DiffusionGemma
with `--gpu-lock`): a release request answered `locked` (nothing freed, the
card at 22604 MiB); unlocked by `/v1/admin/lock`, a release freed its 23.55 GB
and the context (136 MiB left); the chat brought it back (5.4 s) with the same
answers as without; locked again, `locked` again.

**As built.** `server/gpushare.py`: the peer file (RECLANK_GPU_DIR, default
~/.cache/reclank/gpu), the routes (http.py `_admin`), `--gpu-peers`,
`--gpu-wait`, `--gpu-hold`, `--gpu-lock` (the generation server's `--pin` its
other name); `RunnerShare` for the generation and embedding servers (RESIDENCY
on in the embedding server too, its towers' runners; Residency.make_room asks
`RESIDENCY.peers` when its own victims are not enough, the arbiter held until
the load ends), `EngineShare` for an LLM server (its model thread: a release
runs there between jobs, and `ModelWorker.before` brings the device half back
at the start of the next job, room asked first; the peers asked to release
before the load, whose plan takes the free memory). An engine gives
`release_device` (its device-only state to host memory: each lane's context to
the RowCache's host memory, DiffusionGemma's cache rows [0, n)) and
`reload_device` (the load's device half again: Gemma 4 `gpu.offload` with the
load's free memory; Qwen and DeepSeek `runtime.remake`, a new object of the
host half's attributes and `_device_init`; Nemotron a new NemotronGPU of the
load's free memory, link rate and registrations; then the load's warm-up, lane
B, the hot experts grown to the load's count). Between them
`runtime.close_all` frees what is left (every GPU program, runner and block)
and the context goes (`gpumm.on_context_release` hooks drop module caches of
device memory: Gemma's rope table). An expert device on a second GPU, or the
Qwen image encoder's held room (`--mmproj-gpu reserve|lend`), keeps the engine
locked.

### 3.7 One weightkeeper for every model, shared by every server

Today a weightkeeper holds one checkpoint (runtime/weightkeeper.py: a plugin
each for DeepSeek V4.1, Qwen3.8's GGUF, EXL3 and FP4-Local, Nemotron, Wan, a
Gemma 4 GGUF); several run side by side (`--name`), each checks the memory on
its own, and a server finds the one holding its checkpoint (find). The
generation and embedding servers keep their models' weights in their own
memory. With one keeper:

- **One process, a catalog.** The keeper (the default socket) holds entries: an
  entry is today's keeper state (the index, the segments, the plugin), keyed by
  the checkpoint's identity (its files' sizes and header hashes, the plugin, its
  options: the forms such as `--experts-type`). The requests:
  - `list`; `status ID`;
  - `load CKPT {options}`: the entry is read in the background (the
    segment's pages, the plugin's fill), `status` shows how far it got;
  - `get ID|CKPT`: the index and the memfds, as now;
  - `drop ID`: the keeper closes its memfds (a server that maps them keeps
    its pages until it unmaps them: the kernel counts the maps);
  - `lease`: a server's `get` keeps its connection open as a lease; the
    keeper knows who maps what, and a lease ends with its process.
  Named keepers stay for isolation (a test, a second user).
- **Every model a plugin.** The existing ones, plus the generation and
  embedding models over about 2 GB as their converted rows (F20, BF12: the
  forms their servers make now), so a load is a map, not a read and a
  conversion: FLUX.2, Qwen-Image (and its text encoder), Z-Image, SD, Higgs,
  ACE-Step, Whisper, SeedVR2, umT5, EmbeddingGemma 2 and the music towers,
  DiffusionGemma's BF12 GGUF. The small ones (RIFE, Real-ESRGAN, the depth
  models) stay in their servers.
- **The host budget lives in the keeper** (phase 3 becomes this):
  `--host-budget GB` (and per node, the GPU's node first) for all entries; for a
  new load the least recently leased entries with no lease go first (dropped:
  their next use reads the checkpoint again); `--keep ID,...` keeps entries for
  good (the host side of 3.6's locks). check_memory counts all entries at once,
  not keeper by keeper.
- **The converted-weights cache** is the keeper's segments written to disk
  (`--save DIR`: each entry's segments and index, keyed as above); a load finds
  its entry there first and reads it with O_DIRECT (no conversion, no page
  cache), then the checkpoint. Wan's 866 s ready would be about the time to
  read 72.9 GB.
- **Shared between servers.** Two servers that use one model (a text encoder
  in two pipelines; the same LLM in two servers) map the same entry: one copy
  of its pages. Each process registers its own map with CUDA
  (cudaHostRegister counts per process; the pages are pinned once).
- **The servers' host half is the keeper's.** A server keeps only what its
  programs write (activations, caches, its host programs) and maps the rest:
  - a GPU reload (3.6) uploads from registered maps (DMA, 11-12 GB/s against
    pageable 4.5), so 3.6's reloads go faster (part of phase 2's goal);
  - a third tier past 3.6's: an idle server may exit (`--idle-exit S`; the
    context and all its own memory gone) and start again on its next request,
    attaching in seconds (Wan: 5-19 s against 446 s from the shards);
    `scripts/serve_forever.sh` and the peer files (3.6) restart it, a request
    in the meantime waits.
- **Faults.** The keeper holds no CUDA context and runs no model. A keeper
  that exits takes nothing from the running servers (their maps stay); a new
  keeper loads its entries again (from `--save` first) and serves new leases.
  As root it keeps today's rules (the hugetlb pools, `--lock`, the settings
  given back by the last root keeper).
- **The tools.** `scripts/weightkeeper.py --add CKPT [options] | --drop ID |
  --list | --status ID`; the server scripts attach from the shared keeper by
  checkpoint (find, extended to entries); start_server.sh starts the keeper
  when none answers.

**The steps** (each with its checks, in this order: each one needs the one
before):

1. **The catalog.** Today's per-checkpoint state (`Weightkeeper`) becomes an
   `Entry`; the keeper holds entries by ID and serves `list`, `load`, `get
   ID|CKPT`, `drop`, with today's `get` and `status` (one entry) as they are,
   so the servers attach unchanged; `find` asks `list`. A keeper started with a
   checkpoint holds it as its first entry (as root too: later loads run as the
   user, with the pools as they are). `scripts/weightkeeper.py --add / --drop /
   --list / --status ID`, and `--serve` for an empty catalog. **Done:** the
   keeper's tests (one keeper, two entries added in the background, each found
   by its checkpoint and checked against its shards, the same load again its
   entry, a drop: the round's map reads the same bytes); on real models one
   keeper (`--serve`) with the Gemma 4 26B and E4B GGUFs added, each server
   attached to its entry: the bench's answers and the chat the bytes of the
   servers without a keeper. Found there: the Gemma GGUF plugin splits the
   experts (and the E4B's per-layer token table) between the nodes, which the
   GPU loader cannot read (it takes a stack whole); with `--nodes 0` it holds
   them on one node (the per-layer table read whole now, as the experts were).
2. **Leases and the host budget.** A server's `get` keeps its connection as the
   lease; `--host-budget GB` and `--keep ID,...`; an entry with no lease goes
   first, least recently leased first.
   **Done:** a round's get keeps its socket open for the process's life (the
   maps are never unmapped; `Weights.end_lease` for a test), the keeper
   watching the leases' sockets (the peer's pid; the lease's end its EOF).
   With `--host-budget GB`, a load over it, or one the memory cannot take (on
   a node short of it, the entries with pages there), drops the entries no
   round leases and none keeps, the least recently leased first; with all of
   them leased or kept the load fails, saying which. Without a budget nothing
   is dropped but by `drop`, as in step 1 (a keeper's entries do not go
   because someone added another). `--keep` alone keeps the entry the
   command loads; `--keep ID,...` and `--unkeep ID` the keeper's (at its
   start, or of the running keeper: a checkpoint not loaded yet kept when it
   comes). `--list` shows each entry's leases (or how long it has been idle),
   kept, and the budget. The budget is the host's in all, not per node: a
   node short of memory is the memory check's. Checks: the keeper's tests
   (three entries, a budget for two: the least recently leased dropped, a
   leased and a kept one not, a lease of another process ending at its exit);
   on real models a 14 GB budget with the Gemma 4 26B (12.8 GB) and E4B (1.6
   GB): the 26B's server leasing its entry, the E4B's load failed; the server
   stopped, the E4B's load dropped the 26B (9.5 GB more available) and its
   server attached. The bench's answers the bytes of the servers without a
   keeper, and the E4B's chat; the 26B's chat that of the swap check above
   (the near tie of its hot experts).
3. **`--save DIR`** (off unless given): the entries' segments and index written
   once, a load reading them back with O_DIRECT before the checkpoint.
   **Done:** each entry's segments (the bytes in use) and its index written to
   `DIR/<ID>-<code key>/` once, after its fill (in a directory of its own,
   renamed when complete; skipped, said, when the disk lacks the room and 4
   GB); a load finds it first and reads it with O_DIRECT, no conversion. The
   key: the entry's ID (the checkpoint's identity, the options) and a hash
   of the code that makes the bytes (the plugin's module, or DeepSeek's
   fill; the CPU library's name, itself a hash of every C source and the
   compiler; VERSION); the layout made again must be the saved one. So a
   change of a kernel makes every entry convert again once (the old copies
   stay, named in the log, until someone removes them). An entry's status
   says where it came from (`source`: checkpoint or saved). Checks: the
   keeper's tests (an entry with its dense segment saved, read back by a
   second keeper and checked against its shards; another checkpoint and
   other options their own entries; a copy of other code not read; a Gemma
   GGUF's KQ_Q4X stacks and dense tensors, the index the same). On real
   models, the Gemma 4 26B and E4B on the NVMe: the 26B ready in 9.8 s
   against 111.4 s (of which 94.6 s the KQ_Q4X packing of one thread), the
   E4B 1.3 against 9.1 s, the reads 2.25 GB/s; the 26B's saved 12.8 GB the
   bytes of a fresh conversion in full, both entries checked against their
   files (`--check`, which now packs the file's Q4_0 matrices that the
   keeper holds as KQ_Q4X), and both servers' bench answers and chats the
   bytes of the servers without a keeper.
4. **The generation and embedding models as plugins**, their servers mapping
   their converted rows from the keeper (a load a map); their reloads from the
   registered maps (DMA).
   **Done, in another shape than plugins:** a model's own loader makes its
   arrays and linears (runtime/kept.py: `Kept(label, checkpoint, options,
   code)`, `k.linear(name, make)`, `k.array(name, make)`, `k.done()`); the
   first load copies each into one memfd of its process (the model holding
   views of it: one copy) and puts it in the keeper ("put": the memfd sealed
   for size, its index; the connection the lease), a later load maps the
   entry and calls no make. The keeper runs no model code and the bytes are
   those of the model's one loader by construction; a second loader for
   each model (a plugin) would have to make the same bytes on its own. With
   `--save` the entry is written as any other and a new keeper reads it back
   at the first `restore ID`. The ID: the label and a hash of the checkpoint's
   files (names, sizes, mtimes), the options that shape the arrays (the GPU's
   forms or the CPU's), and the code (the model's package, the encoders'
   runtime, ops, kept.py, the CPU library's hash, VERSION). **Opt-in**
   (`RECLANK_KEPT=1`, the default socket or `RECLANK_WEIGHTKEEPER=NAME`): a
   put entry stays in the keeper after its server exits, until a drop or the
   budget takes it, so a keeper does not grow by a server's models unless
   asked. Wired in where a load converts: Z-Image's DiT (F20), Wan's DiT
   (F20; not with the A14B's keeper plugin nor the streamed blocks, which
   `_pack` copies), SeedVR2's DiT (F20; not the 7B's streamed blocks),
   Qwen-Image's text encoder (BF12). Each model's check (its regression
   check's arguments) without a keeper, then with one: made and put, mapped,
   and after a new keeper restored from `--save`, the numbers of the run
   without it in every case:

   | Model (part)            | Held    | Base    | Mapped | Restored |
   |-------------------------|---------|---------|--------|----------|
   | Z-Image DiT             | 15.8 GB | 47.3 s  | 0.5 s  | 8.5 s    |
   | Wan 2.2 TI2V-5B DiT     | 12.8 GB | 38.7 s  | 1.5 s  | 7.9 s    |
   | SeedVR2-3B DiT          | 8.7 GB  | 76.2 s  | 0.0 s  | 4.7 s    |
   | Qwen-Image text encoder | 10.6 GB | 12.7 s  | 0.4 s  | 5.8 s    |

   Not wired, their loads short or of another shape: SD's UNet (its BF12
   made after all its convs, 4 s), Whisper's encoder (7.6 s), EmbeddingGemma
   2 and the music towers (small), FLUX.2, Higgs, ACE-Step and umT5 (no
   conversion: the checkpoint's bytes as they are), DiffusionGemma (a GGUF:
   the keeper's GGUF plugin's shape). The reloads from registered maps (DMA)
   are not done: a GPU reload still uploads from the maps by the staged path
   (3.2).
5. **`--idle-exit S`** for a server whose weights the keeper holds.
   **Done:** `--idle-exit S` on every server (`python -m server` and its
   scripts, `server.embed`, `server.generate`; server/idle.py): no request in
   flight and none for S seconds, and nothing busy (the generation server:
   a job queued or running, which a new start would fail), the server stops
   accepting, serves the requests of its open connections (an idle
   keep-alive connection ends after 5 s of waiting for its next request
   line, its requests not timed), and exits with 75; a request in that drain
   keeps it serving. `scripts/serve_idle.py --port P -- CMD` binds the port
   and keeps the listening socket, running the server with it
   (`RECLANK_LISTEN_FD`): after an idle exit it waits for a connection
   without taking it (it waits in the backlog), starts the server again,
   which takes it; a crash starts it again after a pause (as
   serve_forever.sh), a clean exit or a stop ends both. Checks: a test with
   a small server (three idle exits, each next request answered by a new
   server; a request longer than S not cut; an idle keep-alive connection
   not keeping the server; busy keeping it; the launcher's stop); the Gemma
   4 E4B from a keeper behind the launcher (`--idle-exit 20`): the idle exit
   gave back all of the card, the next request answered in 9.3 s (the start,
   the attach, the GPU's load), the bench's answers and the chat before and
   after the bytes of the server without a keeper; the generation server
   with Z-Image (its DiT kept, `RECLANK_KEPT=1`): an image of 58.9 s at the
   first start, the same request after the idle exit 31.6 s (the DiT mapped;
   the text encoder, not kept, and the image the rest), the same bytes.

Checks: each plugin's entry the same bytes as its own keeper's (or its
server's conversion: byte compare) and its model's check the same numbers; two
servers attached to one entry, one copy of its pages (numa_maps, the RSS of
the keeper and of each server); a keeper stopped and started again while two
servers run: both go on, and the new keeper's entries from `--save` the same
bytes; the host budget: an entry dropped by LRU for a load, its next use loading
again with the same bits; an idle server's exit and its restart serving the same
outputs.

## 4. Phases

| Phase | Work | Check |
|---|---|---|
| 1 (done) | Residency (3.1): the registry, make_room, pins, the OOM retry, `--gpu-budget`; speech and music evictable; the ad-hoc releases removed | Each model's check script with a forced eviction before every call (`RECLANK_EVICT_ALWAYS=1`): the same bits as without. A server session of image -> speech -> music -> video -> image -> speech, each output the same bits as a fresh server's, the GPU never over budget (nvidia-smi sampled). Unit tests for the LRU order and pins |
| 1.5 | Every server swaps (3.6): the engines' device halves, the context released, the servers asking each other for room | 3.6's checks: each engine's outputs with a release and a context reset before every request the same as a fresh server's; a session across the servers |
| 2 (the upload done) | Staged upload (3.2); prefetch | Wan's reload 3.6 s -> under 2 s; the same bits; prefetch not slowing the running request by more than the noise |
| 3 (steps 1-5 done) | Lazy load, `--host-budget`, the converted-weights cache (3.3), all in the one shared weightkeeper (3.7) | A cold model from the cache: the same F20/BF12 bytes as a conversion (byte compare); load times; RSS within the budget with every model registered |
| 4 | Admin routes (3.4); release across servers (3.5) | Adding SD 2.0 to a running server and generating; Wan 720p after the embedding server releases, then the embedding server serving again |

Phase 1 is the core and is self-contained: about 1-2 days. Phases 2-4 are
independent of each other.

Phase 1 is done (docs/GENERATION_PLAN.md has its entry and its numbers). As
built it differs from 3.1 in three points:

- The manager is opt-in: the generation server turns it on (and
  RECLANK_RESIDENCY=1, RECLANK_EVICT_ALWAYS=1); elsewhere a _Runner runs as
  before.
- A scope pins each part from its first run in it, not from the scope's start,
  so a part the request has not reached yet can still go.
- The budget refuses only on the known bytes; the first build's estimate of
  its scratch only drives the evictions.

--keep stays a flag (the image models' text encoders and VAEs), and --pin ID,...
pins whole models.

## 5. Decisions for the user

- The default for `--keep`: today it keeps the text encoders and VAEs of image
  models on the GPU. Under the manager, the natural default is "no pins, LRU
  decides", with `--keep ID,...` to pin.
- The converted-weights cache writes model-sized files to disk; its default
  location: off unless given. **Decided:** off; written only where a path is
  given.
- Phase 4's cross-server release: whether the generation server may ask the
  embedding server to release on its own, or only an operator.
