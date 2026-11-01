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

## 4. Phases

| Phase | Work | Check |
|---|---|---|
| 1 (done) | Residency (3.1): the registry, make_room, pins, the OOM retry, `--gpu-budget`; speech and music evictable; the ad-hoc releases removed | Each model's check script with a forced eviction before every call (`RECLANK_EVICT_ALWAYS=1`): the same bits as without. A server session of image -> speech -> music -> video -> image -> speech, each output the same bits as a fresh server's, the GPU never over budget (nvidia-smi sampled). Unit tests for the LRU order and pins |
| 2 | Staged upload (3.2); prefetch | Wan's reload 3.6 s -> under 2 s; the same bits; prefetch not slowing the running request by more than the noise |
| 3 | Lazy load, `--host-budget`, the converted-weights cache (3.3) | A cold model from the cache: the same F20/BF12 bytes as a conversion (byte compare); load times; RSS within the budget with every model registered |
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
  location: off unless given.
- Phase 4's cross-server release: whether the generation server may ask the
  embedding server to release on its own, or only an operator.
