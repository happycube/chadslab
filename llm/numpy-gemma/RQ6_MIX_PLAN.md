# Plan: RQ6_K experts where the UD-Q4_K_XL file shows it is safe, and mixed expert types on the GPU

Status: a plan. No runtime code has changed for it. The
measures below ran on this machine (jackal: Xeon W-2295, 18 cores, one
socket; RTX 5060 Ti 16 GB, 448 GB/s) with scratch scripts (section 7).

## 0. Progress (on the 2-socket Xeon and the 3090)

Steps 1 to 4 for M1 are done (25dc887 and after):

- The experts-only file of M1 (convert_q8_gguf.py --experts mix
  --experts-only): models/Qwen3.8-Flash-Next-RQ6mix/
  Qwen3.8-Flash-Next-experts-RQ6mix.gguf, 108.8 GB; gate and up of the 48
  layers RQ6_K (-35.0 dB), down RQ8_0 (-45.4 dB; 640 inputs). The model
  takes it over the RQ8_0 file (GGUFOverlay: Qwen4CPU experts=,
  NP_GEMMA_EXPERTS, serve_qwen4 --experts). check_q8_gguf.py and
  check_qwen4_gpu.py PASS.
- KT_Q6K: the prompt groups stay on the tensor cores (Q6_K tiles against
  float32 tiles: KL 0.016 on 8K of source, the noise of the products).
- The quality at 64K (a prompt of 65470 tokens of this project, a greedy
  answer of 512 tokens, and the log-probs of the answer of the first run):

      run          same tokens  KL (top 64)  top-1    NLL
      RQ8_0 1      512          -            -        0.3265
      RQ8_0 2      76           0.0015       98.63%   0.3273
      M1 1         22           0.0020       98.44%   0.3274
      M1 2         38           0.0019       98.44%   0.3280

  M1 against RQ8_0 is about the noise of two runs of RQ8_0; no token where
  the reference was sure (p > 0.9) changes.
- The speeds of that day were wrong: numad (a root daemon, on at boot)
  moved each large Python process to node 0 (217 moves that day), so the
  team of the CPU part shared the cores of one node. The rates of M1 in the
  model (8.8 to 12.9 tok/s, against 26 for RQ8_0) are not valid; measure
  again with numad off (systemctl disable --now numad).
- Micro benchmarks (short processes, numad checked; the cold experts of a
  step, 8 of them, node-1 copies; kq_calib_nodes):

      gate/up, down     24 threads   32 threads   per thread (2 threads)
      Q8_0, Q8_0        496 us       446 us       gate/up 5.9 GB/s, down 3.7
      Q6_K, Q8_0        429 us       391 us       gate/up 5.1 GB/s

  The CPU part runs at a rate per thread (the latency of memory), not at
  the bandwidth of the nodes; 32 threads gave 10% more than 24 when the
  machine had no numad move; 40 and more meet the other load of the
  machine. M1 takes 14% less than RQ8_0 (its bytes: 15%).
- Two changes to the CPU part (the same bits): kq_dot4_q8_0 (four Q8_0
  rows of one token in one loop) and 16 rows a down task (NP_GEMMA_MOE_NDN;
  4 before): the down phase 11% shorter on 2 threads; M1 at 32 threads 391
  -> 382 us a call. The bf16 products of a step (60 to 77% of the bandwidth
  of the 3090 for the middle sizes) did not gain from two loads in flight
  once a token alone and a group had to keep the same bits.
- Software prefetch 1024 bytes ahead in the Q6_K and Q8_0 row loops of one
  token (NP_GEMMA_KQ_PF, the same bits): the cold experts of a step on 24
  threads 484 -> 409 us (Q8_0), 408 -> 360 us (M1); on 32 threads 453 ->
  430 and 390 -> 357. With it, 24 threads do about what 32 did without.

## 1. Why

The RQ8_0 file (RQ8_EXPERTS_PLAN.md; all the experts RQ8_0, type 55) is
near the original (-45.5 dB a matrix; KL to it: Q8_0 0.0124, about the noise
of two runs, NVFP4 0.0354), but it is slower than NVFP4: plain decode 29.5
against 42.3 tok/s, MTP 34.5 against 57.6, the 8K prompt 633 against 910.
An RQ8_0 expert is 5.22 MB, an NVFP4 one 2.77 MB.

RQ6_K (Q6_K blocks of the rotated rows) is -35.0 dB a matrix at 6.56 bits:
4.03 MB an expert. This plan uses the per-tensor choices of Unsloth's
UD-Q4_K_XL file as a map of the sensitive tensors: RQ6_K where Unsloth used
5 bits or less, RQ8_0 where Unsloth paid for 8.

## 2. The map of UD-Q4_K_XL

models2/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL (4 shards). The types are per
layer and per matrix, never per expert:

    experts (gate / up / down)   layers
    Q4_K / Q4_K / Q5_1           43 layers: all but 2, 4, 30, 46, 47
    Q5_K / Q5_K / Q8_0           2
    Q4_K / Q4_K / Q8_0           4, 30, 46, 47
    (no MTP layer: a separate file, MTP/mtp-...-shared-Q8_0.gguf, all Q8_0)

    everything else              Q8_0: attn_*, ssm_out, attn_qkv, attn_gate,
                                 the shared experts, hc_*, ple_key/value,
                                 output, token_embd; BF16: the indexer;
                                 F32: norms, routers, small matrices;
                                 IQ4_NL: the n-gram table

- Q5_1 for down is the fallback of Q5_K: down has 640 inputs, not a
  multiple of 256. In the same way, the Q8_0 of down in layers 2, 4, 30, 46,
  47 is most likely the fallback of a Q6_K that Unsloth wanted there (the
  "more bits" layers of llama.cpp: the first, the last, and a few in the
  middle). So Unsloth marks these 5 down matrices, and the gate/up of layer
  2, as the sensitive ones.

The UD tensors against ORIG (scratch ud_vs_orig.py: 3 experts each of
layers 0, 2, 4, 12, 30, 46, 47), with RQ6_K and Q8_0 of the same experts:

    UD type        where                   UD (vs ORIG)       RQ6_K      Q8_0
    Q4_K gate/up   47 layers               -22.3 to -22.9 dB  -35.0 dB   -45 dB
    Q5_K gate/up   layer 2                 -28.5 to -28.7     -35.0
    Q5_1 down      43 layers               -28.7 to -29.4     -35.0
    Q8_0 down      layers 2, 4, 30, 46, 47 -45.1 to -45.4     -35.0      -45.1 to -45.4
    Q8_0 dense     shared gate (L23), attn_v (L23): -44.3, -42.3 (plain RTN Q8_0)

RQ6_K is 6 to 13 dB better than the UD tensors everywhere UD used Q4_K,
Q5_K, or Q5_1. Only the 5 down matrices where UD used Q8_0 are better in UD
than in RQ6_K (10 dB).

## 3. The assignment

    part                                  UD        this plan
    gate, up: all 48 layers               Q4_K/Q5_K RQ6_K (2560 inputs: whole Q6_K blocks)
    down: the 43 layers of Q5_1           Q5_1      RQ6_K with a block of 128 at the end (M2),
                                                    or RQ8_0 at first (M1)
    down: layers 2, 4, 30, 46, 47         Q8_0      RQ8_0
    shared experts                        Q8_0      RQ8_0 (as now)
    MTP experts                           Q8_0      RQ8_0 (as now); RQ6_K is an option
                                                    (-35.0 dB, better than the shipped FP8)
    dense, head, embeddings, indexer      Q8_0/BF16 as now (bf16 in the file)
    n-gram table                          IQ4_NL    as now

- Rotated, not plain Q6_K: on the experts the rotation gains only 0.2 to
  0.3 dB, but the runtime already rotates the x of every MoE and the act of
  every pair when the experts are RQ8_0 (kq_moe_rot, gg_moe_rot, and the
  program's TQ_ROT of the MoE input: a state of the process). With RQ6_K
  every expert matrix of the file is still rotated, so that invariant
  ("all the experts rotated, or none", qwen4.py) holds and no per-layer
  rotation flag is needed. Plain Q6_K next to RQ8_0 would need one.
- Layer 2 gate/up: UD paid 5 bits there (Q5_K, -28.6 dB); RQ6_K is 6.5 dB
  better than that. A cautious form keeps all of layer 2 in RQ8_0 (+0.3
  GB).

The sizes (48 x 512 routed experts):

    form                                            an expert   all       vs RQ8_0
    RQ8_0 (now)                                     5.22 MB     128.3 GB  1.00
    M1: gate/up RQ6_K, down RQ8_0                   4.43 MB     108.8 GB  0.85
    M2: gate/up RQ6_K, down RQ6_K (5 down RQ8_0)    4.03 MB     100.2 GB  0.78
    UD-Q4_K_XL                                      3.07 MB      75.5 GB  0.59
    NVFP4 (KQ_NVX)                                  2.77 MB      71 GB    0.53

M2 fits a full copy on each NUMA node of the target more easily (node 1:
129 GB) than RQ8_0 (131 GB with MTP, which needs the partial copy).

## 4. What the speed may be (microbenchmarks on this machine)

The CPU part (the cold experts of one layer, one token; kq_moe_small_body
through cops.kq_calib_nodes, 18 threads spread, 2 layers of 512 random
experts with sane scales; scratch bench_cpu_types.py):

    gate / up / down      MB an expert  cold  us a call   GB/s   us an expert  vs Q8_0
    Q8_0 / Q8_0 / Q8_0    5.22          5     579         45.1   116           1.00
                                        10    1069        48.9   107           1.00
    Q6_K / Q6_K / Q8_0    4.43          5     526         42.1   105           0.91
                                        10    1041        42.6   104           0.97
    NVX (now NVFP4)       2.77          5     318         43.6   64            0.55
                                        10    627         44.2   63            0.59
    Q4_K / Q4_K / Q5_1    3.07          5     409         37.6   82            0.71
                                        10    830         37.0   83            0.78

- The CPU reads about 42 to 49 GB/s for every type here, so the time
  follows the bytes, but Q6_K costs more work per byte than Q8_0 (42.6
  against 48.9 GB/s). M1 gains only 3 to 9% of the CPU part over RQ8_0. M2
  (down 1.35 MB in place of 1.74) would be about 0.88 to 0.90 of RQ8_0, by
  the same rates. NVFP4 is 0.55 to 0.59.
- So RQ6_K alone does not bring back the rate of NVFP4: at best about 10
  to 12% of the CPU part of a step. The rest of the gain is the GPU: the
  same GB of hot slots hold 18% (M1) or 30% (M2) more experts, so fewer are
  cold. Measure on the target (2 sockets, about 100 GB/s): the ratio of the
  rates of Q6_K and Q8_0 can differ there.
- A faster Q6_K body (the CPU kernel is compute-limited at 42.6 GB/s here)
  is the lever if M2 is chosen: kq_dot_q6k and kq_tile4 for 4 rows; a form
  in groups of 16 rows as KQ_Q8X16 / KQ_Q4X.

The GPU hot experts (10 pairs a launch, random experts of 256 in the slots,
the device code of gpu.cu; scratch bench_gpu_types.cu), GB/s of the weight
bytes:

    matrix            type   MB an expert   warp a row   4 warps a row   8 warps a row   NVX block
    gate (640x2560)   Q8_0   1.74           375          369             343             -
                      Q6_K   1.34           322          323             258             -
                      NVX    0.92           238          -               -               400
                      Q4_K   0.92           312          278             190             -
                      Q5_1   1.23           376          341             255             -
    down (2560x640)   Q8_0   1.74           347          288             146             -
                      NVX    0.92           367          -               -               403
                      Q5_1   1.23           346          192             105             -
                      Q6_K   (no tail: not possible today)

- The generic kernel (kq_row, a warp a row: k_kqh_gu / k_kqh_dn) is
  already at 72 to 84% of the bandwidth for Q8_0 and Q6_K; only NVX needed
  its block kernel (238 -> 400 GB/s). So the "Q8_0 block kernel of the hot
  experts" of RQ8_EXPERTS_PLAN.md is not needed on this GPU (measure on the
  3090 before writing it). A time for a pair: Q8_0 14.3 us, M1 13.4, M2
  about 12.5, NVX 6.9.

## 5. Mixed types on the GPU and the CPU: what exists, what to change

(From a read of the code at 1c67389 plus the uncommitted edits of that
time; file:line of np_gemma/.)

What already works:

- Types per layer: each layer has its own stores (_store, qwen_gpu.py:696-
  712: per-matrix nb = nbytes // E, its own device buffers), its own
  GP_KQ_HOT_MOE / GP_KQ_GROUP_MOE record with its own types, and its own
  CPU kq_moe_mats (6 x (data, type)). The MTP layer (Q8_0 next to NVX
  layers) is the precedent: _more_stores (qwen4_gpu.py:144-152).
- Gate/up type != down type: the records have gtype and dtype; the CPU has
  ngu (from gate) and ndn (from down).
- GP_MOE_PLAN, GP_FETCH, the NUMA copy: per-layer nb, byte copies.

What to change:

1. The warm slots (a bug today, by reading; not run). enable_warm gives row
   r the slots WARM + r k + i at ptrs[part] + (slot - WARM) nb_row
   (gpu.py:2468-2503, kqh_wslot gpu.cu:7211-7229, moe.c:204-205). With rows
   of different nb the rows overlap; the buffer is cap * self.per (the
   largest expert of the main layers, not the MTP layer). The MTP row (Q8_0,
   1.89x an NVX gate) already lands past the buffer when the warm slots are
   on (--mmproj-gpu lend, NP_GEMMA_GPU_WARM=ring). Fix: a base address for
   each row (or one stride, the largest nb) in HotCache, gg_set_warm/
   kqh_wslot, and the warm entries of desc. About 30 to 40 lines. Needed
   before any mix, and now for the MTP layer.
2. The slot count: one n_slots for the model, from self.per (the largest
   expert: qwen_gpu.py:249-267); the fetch staging, the mixed-group ring,
   and the warm buffer are sized by self.per too. Safe, but a Q6_K layer
   then gets the slots of an RQ8_0 layer, and the GB of the budget are not
   all used. Optional: slots per layer from its own nb (about 40 lines;
   fetch staging E - min(n_i)).
3. The cost model of mixed groups: calibrate_mix writes the costs of
   layers[0]'s type for all layers (qwen_gpu.py:1301-1303; per layer, about
   3 lines); _moe_mix reads the gate type only (1039); MIX_CPU (79) has only
   type 53: entries for Q8_0 and Q6_K (tuning).
4. The prompt on the tensor cores: GP_KQ_GROUP_MOE runs k_qmoe_*_tc only if
   both gtype and dtype have a KT format (Q8_R, Q8_0, Q5_1, Q4_K, NV4, NVX;
   gpu.cu:8718-8741, 11612-11650). Q6_K has none: a layer with any Q6_K
   matrix falls back to the float32 tiles (11656-11659), a large loss for
   the prompt. Add KT_Q6K (kt_load_w / kt_tile, the enum, KT_SET): about
   150 lines. Also choose the TC path per matrix (gate/up TC, down not) in
   that condition: about 20 lines.
5. The NVX fast path per matrix (gpu.cu:11823): today only when gate and
   down are both NVX. Not needed for M1/M2 (no NVX).
6. The 128 tail of Q6_K (M2 only): a new type (RQ6_K = 56 in gguf.py, row
   bytes cols / 256 * 210 + (cols % 256 == 128 ? 106 : 0)); the GPU
   kq_row_bytes (6303), kq_row_part (~6559), kq_dequant8 (8600), and KT_Q6K;
   the CPU kq_row_bytes (117), kq_block_values (~471), kq_row_scales (525),
   kq_prep (567), kq_dot_q6k (708), kq_tile4 (808, 833). About 150 to 250
   lines with tests. M1 needs none of this (gate/up have 2560 inputs).
7. Gate type != up type: not needed (UD never does it). It would need an
   up-type operand in both records and the TC launch (60 to 100 lines).
8. Per-expert types: not needed (UD types are per tensor) and large.
9. Zero copy (zc_dev): only all-NVX layers (qwen_gpu.py:406-407); a mixed
   file has none. No change.

## 6. The steps

1. Fix the warm slots (5.1), with a check that turns the warm slots on with
   the MTP layer (the bug of today).
2. gguf.py: RQ6_K = 56 (Q6_K blocks of the rotated rows; dequant: Q6_K
   then the inverse rotation; the rotation metadata as RQ8_0). For M1 the
   rows are whole blocks (2560); the tail comes with M2. Qwen4CPU.K maps 56
   to KQ_Q6_K as it maps 55 to KQ_Q8_0, and the "all rotated or none" check
   counts 55 and 56.
3. convert_q8_gguf.py --experts mix: the type of each (layer, matrix) from a
   map. --map ud (read the types of the UD-Q4_K_XL file: Q8_0 there ->
   RQ8_0, anything else -> RQ6_K) or a written list. A Q6_K quantizer in C
   (make_qx_quants, rmse_type 1, of llama.cpp; study_tq6_experts.q6k is
   the reference) after cops.tq6_rotate. The per-part error sample of the
   converter reports RQ6_K and RQ8_0 apart.
4. M1 on the GPU: per-layer calibrate_mix costs (5.3), KT_Q6K and the TC
   choice per matrix (5.4), MIX_CPU entries. Optional: slots per layer
   (5.2).
5. Measure M1 against the RQ8_0 file on the target: the quality
   (scripts/chat_quality.py on tests/texts/qwen38_chat.json: KL to the
   RQ8_0 file, top-1 agreement), and plain decode, MTP (3 drafts), pp512,
   the 8K prompt. check_q8_gguf.py, check_qwen4_cpu.py, check_qwen4_gpu.py.
6. If M1 gains: M2 (the 128 tail, 5.6), then a faster CPU Q6_K body if the
   CPU part is still the longer one. If M1 gains little: stop; the CPU part
   is the limit, and only fewer bytes per cold expert (NVFP4-class) moves
   it.

## 7. The scripts of this plan

In plan-scripts/ (plan-scripts/BF12_PLAN.md section 7 lists them; run from
numpy-gemma with PYTHONPATH=.):

- ud_vs_orig.py: the UD-Q4_K_XL tensors (g.dequant of the shards) against
  ORIG, with RQ6_K and Q8_0 of the same experts.
- bench_cpu_types.py: the cold experts of one layer on the CPU for each
  type (cops.kq_calib_nodes, random blocks with sane scales).
- bench_gpu_types.cu: the hot-expert products of one token for each type,
  with gpu.cu included read-only (nvcc -O3 -arch=native -I np_gemma/csrc;
  /usr/local/cuda/bin/nvcc, not on the PATH).
- mtp_fp8_vs_rq6k.py: the FP8 MTP experts against RQ6_K, Q6_K, Q8_0 of ORIG.

## 8. Phase 2: the dense matrices

The RQ8_0 file keeps the dense matrices in bfloat16 (9.9 GB; 4.95G
values). The runtime can requantize them to plain Q8_0 at load (--dense q8,
qwen4.dense_mode). This phase chooses a form for each dense matrix.

### What the data says

- UD-Q4_K_XL keeps every dense matrix at Q8_0 (the indexer BF16): Unsloth
  never went under 8 bits there. So RQ6_K is not "safe" for the dense
  matrices by the map of section 2: the choice is bf16, BF12, Q8_0, or
  RQ8_0.
- BF12 (plan-scripts/BF12_PLAN.md): the bf16 bits in 12.25 bits a value
  (a byte of sign and mantissa, a 4-bit exponent gap, a group exponent);
  99.992 to 99.995% of the values exact, the rest (under 2^-15 of their
  group's largest) zero by the neg0 code: -121.6 to -129.1 dB, 60 dB under
  the rounding of bf16 itself. Its test kernels run at the bandwidth of
  bf16: about 20% faster than bf16 on the GPU (qkv 105 against 135 us),
  24 to 29% on the CPU from DRAM, and the same as bf16 when a matrix is in
  the CPU cache. It is the form for "the dense matrices must stay bf16".
- The speed of Q8_0 dense (QWEN38_PLAN.md, the NVFP4 file; bf16 -> q8):
  decode 42.3 -> 51.2 tok/s, MTP 57.6 -> 70.5, the 8K prompt 649 -> 715,
  and 4.6 GB of the GPU freed for hot experts (42 -> 73 in each layer at
  256K). The largest lever left after the experts.
- The cost of Q8_0 dense (8192 tokens of source, KL to the float32
  products): bf16 0.0104 to 0.013, Q8_0 0.019; two float32 runs differ by
  0.0066; the NLL did not move. That KL also has the int8 x of the products.
- Q8_0 against RQ8_0 on the matrices of ORIG (study_orig_q8.py):

      matrix                 reads (the input vector)   Q8_0      RQ8_0     gain
      DeltaNet in_proj_qkv   mixed (block input)        -43.58    -45.67    2.08 dB
      DeltaNet in_proj_z     mixed                      -43.65    -45.66    2.01
      attn q_proj            mixed                      -44.02    -45.59    1.57
      attn k_proj            mixed                      -42.55    -45.88    3.33
      attn v_proj            mixed                      -42.18    -46.03    3.84
      DeltaNet out_proj      att (block output)         -43.97    -45.60    1.64
      attn o_proj            att                        -44.43    -45.55    1.12
      hc mix down            hn (4 normed streams)      -44.18    -45.58    1.40
      hc mix up              loa (320)                  -42.97    -45.77    2.80
      ple key/value          ple                        -45.43    -45.45    0.01
      lm_head                xn (the head mixer)        -45.10    -45.46    0.36
      embed_tokens           (rows)                     -45.24    -45.45    0.21

  Not measured yet: the MTP layer's dense matrices (eh_proj 2560x5120 on
  cat, its attention, its hc) and the indexer q/k (BF16 now).

### The rule: rotate by input vector, not by matrix

A rotation costs one transform of a vector; every matrix that reads that
vector then gets it for nothing. The vectors of a layer
(compile_qwen4_step, qwen4.py) and their readers:

    vector   size    quantized readers (big)              float readers (small)       rotate?
    hn       10240   hc_attn_down, hc_ffn_down            hc_*_inject (F32 4 x 10240)  yes (+1.4)
    loa      320     hc_attn_up, hc_ffn_up                -                            yes (+2.8)
    mixed    2560    attn_qkv, attn_gate | attn_q/k/v     ssm_alpha, ssm_beta (F32);   yes (+1.6 to +3.8)
                                                          indexer q/k (BF16 -> F32)
    att      6144    ssm_out | attn_output                -                            yes (+1.1 to +1.6)
    ple      2560    ple_key, ple_value                   -                            no (+0.01)
    xn       2560    output (the head)                    -                            no (+0.36)
    mixed    2560    (the MoE) routed + shared experts    ffn_gate_inp, _shexp (F32)   already (RQ8_0 experts)
    (MTP) cat 5120   eh_proj                              -                            measure first

- att on the full-attention layers with the TQ6 cache: attn_qsa already
  computes the output in the rotated form of tq6.py (the same signs, WHT32
  in each 32) and then undoes it (TQ_ROT inverse, qwen4.py attn_qsa). With
  an RQ8_0 attn_output that inverse rotation is dropped: the rotation of
  att is free there.
- The float readers of a rotated vector: either they read the unrotated
  copy (two copies of the vector, as the MoE input today: COPY, TQ_ROT,
  KQ_QUANT of mixr), or their weights are rotated once at load (W' = W R^-1
  in each 32 columns: exact in float32; ssm_alpha/beta, hc inject, the
  indexer q/k as F32). The second needs one copy of the vector, but every
  path must then read it rotated.
- Every reader of an RQ8_0 matrix must read the rotated x in every form it
  uses: the int8 x (KQ_QUANT) and the float x (the GPU kq_row path with
  NP_GEMMA_GPU_I8X=0, the bfloat16 planes of the prompt GEMMs, kq_tile on
  the CPU). Check each path, as for the MoE (RQ8_EXPERTS_PLAN.md: "every MoE
  path reads that x").

### The steps

1. The study: add the MTP dense matrices (eh_proj, its attention and hc),
   the indexer q/k, and the head mixer to study_orig_q8.py dense(); also
   a "rotated int8 x" measure: the output error of W x with x from a real
   run (dump hn, loa, mixed, att of a few layers on chat text; the int8 x
   of the products in both forms), since the rotation of x may matter more
   than that of W.
2. The cheap measure first, no new code: the RQ8_0 file with --dense q8
   (plain Q8_0 dense) against --dense bf16: scripts/chat_quality.py
   (tests/texts/qwen38_chat.json) KL and top-1, NLL; the rates (decode,
   MTP, pp512, 8K prompt). If plain Q8_0 dense is within the noise of two
   bf16 runs, stop here: --dense q8 is the answer, and neither RQ8_0 nor
   BF12 dense gives anything to see.
3. Else two ways, by the measure of step 2 and the memory left on the GPU:
   - BF12 dense (the values of bf16, no new rotation): the steps of
     plan-scripts/BF12_PLAN.md (gguf.py type 57, a dense mode "bf12" that
     converts the bf16 matrices at load, KQ_BF12X16 on the CPU, the
     kq_row_part branch and the bf16 tensor-core tile loader on the GPU).
     It needs no change of the program (no rotated vectors). Then measure
     bf12 against bf16 (KL under 1e-5, the same top tokens) and the rates.
     The converter reports, for each tensor it writes in BF12, the values
     it cannot hold (BF12_PLAN.md step 1): the count of zeroed values, the
     largest of them in absolute value and against the RMS of the tensor,
     the worst 10 with their row, column, value, and group maximum, the
     neg0 collisions (the exact -2^(E-15)), and any Inf or NaN (an Inf makes
     E = 255 and zeroes every finite value of its group). The rule (the
     user's): the zeroed small values are acceptable; a
     matrix with a large one (a zeroed value above 2^-8 of the tensor's
     RMS: an outlier pushed an ordinary value out of its group) or with an
     Inf or NaN stays bf16, always (no flag), and the report names it. The
     dense part is then a mix of BF12 and bf16 matrices, one type each,
     as the runtime already takes. On ORIG (qkv and ssm_out of layer 22,
     v_proj of layer 23) the largest zeroed value is 3.0e-5, at most
     1.6e-3 of the RMS: none is large, so no matrix stays bf16 there.
   - RQ8_0 dense (steps 4 to 6 below), if BF12 is too large or too slow
     and Q8_0 dense was measurably worse than bf16.
4. RQ8_0 dense: convert_q8_gguf.py --dense rq8 (a list of the
   matrices by the vector table above: RQ8_0 for the readers of hn, loa,
   mixed, att; Q8_0 for ple, the head; bf16 for the indexer, or F32
   rotated), the rotation metadata as for the experts.
5. The runtime, design A (simple): for each rotated vector a COPY, a
   TQ_ROT, and its KQ_QUANT, as mixr of the MoE; the float readers read the
   unrotated copy; drop the inverse TQ_ROT of att in the TQ6 layers. A flag
   per vector from the types of its readers (all RQ8_0 or none: assert).
   About 4 more records a layer: on the GPU about 190 small kernels a step
   (the trace of QWEN38_PLAN.md: 1300 small kernels cost 4.5 ms), so about
   0.7 ms; on the CPU about 600 WHT32 a layer for a token, under 1% of a
   step.
6. Design B (if A costs too much): a KQ_QUANT with the rotation inside
   (rotate in registers, then the int8; one record, no copy), with the
   float readers' weights rotated at load, and HC_NORM / HC_ACT / GDN /
   SIGMUL writing the rotated vector where they produce it.
7. The MTP layer: the same rule for its dense matrices (blk.48), after the
   measure of step 1 (BF12 or RQ8_0, as the main layers).
8. The checks: check_qwen4_cpu.py and check_qwen4_gpu.py (the CPU program
   against the NumPy model on the true values; the GPU against the CPU),
   check_q8_gguf.py (the dense parts in dB), then step 2's measures for
   bf16, BF12, Q8_0, and RQ8_0 dense.

The sizes: the dense part is 9.9 GB in bf16, 7.6 GB in BF12, and 5.3 GB in
Q8_0 or RQ8_0 (the head and the embeddings: 1.36 GB of it in Q8_0).
