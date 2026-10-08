# Plan: the experts of Qwen3.8 in Q8_0 from the original model (rotation: only where it pays)

Status: a study only. No runtime code has changed. The numbers
come from two new scripts:

- scripts/study_tq6_experts.py: the forms on the bfloat16 shared experts of
  the ModelOpt checkpoint (the first data; about two minutes);
- scripts/study_orig_q8.py: the forms on the original weights (ORIG), the
  routed experts and the dense matrices (about six minutes).

ORIG, the original bfloat16 checkpoint (Qwen/Qwen3.8-Flash-Next, 336 GB,
131 shards), is at models/Qwen3.8-Flash-Next.

History: the study began with TQ6 (np_gemma/tq6.py) for the FP8 experts of
the MTP layer, then rotated Q6_K, then rotated Q8_0 ("RQ8_0"), which the
user chose on the data of the shared experts. The routed experts of ORIG
then showed that the rotation gains only 0.2 dB on them (section 2). So the
plan is now: plain Q8_0 of ORIG for the experts (no new kernel), and the
rotation as a later, optional part for the matrices with heavy tails (the
dense matrices and the shared experts), if a measure on the model asks for
it. The user has the last word on that change.

## 0. Progress

The user's choice (after the plain Q8_0 run below): the rotation for all
the experts, now, not as a later part C. The routed experts, the MTP
experts, and the shared experts are RQ8_0 (type 55); the dense matrices
stay bfloat16 in the file (the rotation matters only for quantized
matrices). The plain Q8_0 file was replaced by
/mnt/pmem/Qwen3.8-Flash-Next-RQ8-GGUF/Qwen3.8-Flash-Next-RQ8-bf16.gguf
(scripts/convert_q8_gguf.py, --experts rq8 the default). Against ORIG (a
sample): routed -45.45 dB (Q8_0 -45.2), MTP -45.46, shared -45.67 (Q8_0
-43.5).

The runtime of RQ8_0 (no new kernels of products):

- gguf.py: RQ8_0 = 55 (the Q8_0 block), dequant gives the true values
  (Q8_0, then the inverse rotation), and a file with RQ8_0 tensors must
  have np_gemma.rotation.group/signs/order equal to the tables of the build.
- Qwen4CPU: the RQ8_0 tensors are KQ_Q8_0 to the kernels; rot_experts
  (all the experts or none). compile_qwen4_step: after the routers, a copy
  of the MoE input, TQ_ROT, KQ_QUANT: every MoE path (the CPU program, the
  hot split, KQ_HOT_MOE, KQ_GROUP_MOE, the CPU part of the GPU) reads that
  x; the routers read x.
- The act of each pair before down: kquants.c kq_moe_rot (both MoE
  bodies), gpu.cu gg_moe_rot (k_kqh_act, k_qmoe_act; tq6_rot_warp). A
  state of the process, set when the model loads.

The checks of RQ8_0: check_q8_gguf.py PASS (routed and MTP experts -45.4
to -45.5 dB, shared -45.5 to -46.1, the dense bits of the NVFP4 GGUF);
check_qwen4_cpu.py PASS (the CPU program, x and act rotated, against the
NumPy model on the true values: logits within 1.2e-2, the same top token);
check_qwen4_gpu.py PASS (the 48 tokens of the CPU, verify and MTP exact).

    experts      plain   MTP (3 drafts)   8K prompt   NLL (8K of source)
    RQ8_0        29.5    34.5 (70%)       633         0.9201
    Q8_0         30.2    35.0 (66%)       686         0.9190
    NVFP4        42.3    57.6             910         0.9193

KL to RQ8_0: Q8_0 0.0124 (about the noise of two runs: 0.007 to 0.012),
NVFP4 0.0354. The prompt of RQ8_0 was 8% slower in one run (a copy, a
rotation, and a quantization more in each layer; the machine was also
reclaiming the page cache of the reads of ORIG): to measure again. Next: a
chat transcript (the quality), the Q8_0 block kernel of the hot experts.

### The plain Q8_0 run (before the rotation)

Done: the file (part A and the conversion of part B), the NUMA layouts 1
and 3, the first measures. Not done: the Q8_0 block kernel of the hot
experts, layout 2, the MTP comparison against the FP8 file, and the
quality on a chat transcript. Part C waits on those.

- ORIG is /space/models/Qwen3.8-Flash-Next on this machine (an HDD, about
  220 MB/s). scripts/convert_q8_gguf.py wrote
  /mnt/pmem/Qwen3.8-Flash-Next-Q8-GGUF/Qwen3.8-Flash-Next-Q8-bf16.gguf
  (195.7 GB, 26 min, the source read ahead in the order of the output).
  The NVFP4 GGUF is no longer on Optane; its copy is in models2 (NFS).
- scripts/check_q8_gguf.py: PASS. Experts -44.3 to -45.4 dB (layers 0, 13,
  26, 47, MTP), n-gram rows -45.4 dB, and the dense tensors the bits of the
  NVFP4 GGUF (the tiled DeltaNet heads too).
- Loading: gguf._stage_dax puts the experts on node 0 when the staged
  tensors pass 85% of the module's node (NP_GEMMA_DAX_EXPERTS), through two
  buffers on node 1 (node-1 threads writing node-0 pages went at 1.8 GB/s;
  now 9.9 GB/s, 141 GB in 14 s).
- Layout 1: QwenGPU._numa_partial copies the most used experts of each
  layer to node 1 (NP_GEMMA_GPU_NODE1_GB, auto: 70% of node 1 less the
  tensors staged there: 329 of 512, 84 GB, 8 to 24 s) into a compact array;
  KQ_MOE takes slot1 (the slot of each expert, or -1): a thread of node 1
  reads that copy for those experts, and kq_moe_small_body puts them last
  in its list, so the second half of the team (node 1) takes them.
  NP_GEMMA_EXPERT_COUNTS: an .npy of HotCache.profile() (all the uses of
  each expert) picks them.
- check_qwen4_gpu.py: PASS (the 48 tokens of the CPU, verify and MTP exact).

The rate (bf16 dense, TQ6 cache, 48 CPUs at 2.4 GHz; the NVFP4 model as of
QWEN38_PLAN.md):

    experts                        hot/layer  plain   MTP (3 drafts)  8K prompt
    NVFP4                          -          42.3    57.6            910 tok/s
    Q8_0, one copy (layout 3)      36         25.4    29.5            -
    Q8_0, 329 on node 1, by index  36         29.2    34.3            -
    Q8_0, 329 on node 1, profile   36         30.2    35.0 (66%)      686

The quality (scratchpad qacc.py: 8192 tokens of source, the log-probs of
every 64th position):

    Q8_0 experts       NLL 0.9190, top-1 78.0%
    NVFP4 (bf16 tc)    NLL 0.9193; KL to Q8_0 0.0326, top-1 agree 94.2%
    NVFP4 (float32)    NLL 0.9208; KL to Q8_0 0.0363, top-1 agree 94.5%

So the NVFP4 experts move the logits by KL 0.033 (the products of the
kernels: 0.010 to 0.015), but the NLL on this text does not move. A chat
transcript is the next measure.

## 1. The layout of ORIG

- The names are those of the ModelOpt checkpoint, but the experts are fused:
  model.language_model.layers.N.mlp.experts.gate_up_proj (512, 1280, 2560)
  with gate in rows 0-639 and up in rows 640-1279 (checked against the
  NVFP4 of ModelOpt: correlation 0.9967 for each), and
  ...experts.down_proj (512, 2560, 640). The same for
  mtp.layers.0.mlp.experts.*. All bfloat16.
- The shared experts and the head of ORIG are bit-equal to the bfloat16
  ones of the ModelOpt checkpoint (checked): ModelOpt quantized only the
  routed experts (NVFP4), the MTP experts (FP8 128x128), and the n-gram
  table (FP8).

## 2. The results (against ORIG)

The routed experts: 6 experts of each of layers 0, 12, 24, 36, 47 and the
MTP layer, gate, up, down: 108 matrices. Their kurtosis is 3 to 7 (near
Gaussian; the shared experts have up to 118).

    form                     bits    mean      worst     the MTP layer
    Q8_0                     8.5     -45.24    -44.37    -45.20 dB
    rotated Q8_0             8.5     -45.45    -45.43    -45.45
    rotated Q8_K             8.125   -43.18    -43.11    -43.17
    rotated Q6_K             6.56    -35.04    -35.01    -35.04
    Q6_K, 128 tail           6.56    -34.87    -34.13    -34.84
    TQ6                      6.5     -32.65    -32.63    -32.65
    NVFP4, searched          4.5     -21.32    -21.29    -21.32
    ModelOpt as shipped      -       NVFP4 -21.80 (layers 0-47), FP8 -31.50 (MTP)

The dense matrices, Q8_0 against rotated Q8_0 (the head and the embeddings:
40000 random rows):

    matrix                 kurtosis   Q8_0      RQ8_0     gain     worst Q8_0 / RQ8_0
    DeltaNet in_proj_qkv   23         -43.58    -45.67    2.08 dB  -43.0 / -45.5
    DeltaNet in_proj_z     44         -43.65    -45.66    2.01     -42.9 / -45.5
    DeltaNet out_proj      25         -43.97    -45.60    1.64     -43.8 / -45.6
    attn q_proj            20         -44.02    -45.59    1.57     -43.5 / -45.5
    attn k_proj            30         -42.55    -45.88    3.33     -41.3 / -45.6
    attn v_proj            134        -42.18    -46.03    3.84     -41.5 / -45.8
    attn o_proj            30         -44.43    -45.55    1.12     -44.3 / -45.5
    hc mix down            49         -44.18    -45.58    1.40     -43.5 / -45.5
    hc mix up              117        -42.97    -45.77    2.80     -42.6 / -45.7
    ple key/value          26         -45.43    -45.45    0.01     -45.4 / -45.4
    shared experts         118        -43.54    -45.68    2.14     -42.7 / -45.5
    lm_head                4          -45.10    -45.46    0.36     -45.1 / -45.5
    embed_tokens           3          -45.24    -45.45    0.21     -45.2 / -45.5

What this says:

- Q8_0 of ORIG is the change that matters for the experts: -45.2 dB in
  place of -21.8 (NVFP4) for the main layers and -31.5 (FP8) for the MTP
  layer. The rotation adds 0.2 dB (1 dB on the worst matrix): the routed
  experts have no heavy tails to spread.
- The rotation pays where the tails are heavy: the dense matrices and the
  shared experts, 1 to 4 dB. All the rotated forms land at -45.5 to -46 dB,
  whatever the kurtosis.
- Would the dense gain show? The model data of QWEN38_PLAN.md (a prompt of
  8192 tokens of source, KL to the float32 products): bf16 dense 0.0104 to
  0.013, Q8_0 dense 0.019, and two float32 runs differ by 0.0066; the NLL
  did not move (Q8_0 0.9205, float32 0.9208). So Q8_0 dense adds about
  0.006 to 0.009 of KL. If the KL follows the error power, 2 dB less error
  takes about a third of that away (0.002 to 0.003): under the noise of two
  float32 runs, not visible in the text. But that KL also has the int8 x of
  the dense products, and the rotation spreads the outliers of x as well
  (the residual of this model has large channels). Only a model measure can
  say how much of the 0.006 to 0.009 the rotation removes.

## 3. Part A: the MTP layer in Q8_0 of ORIG (no runtime change)

1. scripts/convert_mtp_q8.py ORIG OUT: the MTP layer as a GGUF of blk.48
   alone (as the MTP file of Unsloth; Qwen4CPU(path, mtp=...) takes the
   MTP layer from its own file). The experts and the shared expert in Q8_0
   from ORIG (kq_to_q8_0 on the bfloat16 rows; the fused gate_up split at
   row 640); the rest of blk.48 and nextn.* as convert_nvfp4_gguf.py makes
   them (they are the same bfloat16 in ORIG).
2. Check: the experts of the file against ORIG (-45.2 dB, the numbers of
   study_orig_q8.py); the file loads and the MTP layer runs
   (check_qwen4_mtp.py).
3. The measure, against the MTP file of now (FP8 -> Q8_0, -31.3 dB): the
   agreement of the top token of the MTP layer, the rate of the accepted
   drafts and tok/s (bench_mtp_gpu.py; 1 and 3 drafts; a code answer,
   prose, a chat transcript: the memory note says judge on chat text).
   Greedy text equals the plain decode. Memory and time as now.

## 4. Part B: the routed experts in Q8_0 of ORIG

### The memory

    the routed experts      an expert   48 x 512 (one copy)
    NVFP4 (KQ_NVX, now)     2.78 MB     about 71 GB (141 GB with a copy on each node)
    Q8_0                    5.22 MB     128.3 GB (+ 2.7 GB of the MTP layer)

- The target machine: node 0 192 GB, node 1 129 GB, and the Optane region
  of 252 GiB on socket 1.
- Node 0 holds one full copy and the rest (dense 4 to 8 GB, the head and
  the embeddings, the caches, the pinned buffers of the GPU).
- Node 1 cannot hold a second full copy (131 GB). Today each node has its
  own copy (_numa_copy_experts, NP_GEMMA_GPU_NUMA_COPY): every read of the
  CPU part is local, and that gave 52.7 tok/s against 32.2 for the staged
  copy on node 1 alone (QWEN38_PLAN.md). The layouts to try, in order:
  1. A full copy on node 0 and a partial copy on node 1 (about 90 to 100
     GB: the experts used most, by the counts of HotCache or a profile of a
     chat). A cold expert with a copy on node 1 goes to the threads of node
     1, the others to node 0. kq_moe_small_body picks mats or mats1 by
     kq_my_node; it needs a table "has a copy on node 1" for each expert
     and a split of the tasks by it.
  2. One copy, split: expert e of each layer on node e % 2 (64 GB a node),
     each node's threads on its own experts. A note of QWEN38_PLAN.md says
     this gave no gain once the threads were spread, but that was next to
     full copies; measure it again.
  3. One copy on node 0 only: the floor.
- Optane: the GGUF is about 195 GB (section 6). It fits the region alone,
  not next to the NVFP4 file.
  gguf._stage_dax stages the file on node 1, then QwenGPU copies the
  experts to node 0; 131 GB of staged experts do not fit node 1. Stage the
  experts straight to node 0 (the read threads on node 1, the pages on node
  0), and only the partial copy of layout 1 on node 1.

### The time

- A cold expert reads 1.9 times the bytes of NVFP4. In the decode the CPU
  part was 10 ms of a step of 24.7 ms (208 us a layer for 12.5 MB), not
  bound by the bandwidth (the start of many short streams): measure, do
  not scale.
- The same GB of GPU slots hold 53% of the experts, so more are cold. The
  GPU hot kernel for Q8_0 is kq_row (a warp for each row); KQ_NVX has a
  block kernel for 16 rows (k_kqh_nvx: the hot experts 4.8 -> 1.7 ms a
  step). A Q8_0 form of it (16 rows, as KQ_Q8X16 of the CPU) is likely
  needed.
- The prompt: the GPU MoE gemm has Q8_0 (KT_Q80); the copies of a mixed
  group are 1.9 times larger.
- If the time is too high: rotated Q6_K is -35.0 dB at 6.56 bits (99 GB a
  copy), with a block of 128 at the end of the rows of down (the Q6_K
  kernels run half by half; the tail is one more half).

### The steps

1. scripts/convert_q8_gguf.py ORIG OUT: the GGUF of section 6. Out to
   models2 (466 GB free at the time) or the Optane region.
2. The NUMA layouts above (QwenGPU._numa_copy_experts, kq_moe_small_body,
   gguf._stage_dax), and the Q8_0 block kernel of the hot experts.
3. The checks:
   - The quality: KL and top-1 agreement of the logits on a chat transcript
     (the memory note), the NVFP4 model against the Q8_0 model. At -45 dB
     the Q8_0 model is close to ORIG, so it serves as the reference; check
     it against transformers on a short prompt where the RAM allows
     (scripts/check_qwen4_st.py has the pattern).
   - The rate: plain and MTP decode, pp512 and pp2048, against the NVFP4
     model, with each NUMA layout.

## 5. Part C (optional): the rotation for the matrices with heavy tails

Do it only if a measure asks for it: run the model (part B) with the dense
matrices in bfloat16 and in Q8_0 (--dense, NP_GEMMA_DENSE) on a chat
transcript. If Q8_0 dense against bf16 dense is under the noise (two
float32 runs: 0.0066), stop here.

- The form RQ8_0: the bytes of Q8_0, of the row rotated in each 32 values
  (gemma_tq6_rotate of bf16_linear.c, then kq_to_q8_0). w.x = (R w).(R x),
  so the Q8_0 kernels run as they are on the rotated x. A GGUF type
  KQ_RQ8_0 = 55 (check that 55 is free in gguf.py and kquants.c); in
  memory KQ_Q8_0 with a bit KQ_ROT (0x1000) that the kernels mask.
- Where: the dense matrices (largest gains: attn v_proj and k_proj 3.3 to
  3.8 dB, hc mix up 2.8, the DeltaNet in_proj 2.0) and the shared experts
  (2.1). The head and the embeddings gain 0.2 to 0.4 dB: leave them.
- The rotation of x: once for each input vector that feeds RQ8_0 matrices
  (the input of the attention or the DeltaNet block, the input of the hc
  mixers, the act of the shared expert, the output of the attention before
  o_proj). One rotated copy serves all the matrices that read the same x.
  On the CPU before KQ_QUANT and in kq_moe (the shared expert is one more
  pair); on the GPU a GP_TQ_ROT record (it exists) before the product, and
  tq6_rot_warp in the act kernels. The prompt paths too (the tensor-core
  gemms read the rotated x).
- The measure: KL and top-1 on a chat transcript against bf16 dense; the
  int8 x of the products then also gets the rotation, which may be the
  larger part of the gain.

## 6. The custom GGUF

One GGUF file for this runtime (as the file of convert_nvfp4_gguf.py:
llama.cpp cannot read its own types), made from ORIG by
scripts/convert_q8_gguf.py.

### The new type

Only one: RQ8_0 = 55 (part C; parts A and B need none, Q8_0 is type 8).
51 to 53 are the file types NV4, E4M3_ROWS, NVX of gguf.py; 54 (KQ_Q4X)
and 60 to 62 are the in-memory types of kquants.c; 55 is free in both
(check again before the work).

- The bytes: those of Q8_0 (blocks of 32 values, 34 bytes: a float16 d and
  32 int8). A block is one rotation group, so each block decodes alone: the
  values are R^-1 (d q), R the rotation of tq6.py (the signs TQ6_SIGNS,
  WHT32 with the butterflies of stride 1 first, / sqrt(32)).
- gguf.py: RQ8_0 = 55 in _BLOCK (32, 34), _BLOCK_DT (the Q8_0 block),
  _TYPE_NAME ("RQ8_0"), tensor_bytes; _dequant: the Q8_0 values, then
  tq6.unrotate on each 32 (so g.dequant and the checks see the true
  values). cops.kq_rows on a RQ8_0 table: dequant, then unrotate (only if
  a table is ever RQ8_0; "The n-gram table" below says no).
- The metadata of the rotation, so a file cannot be decoded with other
  tables: np_gemma.rotation.group = 32, np_gemma.rotation.signs =
  0x0068ABD8 (TQ6_SIGNS of tq6_tables.h), np_gemma.rotation.order =
  "wht32-stride1-first". The reader asserts that they equal the tables of
  the build when the file has an RQ8_0 tensor.
- In memory: KMat of a type 55 tensor gets type 8 (KQ_Q8_0) with the bit
  KQ_ROT (0x1000) in the type field of the mats tables and of the GPU
  operands; the kernels mask the bit, the code before them rotates x (part
  C). Until part C is done, the loader stops with a clear message on a
  type 55 tensor (not a silent Q8_0 read).

### The tensors of the file

    part                              form                         size
    routed experts (48 x 512) + MTP   Q8_0 (--experts q8; rq8      131.0 GB
      experts (512)                   is the option of part C)
    shared experts (49)               Q8_0, or RQ8_0 (part C)      0.25 GB
    dense matrices                    bfloat16 (--dense bf16), Q8_0 9.9 / 5.3 GB
                                      (q8), or RQ8_0 (rq8, part C)
    head, token embeddings            Q8_0 / bfloat16 as now        -
    n-gram table (51.2G values)       Q8_0 of ORIG (--ngram q8),   54.4 GB
                                      or the E4M3 rows of ModelOpt (51.2 GB)
                                      as now (--ngram e4m3)
    norms, small matrices, routers    float32 (the 1 of the norms   -
                                      added, as convert_nvfp4_gguf.py)

About 195 GB with bfloat16 dense and the Q8_0 n-gram table.

- The names, the metadata (metadata() of convert_nvfp4_gguf.py), the MTP
  layer as blk.48 with nextn.*, the value heads of the DeltaNet in the
  tiled order of llama.cpp (reorder of that converter): as the NVFP4 file.
  Plus general.source = "Qwen/Qwen3.8-Flash-Next (original bfloat16)" and,
  for each quantized group of tensors, the mean error in dB against ORIG
  (np_gemma.quant_error.<part>), measured while writing.
- The experts: ORIG has them fused (gate_up_proj (512, 1280, 2560), gate in
  rows 0-639; down_proj (512, 2560, 640)). The GGUF keeps the stacks of
  llama.cpp: blk.N.ffn_gate_exps.weight, ffn_up_exps, ffn_down_exps, dims
  (cols, rows, 512). The maker of a stack returns a list of 512 arrays (one
  Q8_0 expert each; write_gguf writes a list in turn), so the memory stays
  at one expert, read from the map of ORIG.
- The tokenizer and the chat template: copied next to the file, as now.
  The vision weights (model.visual.*, 0.45G) stay out: --mmproj reads them
  from a checkpoint directory.

### The n-gram table

The table is read 16 rows a token, so its size is the only cost. Against
ORIG (200000 rows of 5 shards; kurtosis 3.1):

    form                         bits   table      per row: median  99%      worst
    FP8, one scale (ModelOpt)    8.0    -31.49 dB  -31.52           -30.16   -29.06 dB
    Q8_0 of ORIG                 8.5    -45.46     -45.52           -43.91   -42.60
    rotated Q8_0 of ORIG         8.5    -45.43     -45.50           -43.88   -42.26

- Q8_0 of ORIG is 14 dB better than the FP8 of ModelOpt, for 3.2 GB more.
  The rotation gains nothing (the rows are Gaussian): plain Q8_0, no RQ8_0
  for the table.
- The read: Qwen4CPU.ple_rows takes the E4M3 rows (type 52) by kq_gather,
  and any other type by cops.kq_rows (Q8_0 works there now). Add the
  advise_random of the map for a Q8_0 table too, and a gather of the 170
  bytes of a row by many threads (as kq_gather), then the dequant.
  gguf._stage_dax keeps the table on Optane (NP_GEMMA_DAX_KEEP): the same.
- The default: --ngram q8. --ngram e4m3 copies the rows of the ModelOpt
  checkpoint (or of the NVFP4 GGUF) as they are, if the 3.2 GB matter.

### The checks of the file

- scripts/check_q8_gguf.py OUT ORIG: for a sample of each part (experts of
  some layers, the shared experts, the dense matrices, n-gram rows),
  g.dequant against ORIG: the dB of section 2 (Q8_0 about -45, RQ8_0 about
  -45.5); the tiled order of the DeltaNet heads against the NVFP4 GGUF
  (the dense tensors of the two files are equal when both are bfloat16).
- The model loads the file and gives the logits of the NVFP4 model within
  the expected KL (part B, step 3).

## 7. Risks

- Part B can make the decode much slower than NVFP4 (twice the bytes of
  each cold expert, half the hot experts). Measure the quality gain on
  chat text against that cost.
- The sample of the study is 108 routed matrices (6 experts of 6 layers);
  the per-layer numbers are flat (within 0.3 dB), so a surprise is
  unlikely. The converter can print the error of each matrix it writes.
- A path that reads RQ8_0 rows outside the products (a check, a dequant)
  must unrotate (tq6.unrotate after the Q8_0 dequant).
