# Handoff: Qwen3.8-Flash-Next in this runtime

This file gives the state of the work on Qwen3.8-Flash-Next (qwen4exp). It
tells what works, the numbers, the files, the open work, and the next steps.
QWEN38_PLAN.md has the full plan and the history of each phase.

## 1. The state now

- The work is on master in the repo chadslab. Do not push. Commit with the
  lines of the session at the end (see the git log).
- The routed experts of the GGUF are in groups of 16 rows (type 53,
  KQ_NVX; section 4). The code still reads type 51 (the file of type 51 is
  deleted; the converter makes it with --experts nv4).
- Do not stop serve.py (port 8080). The user runs other programs on this
  machine; the numbers change with that load.
- Do not use git stash in this repo. It took the changes of the user in
  notebooks/.
- The user keeps notebooks/btrees-chatgpt4.ipynb changed; do not commit it.

## 2. The models and the files on disk

    models/Qwen3.8-Flash-Next-NVFP4/          the checkpoint of NVIDIA ModelOpt (124 GiB)
    models2/Qwen3.8-Flash-Next-NVFP4-GGUF/    the GGUF of this runtime (132 GB, type 53) and
                                              tokenizer.json
    models2/Qwen3.8-Flash-Next-GGUF/          the GGUF of Unsloth (UD-Q4_K_XL), moved here
                                              by the user (it was in /spaceu1, and in models/)
    llama.cpp-qwen4exp/build-cuda, build-cpu  llama.cpp (qwen4exp branch); build-cpu has no CUDA

- models2 is on another NVMe partition (nvme1n1p2); it is in .gitignore.
- The Qwen3.6 tokenizer (models/Qwen3.6-35B-A3B-OptiQ-4bit) is gone. Use the
  tokenizer.json of the checkpoint or of the GGUF of this runtime. Some check
  scripts still have the old paths as their defaults: give --path and --tok.
- /home has only 38 GB free.

## 3. What works (committed)

The model has 48 layers (DeltaNet and QSA attention) and 512 experts (10 for
each token). It also has a shared expert, 4 residual streams, the n-gram
table, and an MTP layer. The code:

    np_gemma/qwen4.py         the NumPy model (Qwen4), the CPU program (Qwen4CPU),
                              MTP on the CPU, dense_mode (bf16 or q8)
    np_gemma/qwen4_gpu.py     Qwen4GPU: the GPU with the experts split, HotCache,
                              the MTP layer on the GPU, mixed prompt groups
    np_gemma/st_qwen4.py      NVFP4Source: reads the checkpoint directory
    np_gemma/gguf.py          reads and writes GGUF; the types 50/51 and 52
    np_gemma/csrc/kquants.c   the CPU products (GGUF formats, BF16, NVFP4, Q8X16)
    np_gemma/csrc/gpu.cu      the GPU program interpreter and kernels
    np_gemma/csrc/qsa.c, hyperconn.c, moe.c   the CPU records of the new parts
    scripts/convert_nvfp4_gguf.py   the checkpoint to one GGUF of this runtime
    scripts/bench_qwen4.py    pp/tg rates as llama-bench measures them
    scripts/check_qwen4_*.py  the checks (st, gpu, mtp, cpu, indexer, ngram)

The main parts, and where they are:

- The GGUF of this runtime: the experts are NVFP4 rows (a type of this
  runtime). The n-gram table is FP8 rows and a scale. The dense matrices
  are BF16. The MTP layer is blk.48. The runtime maps the file; it does
  not repack the experts at each start. llama.cpp cannot read this file.
- dense_mode: "q8" requantizes the BF16 matrices to Q8_0 at the first use,
  for the CPU and the GPU (Qwen4CPU.K). The variable NP_GEMMA_DENSE takes
  q8 (the default), bf16, or auto. The GPU is slower with bf16 here too.
  With auto, a GPU of little free memory takes q8.
- The CPU: KQ_Q8X16 (type 60, only in memory) packs the dense Q8_0
  matrices in groups of 16 rows; Qwen4CPU.KP makes it at the first use.
  NVFP4 has tiles of 4 rows by 4 tokens (kq_rows4_nv4). A group gives the
  same bits as steps.
- The GPU:
  - The hot experts on the GPU; the CPU computes the cold experts.
  - Steps and small groups take x as int8 (dp4a).
  - The prompt runs in mixed groups (256 to 1024 rows). MOE_PLAN splits the
    experts of each layer. The GPU copies the experts with the most tokens.
    The CPU computes the others at the same time.
  - The products of large groups use the tensor cores (int8 mma).
  - The attention of large groups is one record for each layer.
- MTP (3 drafts) works on the CPU and on the GPU. On this GPU it gives
  about the rate of the plain decode: the cold experts of a verify group
  are slow.
- The file builds for sm_86 (the 3090): no griddepcontrol there.

## 4. The layouts of the NVFP4 experts

- Type 51 (KQ_NV4): a row holds all its codes (16 bytes for each block of
  32 values). Then come its E4M3 scales and the float32 scale of the
  matrix, padded to 16 bytes. The GPU loads a fragment with one aligned 32-bit
  load. kt_e2m1x4 changes 4 codes to int8 with prmt and a sign mask (as
  Marlin).
- Type 53 (KQ_NVX, the GGUF now): groups of 16 rows. A group holds 16 bytes
  (the scale of the matrix), then 288 bytes for each block of 32 columns: 8
  steps of 32 bytes and 32 scales. Step s holds values 4s to 4s + 3 of the
  16 rows. Byte 4r + u has row r in its low 4 bits and row r + 8 in its
  high 4 bits.
  - The CPU (kq_nvx_rows): one vpdpbusd for each step, a lane for each row.
    The codes become the value times 2, plus 12 (unsigned). The sum starts
    at -12 times the sum of x of each 16 values.
  - The GPU: the 4 bytes at 32 s + 4 r are the tensor-core fragments of
    rows r and r + 8. A step of a tile is 8 contiguous groups. The step
    kernel of the hot experts uses a warp for each group.
- Checked: the dequant of type 53 equals that of type 51. All the GPU paths
  (tensor cores, float32 tiles, the step kernel) have the same errors
  against the CPU for the two types. A token alone and in a group gives the
  same bits on the CPU. check_qwen4_st: PASS, and the GPU gives the 48
  tokens of the CPU.
- One layer of experts: the CPU with 512 tokens takes 33 ms (type 51: 75
  ms). The GPU tensor cores with 1024 tokens (16 experts) take 6.2 ms
  (8.9 ms). The GPU step kernel is 2.3 to 3 times faster.
- MOE_PLAN gives the CPU more experts for type 53 (qwen4_gpu.mix_cpu_cost:
  50 us for each expert and 5 us for each token; 75 and 15 for the other
  types).
- scripts/convert_nvfp4_gguf.py writes type 53 (--experts nv4: type 51), in
  197 s.

## 5. The numbers

The machine: Xeon W-2295 (18 cores), 188 GB; RTX 5060 Ti with about 8 GB
free, PCIe Gen3 x8.

The GGUF of this runtime, llama-bench method (scripts/bench_qwen4.py):

    GPU (0.5 GB hot experts)   pp512 327   pp2048 496   pp4096 493   tg128 22.8   tg512 23.2 tok/s
    CPU (dense q8)             pp512 93.1  tg128 7.61
    type 51, the same code     GPU: pp512 305, pp2048 441, tg128 20.2; CPU: pp512 64.4, tg64 7.2
    llama.cpp CPU              pp512 27.9   tg128 5.0
    llama.cpp GPU (-ncmoe 48)  pp512 101    pp2048 101  tg128 18.3

- A test of 32001 tokens: the source of program.py, qwen4_gpu.py, and a
  part of qwen4.py, then 3 questions. On the GPU with 0.5 GB hot, the
  prompt takes 100 s (319 tok/s), and the decode gives 19.1 tok/s. The 3 answers
  are correct (MOE_PLAN is 123, MIX_SIZE is 1024, and the split of the
  experts).
- The int8 x of the GPU steps: decode from 16.1 to 19.1 tok/s (one run).
- KQ_Q8X16 on the CPU: the dense products of a prompt of 512 from 5.5 s to
  2.55 s (1.19 T multiply-adds/s against 0.38).
- The float32 and bfloat16 matrices of the CPU (types 61, 62): see
  section 8.

The time of each part (the profiles). Other programs ran on
the machine, so a time can change by 15%:

    CPU, a prompt of 512 (dense q8, 6.1 s)   the experts 1.8 s (about 43 GB/s: the memory),
                                             the dense products 1.9 s, GDN 1.1 s, HC_MIX 0.43 s,
                                             ATTN_QSA 0.36 s, KQ_QUANT 0.31 s
    CPU, a decode step (119 ms)              the dense products 72 ms (51 to 64 GB/s; 320 x
                                             10240 about 52 GB/s), the experts 35 ms (45
                                             GB/s). About 5.8 GB for each token: 49 GB/s, 74%
                                             of the 66 GB/s of the memory.
    GPU, a group of 1024 (1.88 s; 2.67 s     FETCH_WAIT 0.43 s (the copies, 6.8 GB/s), the dense
    with type 51)                            products 0.32 s, the experts on the GPU 0.28 s,
                                             CPU_WAIT 0.17 s, GDN 0.12 s. MOE_PLAN: 27 of 171
                                             experts copied, 183 on the CPU, in each layer.
    GPU, a decode step (47 ms)               CPU_JOIN 22 ms (the cold experts on the CPU: about
                                             0.9 GB at 43 GB/s), the dense products 15 ms (3.9
                                             GB: 260 GB/s, 58% of the GPU), the hot experts 3.6 ms.

## 6. How to run the checks

    G=models2/Qwen3.8-Flash-Next-NVFP4-GGUF
    python scripts/check_qwen4_st.py --path $G/Qwen3.8-Flash-Next-NVFP4-bf16.gguf \
        --tok $G/tokenizer.json --layers 4 --gpu --hot-gb 0.5
    python scripts/bench_qwen4.py -m $G/Qwen3.8-Flash-Next-NVFP4-bf16.gguf --backend gpu \
        --hot-gb 0.5 -p 512,2048 -n 128 -r 2
    NP_GEMMA_DENSE=q8 python scripts/bench_qwen4.py -m $G/Qwen3.8-Flash-Next-NVFP4-bf16.gguf \
        -p 512 -n 64 -r 2

The long-context test (128K tokens of the source, then questions with a
long answer):

    python scripts/long_qwen4.py [--tokens 131072] [--gen 4096] [--backend cpu]

The server (scripts/serve_qwen4.py) runs all the work of the model in one
thread. The CPU part of a step runs in the thread of the step, with an
OpenMP team of its own. With a team for each request, libgomp had more
threads than CPUs, and its barriers slept: 17 tok/s, not 22.

The switches for a comparison:

    NP_GEMMA_GPU_KQTC=0     no tensor cores
    NP_GEMMA_GPU_I8X=0      float32 x in steps
    NP_GEMMA_X16=0          no groups of 16 rows on the CPU (types 60, 61, 62)
    NP_GEMMA_GPU_ATTN_MT=0  the attention of each query
    NP_GEMMA_GPU_MIX=0      no mixed groups

## 7. The next steps

1. The 8-bit product of 320 rows (hc_*_down, 20 groups for 18 threads)
   gives about 52 GB/s in a decode step, not 64. Two halves of the columns
   as tasks were not faster (section 8).
2. The GPU decode of dense matrices with short rows (hc_*_up: 320 values):
   a lane for each row. A GPU kernel that reads KQ_Q8X16 can be a test.
3. Memory: in q8 mode, the Q8_0 copy stays next to KQ_Q8X16 (3.9 GB more);
   drop it when no GPU uses it.
4. The 3090 (sm_86, 24 GB, PCIe 3.0 x16): check the build and the hot
   budget. MTP and the prompt copies can be faster there.
5. The minimum size of the copy buffer of a prompt (a warning now when it is
   less than 32 experts).

## 8. The CPU layouts of float32 and bfloat16 (types 61 and 62)

- Qwen4CPU.KP packs a float32 or bfloat16 matrix of 64 rows or more in
  groups of 16 rows (only in memory). For each column, a group holds the
  values of its 16 rows. A lane is a row, x is float32, and each column is one fma for each
  token (kq_x16f_body).
  - At most 4 tokens: a task has 4 groups, so the sums do not wait on each
    other. With too few groups for the threads, a task has 1 group.
  - More tokens: a task has 4 groups on 16 tokens, with loops of fixed
    counts. A variable count made the compiler keep the sums in memory
    (57 ms, not 28 ms, for 10240 x 2560 on 512 tokens).
- A matrix of fewer than 64 rows stays in rows. Its tasks are a row and 16
  tokens, so the tokens take the threads (4 x 10240 on 512 tokens: 0.33 ms,
  not 1.07 ms).
- A token alone and in a group gives the same bits. The results agree with
  float64 to 3e-6.
- Memory: the packed copies stay in RAM. In dense bf16 mode that is all the
  dense matrices and the head (about 8 GB); the file pages of the
  originals can go.
- NVX (the experts): tokens in batches of 16 (not 8). The gain is 1.5% at
  512 tokens and 7% at 2048: the experts of a prompt are near the rate of
  the memory.
- The results (the same profile program):

      dense q8, a prompt of 512      the float32 matrices 0.93 s -> 0.31 s
      dense bf16, a prompt of 512    15.8 s -> 9.9 s (the dense products 10.7 s -> 4.5 s)
      decode steps                   no change (q8 124 ms; bf16 about 185 ms, 5% of noise)

The 8-bit products (KQ_Q8X16), a step or a verify group:

- A record took about 13 us more than its work: a malloc and 4 barriers
  for x + 128. Now each thread makes x + 128 on its stack (x of at most
  96 KB), and the record has one barrier: about 3 us.
- The tokens go in batches of 8, 4, 2, and 1, with counts fixed at compile
  time. With a variable count, 4 tokens on 10240 x 320 took 132 us, not 64
  us. A token has the same operations in all the batches: the same bits
  (check_qwen4_mtp: the verify group equals the steps).
- A decode step (the profile): the dense products 75.3 ms -> 71.6 ms, the
  step 124 ms -> 119 ms. 320 x 10240: 8.5 ms -> 8.3 ms; 10240 x 320: 7.2 ms
  -> 6.3 ms.
- check_qwen4_mtp --mtp "" takes the MTP layer of the GGUF file.
