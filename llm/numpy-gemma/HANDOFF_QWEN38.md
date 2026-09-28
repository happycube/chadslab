# Handoff: Qwen3.8-Flash-Next in this runtime (2026-09-28)

This file gives the state of the work on Qwen3.8-Flash-Next (qwen4exp). It
tells what works, the numbers, the files, the open work, and the next steps.
QWEN38_PLAN.md has the full plan and the history of each phase.

## 1. The state now

- The work is on master in the repo chadslab. Do not push. Commit with the
  lines of the session at the end (see the git log).
- WORK THAT IS NOT COMMITTED (section 4): the new NVFP4 layout (type 51).
  The GGUF file in models2 already has type 51. The code at HEAD reads type
  50 only. Thus, with HEAD, the file does not load. Commit the work of
  section 4, or make the file again with the code of HEAD.
- Do not stop serve.py (port 8080). The user runs other programs on this
  machine; the numbers change with that load.
- Do not use git stash in this repo. It took the changes of the user in
  notebooks/.
- The user keeps notebooks/btrees-chatgpt4.ipynb changed; do not commit it.

## 2. The models and the files on disk

    models/Qwen3.8-Flash-Next-NVFP4/          the checkpoint of NVIDIA ModelOpt (124 GiB)
    models2/Qwen3.8-Flash-Next-NVFP4-GGUF/    the GGUF of this runtime (133 GB) and tokenizer.json
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
  for the CPU and the GPU (Qwen4CPU.K). NP_GEMMA_DENSE=auto|bf16|q8; auto
  takes q8 with a GPU of little free memory (this machine).
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

## 4. The work that is not committed: the NVFP4 layout for the GPU

The changes: np_gemma/csrc/kquants.c, gpu.cu, cops.py, gguf.py,
st_qwen4.py, scripts/convert_nvfp4_gguf.py.

- Type 51 in place of type 50. A row holds all the codes first (16 bytes
  for each block of 32 values). Then it holds all the E4M3 scales (2 for
  each block) and the float32 scale of the matrix. Zeros fill the row to a
  multiple of 16 bytes. Thus the codes start at a multiple of 16 bytes.
- The GPU tensor cores load a fragment with one aligned 32-bit load. The
  function kt_e2m1x4 changes 4 codes to int8 with prmt (a table of 8 bytes)
  and a sign mask (as the dequantization of Marlin). Before, it was a loop
  of bytes.
- The CPU kernels and kq_nv4_pack use the new layout. The helpers are
  kq_nv4_codes, kq_nv4_scales and kq_nv4_g.
- The GGUF in models2 was made again with type 51 (201 s).
- Checked:
  - The rows of the pack equal a NumPy decode of the checkpoint.
  - The GPU products (t = 1, 4, 32) agree with the exact values to 1e-6.
  - The CPU on 4 layers gives 100% of the top tokens of NumPy. A group
    gives the same bits as the steps.
  - The tensor cores on NVFP4 (8 layers) against the float32 tiles: max
    rel 1.5e-2 (prompt), 3.7e-2 (step), the same top token. With the
    tensor cores, a prompt of 8 layers takes 0.79 s (0.94 s without).
- The speed did not change: the experts of a mixed group of 1024 rows take
  415 ms (408 ms before). The group takes 2.67 s. Of this, FETCH_WAIT
  takes 1.75 s and CPU_WAIT takes 1.41 s. Thus the copies and the CPU set
  the time, not the GPU kernels.
- The change is correct, but it gives no speed now. It can help when the
  copies are faster (the 3090). Commit it, and put type 51 in
  QWEN38_PLAN.md. The alternative is to revert it and make the GGUF again
  with the code of HEAD (201 s).

## 5. The numbers

The machine: Xeon W-2295 (18 cores), 188 GB; RTX 5060 Ti with about 8 GB
free, PCIe Gen3 x8.

The GGUF of this runtime, llama-bench method (scripts/bench_qwen4.py):

    GPU (0.5 GB hot experts)   pp512 248   pp2048 304   pp4096 385   tg128 19.5   tg512 19.7 tok/s
    CPU (dense q8)             tg128 6.9 tok/s; a prompt of 512: 68 tok/s (warm)
    llama.cpp CPU              pp512 27.9   tg128 5.0
    llama.cpp GPU (-ncmoe 48)  pp512 101    pp2048 101  tg128 18.3

- The int8 x of the GPU steps: decode from 16.1 to 19.1 tok/s (one run).
- KQ_Q8X16 on the CPU: the dense products of a prompt of 512 from 5.5 s to
  2.55 s (1.19 T multiply-adds/s against 0.38).
- The limits now:
  - The GPU decode waits for the CPU. With 0.5 GB, only about 3 experts of
    each layer are hot; the CPU computes the cold experts.
  - The GPU prompt waits for the copies (6.8 GB/s over PCIe).
  - The CPU prompt is limited by compute. Of 7.6 s, the experts take 2.8 s
    and the DeltaNet takes 1.0 s.

## 6. How to run the checks

    G=models2/Qwen3.8-Flash-Next-NVFP4-GGUF
    python scripts/check_qwen4_st.py --path $G/Qwen3.8-Flash-Next-NVFP4-bf16.gguf \
        --tok $G/tokenizer.json --layers 4 --gpu --hot-gb 0.5
    python scripts/bench_qwen4.py -m $G/Qwen3.8-Flash-Next-NVFP4-bf16.gguf --backend gpu \
        --hot-gb 0.5 -p 512,2048 -n 128 -r 2
    NP_GEMMA_DENSE=q8 python scripts/bench_qwen4.py -m $G/Qwen3.8-Flash-Next-NVFP4-bf16.gguf \
        -p 512 -n 64 -r 2

The switches for a comparison:

    NP_GEMMA_GPU_KQTC=0     no tensor cores
    NP_GEMMA_GPU_I8X=0      float32 x in steps
    NP_GEMMA_X16=0          no Q8X16 on the CPU
    NP_GEMMA_GPU_ATTN_MT=0  the attention of each query
    NP_GEMMA_GPU_MIX=0      no mixed groups

## 7. The next steps

1. Finish section 4: commit it (or revert it).
2. The CPU experts of a prompt take 2.8 s of 7.6 s. Use the row interleave
   of KQ_Q8X16 for NVFP4 (16 rows, a lane for each row).
3. The GPU decode of dense matrices with short rows (hc_*_up: 320 values):
   a lane for each row. A GPU kernel that reads KQ_Q8X16 can be a test.
4. Memory: in q8 mode, the Q8_0 copy stays next to KQ_Q8X16 (3.9 GB more);
   drop it when no GPU uses it.
5. The 3090 (sm_86, 24 GB, PCIe 3.0 x16): check the build and the hot
   budget. MTP and the prompt copies can be faster there.
6. The minimum size of the copy buffer of a prompt (a warning now when it is
   less than 32 experts).
