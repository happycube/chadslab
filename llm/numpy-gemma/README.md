# numpy-gemma — a NumPy-only Gemma 4 12B runtime

This project runs the model google/gemma-4-12B-it-qat-q4_0-unquantized.
It uses NumPy only. It does not use PyTorch. It does not use transformers.
The project is the Phase 1 baseline of the learning plan in
../Gemma LLM Runtime Learning Plan.md.

It also runs google/gemma-4-E4B-it-qat-mobile-ct, the 4.5B effective dense
model. That model comes from a compressed-tensors SafeTensors file or from the
Q4_0 GGUF file of the same model. It adds Per-Layer Embeddings. Transformers
appears only in the check scripts, where it builds the reference.

It also runs Qwen3.8-Flash-Next (125B, 512 experts) from the NVFP4 checkpoint
of NVIDIA, on the CPU or on a small GPU. See the section
"Qwen3.8-Flash-Next".

A C kernel reads the packed 4-bit and 2-bit weights in place. Thus the file
layout is the runtime layout, and the model builds no float32 copy of a
weight. See the section "Gemma 4 E4B".

This file follows ASD-STE100, Simplified Technical English. A sentence in a
description has 25 words or less. A sentence in an instruction has 20 words or
less.

`scripts/check_ste100.py` checks a mechanical subset of these rules on
any Markdown file. It uses only the Python standard library. Use `--ing-nouns`
to add technical terms for a subject area. It does not check the full standard.

Run it on this file with `python3 scripts/check_ste100.py README.md --ing-nouns scripts/ste100_ing_nouns.txt`.

The commands, the tables, and the quoted model output keep their own words.

Start with [LEARNING.md](LEARNING.md) to follow the dense 12B code path. The
overview page uses the 26B model. Its layer count and dimensions do not describe
the 12B model.

The runtime does five tasks:
1. Read the SafeTensors weights. Convert bfloat16 data to float32 or int8 data.
2. Run the 48 decoder layers.
3. Keep a key and value cache. Generate tokens one at a time.
4. Change text into token ids. Change token ids into text.
5. Keep the weights in memory. Then later passes are fast.

## Files

    numpy-gemma/
    ├── LEARNING.md         Follow the dense 12B path. Check each step.
    ├── WEIGHT_STRUCTURE.md  Four negative results on the linear-algebraic structure of the weights.
    ├── how-llms-work.html  Walk through the eight phases of inference, for a newcomer.
    ├── pipeline.html      The pipeline, with the performance, memory, and quality tradeoffs.
    ├── np_gemma/
    │   ├── st.py          Read SafeTensors files. Use mmap. Convert bf16 to f32.
    │   ├── config.py      Read the configuration. Make a plan for each layer.
    │   ├── ct.py          Read a compressed-tensors file. Decode a packed weight.
    │   ├── e4b.py         Run the Gemma 4 E4B model. The per-layer embeddings.
    │   ├── ops.py         Give rms_norm, linear, gelu_tanh, softmax, softcap.
    │   ├── rope.py        Make the default RoPE and the proportional RoPE.
    │   ├── model.py       Give KVCache and Model. Run the forward pass.
    │   ├── tokenizer.py   Give the BPE tokenizer and the chat template.
    │   ├── gguf.py        Read GGUF files. Map the names. Give the int4 data.
    │   ├── numba_ops.py   Give the Numba JIT kernels. Optional.
    │   ├── cops.py        Build and load the C kernels with ctypes. Optional.
    │   ├── weight_cache.py  Store the converted weights on the disk.
    │   ├── sampling.py    Give the token sampler. Temperature, top_k, top_p.
    │   ├── chat.py        Render the Gemma 4 chat template. Read the output.
    │   ├── chat_template.jinja  The canonical Google Gemma 4 chat template.
    │   ├── server.py      Give the OpenAI compatible HTTP server.
    │   └── csrc/
    │       └── bf16_linear.c  Give the fused bfloat16, int8, and int4 C kernels,
    │                          and the packed 4-bit and 2-bit kernel for E4B.
    └── scripts/
        ├── check_trace.py      Compare each intermediate with a HF trace.
        ├── check_cache.py      Compare the KV cache with a batch prefill.
        ├── check_tokenizer.py  Compare the tokenizer with AutoTokenizer.
        ├── generate.py         Generate tokens from token ids.
        ├── chat.py             Generate text from a prompt.
        ├── session.py          Load one time. Then answer many prompts.
        ├── gen_ids.py          Write greedy token ids for a HF comparison.
        ├── gguf_generate.py    Run a GGUF model and generate text.
        ├── hf_gguf_reference.py  Load GGUF weights into the HF model. Compare.
        ├── bench_numba.py      Compare the NumPy path and the Numba path.
        ├── bench_kernels.py    Compare the NumPy, Numba, and C paths.
        ├── profile_token.py    Time the parts of one decode step.
        ├── profile_parts.py    The time of each operation of a step in parts, one team, or one node.
        ├── profile_prefill.py  The time of each function of a prompt pass, and the traffic of each socket.
        ├── bench_q4x_gemm.py   The products of a prompt (KQ_Q4X, int8 x) alone: the rate, and a check of the bits.
        ├── profile_int8.py     Time each int8 matrix. Show the bandwidth.
        ├── bench_decode.py     Time each decode step. Show the warm-up.
        ├── bench_ram_cache.py  Compare the memory map and the local memory.
        ├── bench_int8_stream.py  Measure the int8 kernel for each stream size.
        ├── bench_prefill.py    Measure the int8 and int4 prompt GEMM.
        ├── bench_threads.py    Measure the kernel speed against the thread count.
        ├── serve.py            Run the OpenAI compatible server.
        ├── check_server.py     Check the server with a fake model.
        ├── check_server_live.py  Check the server with a real model.
        ├── check_chat_template.py  Compare chat.py with the Jinja template.
        ├── check_int4_q8.py    Check the int4 kernel with int8 activations.
        ├── bench_int4_q8.py    Compare the float and the int8 int4 kernels.
        ├── measure_prefill_glue.py  Split a prefill into the C kernels and the glue.
        ├── check_slide.py      Check the sliding window key slice.
        ├── check_flash_c.py    Check the C flash attention against the NumPy reference.
        ├── check_kv_window.py  Check the sliding window key cache bound.
        ├── bench_flash_kernel.py  Compare the flash attention versions with the batched matmul.
        ├── bench_flash_model.py   Compare flash attention with the plain path on a model.
        ├── e4b_inspect.py      Show the tensor layout of the E4B checkpoint.
        ├── e4b_trace.py        Compare the E4B model with a HF reference, layer by layer.
        ├── e4b_generate.py     Generate E4B text. Check the ids against HF.
        ├── check_ct.py         Check the compressed-tensors reader against the library.
        ├── check_ct_kernel.py  Check the packed 4-bit and 2-bit C kernel, every column.
        ├── check_ste100.py     Check any Markdown file against a rule subset.
        ├── ste100_ing_nouns.txt  Allow project-specific technical nouns.
        ├── bench_ct_kernel.py  Compare the packed kernel with the float32 multiply.
        ├── profile_e4b.py      Split one decode step. Find why a token is slow.
        ├── profile_e4b_gguf.py Split a decode step of a GGUF file. Show each group.
        ├── profile_e4b_prefill.py Split a prompt pass. Show the glue and the attention.
        ├── bench_gguf_models.py  Measure pp512 and tg128 in the form of llama-bench.
        ├── bench_e4b_mmap.py   Compare resident, streamed, and packed weights.
        ├── check_e4b_gguf.py   Check the GGUF E4B model against a HF reference.
        ├── check_gelu.py       Check the GELU kernel against a float64 reference.
        ├── peak.c              Measure the peak AVX-512 speed.
        └── membw.c             Measure the memory bandwidth.

## Start here

### Download model files

Google hosts the model files on Hugging Face:

* [Gemma 4 12B QAT, unquantized](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized) — use this as `SNAP`.
* [Gemma 4 12B QAT, w4a16](https://huggingface.co/google/gemma-4-12B-it-qat-w4a16-ct) — use this as `SNAP4`.
* [Gemma 4 E4B QAT, mobile compressed tensors](https://huggingface.co/google/gemma-4-E4B-it-qat-mobile-ct) — use this as `SNAP4B` for E4B.

Hugging Face requires approval for Gemma downloads. Sign in, open each model
page that you need, and accept its terms. Then run the commands below from the
`numpy-gemma` directory. The adjacent project setup installs the Hugging Face
client. See its [setup guide](../gemma4-12b-qat-pytorch/README.md) if its virtual
environment is not ready.

    cd ../gemma4-12b-qat-pytorch
    source .venv/bin/activate
    hf auth login
    python scripts/download_model.py
    python scripts/download_model.py --model google/gemma-4-12B-it-qat-w4a16-ct
    python scripts/download_model.py --model google/gemma-4-E4B-it-qat-mobile-ct

The first command downloads the 12B unquantized QAT checkpoint. Run the other
commands only when you need those models. Each command prints its snapshot path.
Use that full path for `SNAP`, `SNAP4`, or `SNAP4B` below. The files are large;
the 12B unquantized checkpoint is about 24 GB.

Set the paths one time:

    cd numpy-gemma
    PY=../gemma4-12b-qat-pytorch/.venv/bin/python
    SNAP=../gemma4-12b-qat-pytorch/.cache/huggingface/hub/models--google--gemma-4-12B-it-qat-q4_0-unquantized/snapshots/b6ed86275a6a5735884e208bfed95b445a684ca2
    SNAP4=../gemma4-12b-qat-pytorch/.cache/huggingface/hub/models--google--gemma-4-12B-it-qat-w4a16-ct/snapshots/1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee

SNAP is the unquantized checkpoint. SNAP4 is the 4-bit checkpoint from the
quantization-aware training. Use the full path. The command "find" can give the
wrong model, because the cache holds two models.
    The snapshot revisions above match the recorded runs. Use the paths printed by
    the downloader if your cache has a different revision.

Set the thread values before the first command:

    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=6

The default for OMP_NUM_THREADS is one thread for each physical core. The code
also obeys the CPUs that the process can use. Set the variable only to override
the default, for example OMP_NUM_THREADS=18 on an 18-core machine.

The attention uses small matrix products. One BLAS thread is faster than many,
because the many threads fight the OpenMP threads of the int8 kernel. A test at
a context of 1024 tokens gave 0.36 s for the attention with one thread and
1.24 s with six threads.

Use PYTHONPATH=. for each command. The scripts import the np_gemma package from
the project root.

The first int8 load writes the weight cache. This step takes about 400 s. A
later load reads the cache and takes about 5 s. The cache is in
~/.cache/np_gemma/weights.

Test the tokenizer. This step is fast. It does not load the weights.

    PYTHONPATH=. $PY scripts/check_tokenizer.py --snapshot "$SNAP"

Run one prompt in a resident session. This step loads the weights one time.
Then it answers the prompts.

    PYTHONPATH=. $PY scripts/session.py --snapshot "$SNAP" --dtype int8 --max-new-tokens 24 --prompts "Count from 1 to 10, separated by commas."

Run the session without --prompts. The session then reads prompts from the
keyboard. Type :reset to clear the history. Type :q to stop.

    PYTHONPATH=. $PY scripts/session.py --snapshot "$SNAP" --dtype int8

A chat keeps the key and value cache between the turns. The class Session
finds the common prefix of the new turn and the cache. It runs the forward
pass only for the new tokens. Thus a chat does not read the history again.

The class started at 24 of 48 tokens for the second turn of a two turn test.
Set the cache size with --max-len. Use --no-history for independent prompts.

Run the 4-bit mode with the w4a16 checkpoint. This mode gives 48 of 48 tokens
equal to the reference.

    PYTHONPATH=. $PY scripts/session.py --snapshot "$SNAP4" --dtype int4 --max-new-tokens 24 --prompts "The capital of France is"

Change the weight format with --dtype. The default is f32. The choices are f32,
bf16, int8, and int4.

    PYTHONPATH=. $PY scripts/session.py --snapshot "$SNAP" --dtype bf16 --prompts "Hello"

The variable OMP_NUM_THREADS gives the thread count. The default is one thread
for each physical core. The variable OMP_WAIT_POLICY=ACTIVE keeps the
threads awake between the kernel calls. This value gives a better median time
on a loaded machine.

### Measure the speed

Show the bytes and the bandwidth of each int8 matrix:

    PYTHONPATH=. $PY scripts/profile_int8.py --snapshot "$SNAP" --repeats 5

Show the time of each decode step:

    PYTHONPATH=. $PY scripts/bench_decode.py --snapshot "$SNAP" --dtype int8 --steps 16

Compare the memory map cache and the local memory cache:

    PYTHONPATH=. $PY scripts/bench_ram_cache.py --snapshot "$SNAP" --steps 12

### Check against the reference

Write the greedy token ids for the two test prompts:

    PYTHONPATH=. $PY scripts/gen_ids.py --snapshot "$SNAP" --dtype int8 --max-new-tokens 24 --out gen_np_int8.json --prompts "Count from 1 to 10, separated by commas." "Name three primary colors."

Compare the file gen_np_int8.json with gen_hf.json. The ids must be equal.

### Environment variables

    variable               default                 task
    OPENBLAS_NUM_THREADS   1                       One BLAS thread for the attention. Many threads fight the int8 kernel. Set it to 1 for the E4B int4 mode as well: that mode is about eight times slower with a BLAS pool.
    OMP_NUM_THREADS        physical cores          The thread count of the int4 kernel. Set it to override the default.
    OMP_WAIT_POLICY        system                  ACTIVE keeps the threads awake. The median time is better under load.
    OMP_PLACES             cores                   The places of the OpenMP threads. The package sets cores when the variable is not set.
    OMP_PROC_BIND          close                   close, with OMP_PLACES=cores, keeps each thread on one core. A step in parts (NP_GEMMA_PARTS) needs it. For a step of one program the change is small (see SPLIT_PLAN.md). The package sets close when the variable is not set. Set false to turn the binding off.
    NP_GEMMA_ATTN          1                       1 uses the fused attention over the quantized cache, with a float query. 0 runs the float attention of Python over dequantized rows (the cache keeps no float rows). The step programs read the quantized rows in both cases.
    NP_GEMMA_KV_INT8       0                       The form of the cache (KVCache kv=), which keeps only quantized rows with a float32 scale for each group of 32 values, never float32 rows. 0: int16 (a scale of max |x| / 32767). 1: int8 (max |x| / 127). v: int16 keys and int8 values (kv="k16v8"). The readers of float rows (the prompt attention on the CPU) get dequantized rows. On the CPU and the GPU; the GPU drafter of MTP reads each form. The 26B on AVX2 (6 threads) at 4300 tokens: 17.73 (int8) in place of 16.19 tok/s. The 12B on the GPU at 32768 tokens: 44.35 (int8), 43.74 (v), 43.41 (int16) tok/s. KL of the decode of the 12B against float x (160 steps): int16 6.8e-4 (the int8 x), v 2.0e-3, int8 2.1e-3; the NLL of the text does not move (within 0.01). NP_GEMMA_PARTS takes only int16. The server takes --kv-attn int8 or k16v8.
    NP_GEMMA_QWEN_KV       int8                    The cache of the Qwen models: int8 (int8 keys and values), int16, k16v8 (int16 keys, int8 values), tq6 (TurboQuant, 6 bits and a norm for each 32 values: 22% smaller than int8, for very long contexts; np_gemma/tq6.py), or f32. The first full attention layer keeps the form of NP_GEMMA_QWEN_KV_FIRST: f32 (the default), int16, int8, or "same" (the form of the other layers). CPU and GPU, QSA of Qwen3.8 too (QWEN_PLAN.md).
    NP_GEMMA_FUSED_QKV     1                       1 gives the query, the key, and the value their norm in one call, and the query and the key their rope in one call. 0 gives each tensor its own call.
    NP_GEMMA_CACHE_RAM     0                       1 copies the cache into local memory with large pages.
    NP_GEMMA_CACHE         ~/.cache/np_gemma/weights  The cache directory.
    NP_GEMMA_ARCH          auto                    avx2 or avx512 forces one C library.
    NP_GEMMA_KERNEL        auto                    c, numba, or numpy forces one kernel path.
    NP_GEMMA_INT8_INT      0                       1 uses the integer int8 kernel. That kernel is less accurate.
    NP_GEMMA_PREFILL_CHUNK 256                     The prompt pass uses blocks of this many tokens.
    NP_GEMMA_SLIDE         1                       1 drops the keys that no query in the block can see. 0 keeps every key.
    NP_GEMMA_FLASH         1                       1 runs the C flash attention kernel. ref runs the NumPy reference. 0 uses the batched matmul.
    NP_GEMMA_ATTN_IMPL     auto                    c, avx2, or avx512 forces one version of the flash kernel.
    NP_GEMMA_FLASH         1                       1 sends a prompt of more than one token to the C flash kernel. "slide" uses it for a sliding layer only.
    NP_GEMMA_INT4_Q8       1 (16 for the 26B)      The activations of the int4 products of a prompt pass. 1 uses int8 for every product, as llama.cpp does; the default of a dense model. 16 uses int16 (a scale for each 32 values): in the prompt program every product (KQ_LINEAR16, kq_q4x_gemm16 with vpdpwssd; KQ_MOE with int16 rows), in the Python path float32 for the dense matrices and int16 for the experts. It gives the NLL of float32, and is the default of a model with experts when the library has the int16 kernels (VNNI, and AVX2 with vpmaddwd): the prompt of the 26B takes about 1.1 times the time of int8 on VNNI (the 12B about 2 times), 1.45 times on AVX2 (6 cores of the Xeon at 2.4 GHz: pp512 64.2 -> 44.3 tok/s; the KL of 40 decode steps after a prompt of 64 tokens against the NumPy decode 0.22 -> 0.052). 0 uses float32 for every product.
    NP_GEMMA_Q4X           1                       1 keeps a copy of each int4 matrix in groups of 16 rows (KQ_Q4X) for the CPU: the prompt, the decode, the parts, and the cold experts of the GPU. 0 turns it off.
    NP_GEMMA_Q4X_INT8_DECODE AVX2 1, else 0     The int8 x of the KQ_Q4X experts of a decode step (KQ_QUANT, then KQ_MOE). 1 or 0 sets it on any build; VNNI with 1: the experts of the 26B (6 cores, 2.4 GHz) 27.0 -> 15.1 ms a token.
    NP_GEMMA_DECODE_X16    1                       1 (AVX2 with the int8 decode): int16 x in place of int8 for the KQ_Q4X products of a step and of a verify group (the dense matrices, the experts, the head), as the prompt with int16 x; one token takes two int8 planes of x (kq_q4x_rows_p16), a group the unpacked codes (kq_q4x_tile16): the same bits. The 26B, 6 cores of the Xeon at 2.4 GHz: tg64 22.4 -> 19.2 tok/s, the KL of 40 steps after a prompt of 64 tokens against the NumPy decode 0.052 -> 0.0105 (the float decode of VNNI: 0.0135), the top token 35 -> 39 of 40. 0 keeps int8 x.
    NP_GEMMA_INT4_Q8_TOKENS 2                       The smallest token count for the int8 tile. A lower value is slower for one token.
    NP_GEMMA_INT4_MULTI4   1                       1 runs the query, key, and value in one call, and the gate with the up projection. 0 gives one call for each matrix.
    NP_GEMMA_INT4_Q8_GEMV  0                       1 uses the int8 activation for the matrices of one token, on a machine with VNNI. It is not faster. See "What llama.cpp does differently".
    NP_GEMMA_FUSED_STEP    1                       1 uses the fused entry points of the C library for a decode step. 0 gives one call for each kernel, which is slower by about 4 per cent.
    NP_GEMMA_E4B_BF16      1                       1 keeps a bfloat16 copy of a large weight that the quantization did not touch. 0 uses the float32 BLAS path.
    NP_GEMMA_E4B_BF16_MIN  1048576                 The smallest value count for the bfloat16 copy. A smaller matrix keeps the float32 path.
    NP_GEMMA_PROGRAM       1                       1 runs a decode step of one token as one program in C (np_gemma/program.py). 0 uses the Python loop over the layers. The two give the same bits.
    NP_GEMMA_F32_ATTN      c                       c uses the C kernel for the attention of one query over the float cache. numpy uses the batched matmul. The program needs c.
    NP_GEMMA_MT            1                       1 uses the small-group kernels for 2 to 16 tokens, the MTP verify step. 0 uses the prompt kernels.
    NP_GEMMA_MTP           1                       0 turns the MTP drafter off. With NP_GEMMA_GPU=1 the default is 0. See MTP_PLAN.md.
    NP_GEMMA_MTP_PMIN      0                       The drafter stops when its best token has a lower probability than this value. 0 turns the test off.
    NP_GEMMA_PARTS         1                       2 or more runs a decode step of one token as that many programs, each in its own team of threads (np_gemma/parts.py, SPLIT_PLAN.md). This is for a machine with NUMA. The result has the same bits.
    NP_GEMMA_PART_TEAM     0                       The thread count of each part. 0 divides OMP_NUM_THREADS by the count of parts.
    NP_GEMMA_NUMA          1                       With NP_GEMMA_PARTS on a machine with two or more NUMA nodes, each part reads copies of its rows of the weights in the memory of the node of its team (np_gemma/numa.py, mbind). 0 gives the views of the model. The bits are the same.
    NP_GEMMA_PART_HEAD     1                       With NP_GEMMA_PARTS, the parts also run the output head (a Q6_K head, or a Q4_0 head with its KQ_Q4X copy): each part computes its rows from a copy on its node into one logits buffer on the node of part 0, and the main thread is pinned to that node. Model.logits gives that buffer, with the same bits. 0 runs the head in one team after the step.
    NP_GEMMA_PART_KV       1                       With NP_GEMMA_PARTS and the int16 cache, KVCache(...) gives a parts.PartKVCache: the KV heads of each part in its own buffers, on its node. A decode step of the parts and the attention of a prompt block (PART_PREFILL, in the team of each part) use only the buffers of the part. Other readers gather the heads (a copy). The bits are the same. 0 gives one KVCache.
    NP_GEMMA_PART_BALANCE  8                       The steps that a new program of the parts measures (after 2 to warm up) for the balance: the shares of the rows that make the parts end at the same time, from the time of the records that move with the rows and of those that do not. The program is then compiled again with those shares (when they differ by more than 0.5 per cent). The bits stay those of one part. 0 keeps the same share for each part.
    NP_GEMMA_PART_WEIGHTS  -                       The shares of the rows of the parts, as "0.45,0.55": no measure.
    NP_GEMMA_PROMPT_PROGRAM 1                      A block of a prompt runs as one program (np_gemma/prompt.py): the kernels of the Python path with no Python between them, the same bits for the 12B; with a PartKVCache the attention writes and reads the cache of each part (PART_PREFILL). The 26B takes the router of the steps (ROUTER_MT) and kq_moe for its experts: the bits of the Python path with ops.router_mt. With NP_GEMMA_INT4_Q8=16 (the 26B) the products take int16 x (KQ_QUANT16, KQ_LINEAR16, KQ_MOE act bit 2). A prompt with media takes the program too: the soft rows in x, and the last key of each query (Model._media_limit) in the attention (ATTN_PREFILL_QC, PART_PREFILL). 0 keeps the Python path.
    NP_GEMMA_Q4X_GEMM      1                       The products of a prompt (KQ_Q4X, 4 tokens or more) take the tile kernel kq_q4x_gemm (2 groups of 16 rows and 6 tokens in registers, the codes unpacked once for a block of tokens). The same bits as kq_q4x_rows. 0 keeps kq_q4x_rows.
    NP_GEMMA_PART_PREFETCH 1                       The threads of a part that waits at a barrier of the parts prefetch their share of the next weights of the part (up to 512 KB each). 0 turns it off.
    NP_GEMMA_PART_PAIRED   0                       1: the paired split of a step in parts. The output projection and the down map split by columns (the heads and the gate rows of each part), and the parts add their sums: two barriers a layer in place of four. The order of the sums changes, so the bits are not those of one part (the 12B: 5.5e-4 of the hidden state, the same top token in 16 of 16 steps).
    NP_GEMMA_GPU           0                       1 runs the decode steps, the prompt pass, and the output head on a CUDA GPU (np_gemma/gpu.py, SPLIT_PLAN.md). The E4B model runs wholly on the GPU (decode only). The 26B model keeps its cold experts on the CPU. It needs nvcc. The first step copies the weights to the GPU.
    NP_GEMMA_GPU_HOT       the 26B counts          A file of expert counts (scripts/expert_use.py). With NP_GEMMA_GPU=1, the GPU holds the most used experts of the 26B and runs them. 0 keeps all the experts on the CPU. The default is np_gemma/data/gemma-4-26B-expert-counts.npz.
    NP_GEMMA_GPU_HOT_GB    free less 6 GB          The GPU memory for the hot experts, in GB.
    NP_GEMMA_GPU_CPU_THREADS half the cores (2+ nodes)  The team of the CPU programs of a GPU program (the cold experts of a step, gemma_run_task), bound spread over the places. Default: OMP_NUM_THREADS / 2 on a machine of two or more NUMA nodes, else OMP_NUM_THREADS. Qwen3.8 on the 2-socket Xeon and the 3090: 26.0 tok/s with 24 spread, 21.3 with all 48 bound close.
    NP_GEMMA_TEAM_BALANCE  1                       1: the team of the CPU programs of a GPU program (gemma_run_task: the cold experts of a step, the CPU part of a prompt group) pinned with the same count of threads on each NUMA node for the same count of allowed CPUs (the places of OpenMP), evenly spaced, the threads of node 0 first (kq_share_range gives the tasks of node 0 to the first half). proc_bind(spread) alone put a team of 24 on taskset -c 0-19,24-43 as 10 on node 0 and 14 on node 1. 0: proc_bind(spread). scripts/cpu_fence.sh keeps the other processes off the CPUs of the model (status, fence, restore): with java, chromium, and a ClickHouse container on all the cores, decode fell to 4-9 tok/s; fenced to 20-23,44-47, 31-32 tok/s.
    NP_GEMMA_GPU_BF16_NT   1                       1: the bfloat16 products of a small group (an MTP verify group) read each weight once for all its tokens (k_kq_linear_bf16, k_kq_multi_bf16), with the bits of the one-token kernel for each token. 0: the loop over the tokens.
    NP_GEMMA_MOE_X16       1                       1: the experts of Qwen3.8 read x and the GELU with the precision of int16: the CPU parts as int16 in each 32 values split in two int8 planes in the VNNI tile (kq_quant_part16, kq_t2_tile: one decode of each weight block), the GPU experts of a prompt group in float32 (NP_GEMMA_GPU_MOE_F32), those of a step in float32 as before. Real text against float32 activations: KL 5.9e-3, the same top token 99.2% (int8: 1.28e-2, 96.9%); plain decode the same rate, MTP 3 drafts 6% slower, the 8K prompt 2-11% slower. 0: int8.
    NP_GEMMA_GPU_MOE_F32   1 with NP_GEMMA_MOE_X16  1: the experts of a prompt group (GP_KQ_GROUP_MOE) in float32 (k_qmoe_gu, k_qmoe_dn), not on the int8 tensor cores. The prompt rate was the same either way.
    NP_GEMMA_GPU_BF12_TC   1                       1: the BF12 dense matrices of a step and of a group of up to 8 tokens on the tensor cores (k_kq_bf12_tc, mma tf32; x as hi + lo tf32): the same time for 1 and for 4 tokens, below bfloat16, and the bits of a step in a verify group. 0: the CUDA-core kernels (k_kq_linear_bf12, k_kq_multi_bf16).
    NP_GEMMA_GPU_SYNC_CHECK 0                      1: no CUDA graphs, and a wait after each record of a GPU program, so that a kernel error names its record (slow; scripts/replay_crash.py --sync). Without it an error names the record queued last (np_gemma.gpu.where): the one at fault is at or before it.
    NP_GEMMA_GPU_BF12_FUSED 1                      1: the BF12 matrices of a prompt group decoded in the tile of k_gemm_bf16_tc. 0: chunks of bfloat16 rows in a scratch, then gg_gemm_bf16 (the same prompt rate).
    NP_GEMMA_DENSE         bf12 for a BF12 file     The dense matrices of a file of convert_q8_gguf.py --dense bf12 stay BF12 (the bfloat16 values in 12.25 bits); q8 or bf16 converts them at the load.
    NP_GEMMA_GPU_STAGE     1                       1: the copies of experts to the GPU (HotCache, the prompt) go through two pinned buffers of 4 MB of each worker (a memcpy, then DMA), not a cudaMemcpyAsync from the pageable map of the file, which held the driver: Qwen3.8 q8 decode 32 -> 37 tok/s (with NP_GEMMA_GPU_COPY_CPU 44.6). 0: the plain call.
    NP_GEMMA_GPU_COPY_CPU  the last CPU of the GPU node  The CPU of the copy workers of the GPU (pthread affinity); by default the last CPU of local_cpulist of the GPU, which the team of the CPU tasks (bound spread) leaves free. -1: no binding (Qwen3.8 MTP 41.0 in place of 58.7 tok/s).
    NP_GEMMA_GPU_KQH_NVX   1                       1: the KQ_NVX hot experts of a step with a block for each group of 16 rows (k_kqh_nvx): 4.8 -> 1.7 ms a step. 0: a warp for each group.
    NP_GEMMA_GPU_ZC        0                       n > 0: the GPU computes the first n cold experts of each layer of a step from a pinned copy of the experts in host memory (about 68 GB, over PCIe), the CPU the others. NP_GEMMA_GPU_ZC_MT: the same for the verify groups. NP_GEMMA_GPU_PIN=1: the copy alone (the CPU and HotCache read it). Slower on the Xeon and the 3090 (plain 44.6 -> 33.7 tok/s): an expert takes 0.25 ms over PCIe 3.0, more than the share of the CPU.
    NP_GEMMA_GPU_HC_FUSE   1                       1: GP_HC_ADD and the GP_HC_NORM of the same H in one kernel (k_hc_add_norm, the same bits), and GP_HC_NORM, GP_HC_ACT, and GP_HC_MIX write the int8 x of the product that follows (qx_fuse past the GP_KQ_QUANT records): Qwen3.8 plain 45.4 -> 46.7 tok/s (q8), 38.3 -> 39.6 (bf16). 0: separate kernels.
    NP_GEMMA_GPU_TOPK_SPLIT 1                      1: GP_ROUTER_TOPK and the GP_HOT_SPLIT of a step in one kernel. 0: two.
    NP_GEMMA_GPU_NUMA_COPY auto                    A copy of the experts of Qwen3.8 on each NUMA node; each thread of the CPU part reads the copy of its node (KQ_MOE mats1). auto: on when the file was staged from a DAX mount (NP_GEMMA_DAX_STAGE: the staged copy is the copy of its node, the other comes from it at about 16 GB/s). 1: also from the map of a file; 0: off. From Optane: plain 52.7 tok/s with both copies, 32.2 with the staged copy alone.
    NP_GEMMA_DAX_STAGE     1                       A GGUF on a DAX mount (persistent memory, Optane in App Direct mode): the tensors go into memory of the process on the node of the module, copied by threads of that node (9-10 GB/s; a thread of the other socket reads the module at 0.4 GB/s), in huge pages. NP_GEMMA_DAX_KEEP (per_layer_token_embd.weight) stays on the module: a step reads a few rows of it. NP_GEMMA_DAX_PLACE=interleave: the copy over all the nodes. Qwen3.8 loads in 36 s (127 s with the plain map). 0: the map.
    NP_GEMMA_GPU_EXPERT_COPY auto                  auto: the experts in memory of the process when the file is on a DAX mount and the reader did not stage it; 1 always; 0 never. NP_GEMMA_GPU_EXPERT_NODE: interleave (the default) or a node.
    NP_GEMMA_CPU_NODE_SHARE 0.5                    The share of node 0 in the tasks of the cold experts of a decode step (the threads of each node take their share). "auto": measured at the start (kq_calib_nodes; it gave 0.62-0.64 on the Xeon and 50.3 tok/s against 51.8 with 0.5); a number sets it.
    NP_GEMMA_GPU_REGISTER  1                       1: page-lock the copy of the experts that the GPU copies from (when it is in memory of the process: staged from a DAX mount, or a copy on each node); the copies of a prompt group are then DMA at the rate of the bus (bf16 8K prompt 564 -> 649 tok/s). The copies of HotCache in the decode stay staged: DMA at the full rate slowed MTP (66.7 -> 58.8 tok/s).
    NP_GEMMA_QWEN_KV       int8 (tq6 with NP_GEMMA_DENSE=bf16)  The form of the cache of Qwen3.8; see NP_GEMMA_QWEN_KV above. tq6 is 18.7 KB a token in place of 21.8 (the first full attention layer stays float32).
    NP_GEMMA_GPU_CTX       0                       The tokens of the largest cache that a QwenGPU will attach (the ctx argument; serve_qwen4.py gives --ctx): the hot experts leave room for its cache. Qwen3.8: 20.6 KB a token and 118 MB of state; bf16 dense with 262144 tokens leaves 42 hot experts in each layer.
    NP_GEMMA_GPU_BF16_TC   1                       The bfloat16 matrices of a prompt group (KQ_LINEAR, more than 16 tokens) on the tensor cores (k_gemm_bf16_tc, float32 sums). 1: x as two bfloat16 planes, hi + lo (about 16 bits; error of a product 1e-5, float32 2e-6; 18-26 TFLOPS); 8: one plane (8 bits, 1.6e-3, 31-53 TFLOPS); 0: float32 products (k_kq_gemm, 12-14 TFLOPS). Qwen3.8 bf16, 8192 tokens: 469 (1), 504 (8), 406 tok/s (0); KL to float32 0.0104, 0.0128 (q8 dense 0.0191; two float32 runs 0.0066).
    NP_GEMMA_GPU_QMOE_SORT_PAR 1                   1: the sort of the pairs of a group MoE on a block of 1024 threads (5 ms -> well under 1 ms a layer of 2048 tokens). 0: one thread.
    NP_GEMMA_CPU_MOE_SMALL 1                       1: the cold experts of a GPU step or verify group (t <= 4) take kq_moe_small_body: the int8 x, the sort, and the rows of the pairs in one single, the act by the thread that ends the last gate/up task of an expert, and the down tasks waiting for their expert (3 barriers in place of 7; the same bits). About +1% on Qwen3.8. 0: KQ_QUANT and kq_moe_body.
    NP_GEMMA_GPU_HOT_DYN   1                       0 keeps the first set of hot experts. 1 lets the set follow the text (HotCache, SPLIT_PLAN.md).
    NP_GEMMA_GPU_HOT_DECAY 0.97                    The decay of the scores of HotCache for each step.
    NP_GEMMA_GPU_HOT_INS   8                       The most experts that HotCache copies to the GPU in each step.
    NP_GEMMA_GPU_HOT_ADMIT 2                       HotCache copies a cold expert to the GPU only from this use on; the first uses run on the CPU.
    NP_GEMMA_GPU_HOT_SEED  0                       The experts that HotCache can change after a prompt pass, from the routers of the prompt. 0 changes none (a test gave no gain).
    NP_GEMMA_GPU_TC        1                       1 runs the int4 products and the attention of a large group (a prompt pass) on the tensor cores, with float16 inputs. 8 gives int8 inputs to the products (the Q8_0 form): about 45% faster for the E4B, and 98.6% of the top tokens agree with float32, against 99.9%. 0 keeps float32 kernels. The 26B uses float32 kernels (with int8 for its int4 products, NP_GEMMA_GPU_I8_MOE) unless NP_GEMMA_GPU_TC_MOE=1.
    NP_GEMMA_GPU_PDL       1                       0 turns off programmatic dependent launch in the CUDA graphs, for a test. See SPLIT_PLAN.md.
    NP_GEMMA_GPU_FLASH     0                       1 selects the old attention kernel of a prompt pass (k_flash_tc), for a test.
    NP_GEMMA_GPU_FLASH512  4                       The query heads in a block of k_flash_qc_h, the attention of a prompt pass for a head of 512 values (the global layers). 4 is the fastest: 475 ms in place of 714 ms for the global layers of the 12B and 8192 tokens. 2 is for a test. 0 keeps k_flash_qc_tc.
    NP_GEMMA_GPU_FLASH256  4                       The tiles of 16 queries in a block of k_flash_qc_h for a head of 256 values (the layers with a window), with 2 query heads. 4: 250 ms in place of 329 ms for the 40 layers of the 12B and 8192 tokens. 0 keeps k_flash_qc_tc.
    NP_GEMMA_GPU_FLASH_FK  32                      The keys of a step of k_flash_qc_h for a head of 256 values: 16 or 32. 32: 235 ms in place of 255 ms for the 40 layers of the 12B with a window and 8192 tokens.
    NP_GEMMA_GPU_CHUNK     2048                    The tokens of a chunk of a prompt pass on the GPU. Each chunk copies the cold experts to the GPU (about 1.6 s for the 26B). With 2048 the copy is hidden behind the other work of the layers.
    NP_GEMMA_GPU_PREFILL_MIN 128                   A shorter part of a prompt runs on the GPU in groups of 16 tokens, with the experts on the CPU.
    NP_GEMMA_GPU_MIX       1                       1 runs the prompt of the 26B in mixed groups: each layer copies only some cold experts to the GPU, and the CPU computes the others. 0 copies all the cold experts.
    NP_GEMMA_GPU_MIX_PRE   auto                    The cold experts that each layer of a mixed group copies. auto uses the model of the time in ModelGPU.plan_mix. A number gives the count; -1 copies all.
    NP_GEMMA_GPU_MIX_CAL   1                       1 measures the model of the time of the mixed groups again after each group (ModelGPU.calibrate_mix). 0 keeps the first values.
    NP_GEMMA_GPU_PREFETCH  1                       A mixed group copies, during the layer before, the experts each layer will likely copy (those the plan of its last group would copy, the most tokens first, not hot now) into one of two sets of pool blocks; the plan takes them as on the GPU and copies the rest, and the GPU runs the hot, the prefetched, and the copied experts in turn. Qwen3.8 8K real text: copy waits 1.73 -> 0.63 s a group, 600 -> 618 tok/s (99% of the prefetched experts used). 0: no prefetch.
    NP_GEMMA_GPU_PREFETCH_FRAC 1.0                 The share of the experts of the last group to prefetch (0.7: 604 tok/s, 1.3: 595).
    NP_GEMMA_GPU_PREFETCH_COST 1.0                 The cost of a prefetched copy for the size of the prediction, as a share of the calibrated cost of a copy (moe.c stats[7], desc[23]). Qwen3.8 8K real text: 1.0 is the best measured; 0.6 and 0.35 (more prefetched) 2-15% slower, 1.5 and 2.0 within the noise.
    NP_GEMMA_GPU_SHARE     1                       The KV caches of every model (gpumm.DeviceCache, position-major) lend GpuMem the rows past those a run needs and a margin; the pool of HotCache places segments there, which go when the cache takes the rows back (anything still locked after the runs is an AssertionError). Qwen3.8 at 256K: 3.4 GB more experts (2632 on the GPU), decode 166 cold experts a step (202). 0 lends nothing.
    NP_GEMMA_GPU_SHARE_MARGIN 4096                 The positions past those of a run that a cache keeps; the lent memory changes only when a run passes them.
    NP_GEMMA_GPU_C_RESERVE 0.8e9                   The free memory that GpuMem keeps for the C side (the code, environment, and graphs of a program, temporary copies): the pool and the other blocks leave it.
    NP_GEMMA_GPU_MIX_THREADS auto                  The team of the CPU part of a mixed group of a prompt (word 3 of its program, gemma_run_task). auto: OMP_NUM_THREADS - 8 on a machine of two or more NUMA nodes (40 of 48), the rest for the GPU runner, the copies, and other processes; 0: the team of a step (NP_GEMMA_GPU_CPU_THREADS). A layer of a real-text group of 2048 rows (Q8_0): 45.3 ms with 24, 38.7 with 40.
    NP_GEMMA_GPU_MIX_NUMA  1                       The CPU part of a mixed group reads the copy of the experts on node 1 (NP_GEMMA_GPU_NODE1_GB) on the threads there, as a step does. A layer of a real-text group, 40 threads: 38.9 -> 31.8 ms. 0: all read the experts' node.
    NP_GEMMA_KQ_TILE2      1                       The tiles of Q8_0 and Q6_K of a prompt (kq_rows4_t2): the scales of 4 rows one time for all the tokens, x + 128 times w from -128 sum(w) (no sign steps), short tiles of 1 to 3 tokens, the next 4 rows prefetched; the same bits as kq_tile4. A layer of a real-text group, 24 threads: 59.4 -> 45.3 ms (Q8_0). 0 keeps kq_tile4.
    NP_GEMMA_KQ_TILE2_MIN  2                       The fewest tokens of an expert for kq_rows4_t2 (fewer: kq_dot4_q8_0, kq_dot1).
    NP_GEMMA_MOE_GPROF     0                       1: the times of the phases of the CPU part of a prompt group (kquants.c kq_moe_gprof): the sort, the copy, gate and up, the act, down, the sum.
    NP_GEMMA_GPU_GDN_PAR   4                       The DeltaNet of a group of 64 tokens or more with no log (a prompt group): the convolution for each token and channel at once, the norms of q and k and the gates before the state, 4 (or 8) threads for each column of the state, then the norm of the output (k_gdn_conv_par, k_gdn_prep, k_gdn_scan, k_gdn_out). Qwen3.8 bf16 8K prompt 757 -> 837 tok/s (1.43 s -> 0.34 s of kernels). 0 keeps k_gdn_heads (one block for each head, 4 us a token).
    NP_GEMMA_GPU_QSA_TC    1                       The attention of the QSA layers of a large group (int8 and TQ6 caches) on the tensor cores (k_attn_qsa_tc): the 12 query heads of a key head are the rows of an mma tile, q as two float16 planes. 2: one plane. 0: k_attn_qsa_mt (float32). Qwen3.8 bf16 8K prompt 860 -> 910 tok/s, KL to float32 0.0120.
    NP_GEMMA_DAX_EXPERTS   auto                    The node of the routed experts when gguf._stage_dax stages a file from a DAX mount: auto takes the module's node if the staged tensors fit 85% of it, else the other node (the Q8_0 experts of Qwen3.8, 131 GB), copied through buffers on the module's node (9.9 GB/s). Or a node.
    NP_GEMMA_GPU_NODE1_GB  auto                    The experts on one node only (they do not fit the other): a copy of the most used experts of each layer on the other node, read by its threads (KQ_MOE slot1). auto: 70% of that node less the tensors staged there (329 of 512 Q8_0 experts of Qwen3.8, 84 GB). 0: none. Qwen3.8 Q8_0 decode 25.4 -> 30.2 tok/s.
    NP_GEMMA_EXPERT_COUNTS (none)                  An .npy of the uses of each expert (layers + 1, experts; HotCache.profile()) that picks the experts of NP_GEMMA_GPU_NODE1_GB. Without it, the experts after the first hot ones.
    NP_GEMMA_GPU_WARM      lend                    The warm experts of HotCache in the blocks of its pool (gpumm.ExpertPool: segments of 8 experts: the first hot experts, the room lent by the image encoder, the rows of the KV cache past those in use (NP_GEMMA_GPU_SHARE), and the free memory; the copies of the mixed groups take blocks of it too, the warm experts of the lowest scores giving theirs). lend: the lent room (serve_qwen4 --mmproj-gpu lend) and the free memory the prompts took, until a program needs it; free: also the free memory at the first decode step; 0: the pool holds only the copies of the prompts.
    NP_GEMMA_MOE_PROF      0                       1: the times of the phases of the CPU part of a decode step (kquants.c kq_moe_prof).
    NP_GEMMA_GPU_MIX_SPLIT 1                       1 splits the GPU experts of a mixed group of Qwen3.8 in two group MoEs: the hot experts and the shared expert run before the wait for the copies (GP_FETCH_WAIT), the copied experts after it (the plan gives their pairs in gidx2), and an add joins the two. 0 runs one group MoE after the wait.
    NP_GEMMA_GPU_MIX_LOG   0                       1 prints the waits, the plan, and the measures of each mixed group.
    NP_GEMMA_GPU_FLAGS     1                       1 makes the decode step one graph: the CPU parts wait on flags in pinned memory (GP_SIGNAL, GP_AWAIT, GP_CPU_TASK). 0 gives the boundary records.
    NP_GEMMA_GPU_FUSED     1                       1 puts the norms and the adds of the end of the attention and of the end of a layer of the 26B step in two kernels (GP_ADD_NORM, GP_FFN_OUT). 0 keeps the separate kernels.
    NP_GEMMA_GPU_ATTN_TC   1                       1 runs the attention of a prompt pass of the 26B on the tensor cores (k_flash_qc_tc, float16 inputs, float32 sums). 0 keeps the float32 kernel.
    NP_GEMMA_GPU_I8_MOE    1                       1 gives int8 x to the int4 dense products of a prompt pass of the 26B (k_gemm_q8), and the rest keeps float32. 0 gives float32 products. With the int16 prompt (Model.prompt_act "16", the default of the 26B) the dense products and the KQ_Q4X experts take the int16 form: x as two int8 planes (q = 128 hi + lo, k_quant_x2) and two int8 products of the tensor cores for each block, exact in int32. The 26B prompt: about 2750 tok/s in place of 3150 (int8).
    NP_GEMMA_GPU_Q4X       1                       1 gives the experts of the 26B to the GPU in groups of 16 rows (KQ_Q4X). 0 gives the old int4 layout. It has no effect when NP_GEMMA_Q4X=0.
    NP_GEMMA_GPU_Q4_I8     1                       1 runs the Q4_0 matrices of a decode step and of an MTP verify group of the E4B and the E2B with int8 x and dp4a (E4B.kq_q4, GP_KQ_LINEAR). A verify group of 3 tokens then costs 1.4 steps, not 1.75. The KL against float32 goes from 0.00002 to 0.0005 (MTP_PLAN.md). 0 keeps float32 x. A prompt pass keeps the int4 records.
    NP_GEMMA_GPU_GROUP_FUSED 1                     1 gives a small group of the 26B (an MTP verify group) the fused norms of the step (GP_ADD_NORM, GP_FFN_OUT). A group of 3 tokens takes 16.6 ms, not 18.4 ms. 0 keeps the separate norms.
    NP_GEMMA_GPU_PREFILL_FUSED 1                   1 gives a prompt pass of a dense model (the 12B) the fused layer of the step (two GP_ADD_NORM in each layer). 0 keeps the separate norms.
    NP_GEMMA_GPU_EMBED     1                       1 makes the token rows of a prompt pass on the GPU from the Q4_0 token table (gg_embed_q4_rows), in place of Model.embed and a copy of the rows. 12B: about 40 ms less for each chunk of 2048 tokens. 0 keeps the host rows.
    NP_GEMMA_GPU_QKV_FUSE  1                       1 runs the norms of q, k, and v and the rope in one kernel (k_qkv_norm_rope). 0 keeps k_qkv_norm and k_rope.
    NP_GEMMA_GPU_Q4_I8_DENSE auto                  int8 x and dp4a for the dense Q4_0 matrices of a step and a small group of the 12B and the 26B. auto: on for the 12B (MTP with 3 drafts: 56.6 to 91.8 tok/s; KL 0.00012 to 0.0004), off for the 26B (KL 0.0001 to 0.0009). 1 or 0 sets it for both (MTP_PLAN.md).
    NP_GEMMA_GPU_MT_ROWS   1                       1 runs the int4 products of a group of 1 to 4 tokens with the lanes and the order of a decode step (k_mt_int4_rows), so a group gives the bits of the steps. 0 selects the old kernel (k_mt_gemv_n), for a test.
    NP_GEMMA_SPARSE        1                       1 samples a row from the candidates of the GPU (the k best logits, gg_topk) when they settle the result. 0 copies the whole row of logits to the host.
    NP_GEMMA_GPU_KV        int16                   The form of the cache of the 26B on the GPU. int16 keeps the int16 copy of the CPU program: about half of the float form. float keeps float32 rows.
    NP_GEMMA_PART_ATTN     heads                   heads gives each part a range of the attention heads. one runs the attention in part 0 only, the first form.

## Weight modes

The command load_all() loads all layers and the embedding table one time.
The model then keeps the data in memory. Later passes do no file input.

    mode               memory     decode speed        notes
    int8 + C           12.1 GB    about 0.35 s/token  Six times less memory.
    int4 + C (w4a16)   9.0 GB     fastest             Eight times less memory. Accurate.
    bf16 + C           23.6 GB    about 1.07 s/token  Three times less memory.
    f32 (default)      70.1 GB    about 2.1 s/token   Uses BLAS.
    bf16 + Numba       23.6 GB    about 3.3 s/token   Three times less memory.
    bf16 + NumPy       23.6 GB    about 17.3 s/token  Three times less memory. Slow.
    no residency       small      about 5-6 min/token Read the file for each token.

A warm 24-token greedy run takes about 10 s on a quiet machine in the int8
mode. The same run took about 30 s when other users loaded the machine. The
time depends on the machine load. The bf16 mode takes about two times longer
than the int8 mode.

The f32 mode keeps the weights as float32 values. It uses the BLAS library for
the multiply.

The bf16 mode keeps the raw bfloat16 values. The int8 mode keeps int8 values
and one scale for each row. Both modes convert the values during each multiply.
Three kernels are available:

1. The C kernel. A small C file gives the multiply. The code compiles the file
one time with cc and then loads the library with ctypes. The ctypes module is part of
Python, so no new package is necessary. The C code uses AVX2, FMA, and OpenMP.
The kernel reads the values, converts them during the multiply, and writes no
float32 block.

The code builds two libraries from the same source. The first library uses an
AVX2 baseline. The second library uses an AVX-512 baseline. The code reads the
CPU features at run time. A CPU with AVX-512 loads the AVX-512 library.

A CPU without AVX-512 loads the AVX2 library. Every target machine gives AVX2.
Set NP_GEMMA_ARCH=avx2 or NP_GEMMA_ARCH=avx512 to force a library. The
bfloat16 kernel reads four output rows in one loop. Thus the loop loads x one
time for four rows.

The int8 kernel reads one row for each token in a decode step. For a prompt
the code uses a tiled GEMM. The code uses the GEMM at 32 tokens or more. Both
targets give the same two prompt tiles.
   * The K-vectorized tile keeps the result of four rows and four tokens in a
     vector. AVX2 uses four rows and two tokens. One instruction converts 16
     weights on AVX-512 and 8 weights on AVX2. Each converted vector serves
     several tokens. The final sum over the columns is horizontal.
   * The multi-level GEMM reads each weight one time. A micro kernel keeps
     ML_MR rows and ML_NR tokens in the registers. The values are 16 and 16 on
     AVX-512 and 8 and 8 on AVX2. An int8 A panel and a float32 B panel stay in
     the cache. A block over the columns then reduces the weight traffic.

The multi-level GEMM is 1.26 times faster at 128 tokens. It is 1.32 times
faster at 256 and 512 tokens than the K-vectorized tile on AVX-512. The code
uses it at 128 tokens or more on AVX-512 and at 64 tokens or more on AVX2. A
shorter prompt keeps the K-vectorized tile. Use cops.set_gemm_ml(False) for a
test and cops.set_gemm_kv(False) for the older token-vectorized tile.

The AVX2 block sizes are ML_KC 64, ML_MC 64, and ML_NC 64. A KC of 64 makes
the B panel 16 KB, so the B panel stays in the 32 KB L1 cache.

A larger KC was about 10 percent slower. The AVX-512 block sizes are 256, 128,
and 128. A long prompt is cut into blocks of 256 tokens. The GEMM is then
always in its fast range. A test of the large matrices at 1024 tokens gave 1.4
times to 2.4 times more speed. Set the block size with NP_GEMMA_PREFILL_CHUNK.

   Speed for one prompt matrix, best of three runs, GFLOP/s at 256 tokens:

       shape                 AVX2    AVX-512
       int8 15360x3840        289       428
       int8 3840x3840         335       490
       int4 15360x3840        308       425
       int4 3840x15360        309       399
       int4 3840x3840         337       432

   The AVX-512 value moves by about 10 percent with the machine load. The AVX2
   values are 68 to 78 percent of the AVX-512 values. Before the AVX2 work the
   int8 value was 120 to 138 GFLOP/s and the int4 path had no GEMM. The AVX2
   path is tuned for a 32 KB L1 cache and six threads. The Intel Core i5-8500
   (Coffee Lake, 6 cores, 2 channels of DDR4) is a target for that path.

   Result for one int8 set of 2.36 GB, best of three runs:

       AVX2 library    19.7 GB/s
       AVX-512 library 27.8 GB/s

   The AVX-512 library is 41 percent faster for this loop.
2. The Numba kernel. A just-in-time kernel does the same task for bfloat16.
3. The NumPy kernel. The code converts a block of output rows to float32. Then
   it multiplies. This kernel writes a float32 block and reads it again. Thus
   it needs more memory bandwidth.

The model selects a kernel in this order: C, then Numba, then NumPy. Set the
environment variable NP_GEMMA_KERNEL to "c", "numba", or "numpy" to select one
kernel. The environment variable NP_GEMMA_NO_NUMBA=1 turns off Numba. The
environment variable NP_GEMMA_BF16_CHUNK sets the number of rows in one NumPy
block. The default is 8192.

The int8 mode quantizes each row with one symmetric scale. The formula is
scale = max(abs(row)) / 127. The int8 mode reads half the bytes of the bf16
mode. The int8 mode is the fastest mode for the full model. The int8 mode has a
small error. The test below shows no difference in the generated token ids.

The C file also gives an integer int8 kernel. The integer kernel quantizes the
activations to int8 as well as the weights. Then it uses integer multiply and
add. The integer kernel is faster for one matrix.

The activation quantization causes a larger error. A test gave 29 of 48 tokens
equal to the reference. The integer kernel is not the default. Set the
environment variable NP_GEMMA_INT8_INT=1 to select it.

## 4-bit weights

The int4 mode packs two values in each byte. A group of 32 values uses 16
bytes. Each group has one scale. The scale is max(abs(group)) / 7. The model
was trained with quantization-aware training on a 4-bit lattice. Thus 4-bit is
the intended format.

The int4 mode has two sources of weights. The accuracy depends on the source.

1. The unquantized checkpoint. The code calculates a new scale for each group
   from the bfloat16 values. The formula is max(abs(group)) / 7. The new scales
   differ from the training scales. This way is not accurate. A test gave 33 of
   48 tokens equal to the reference.
2. The w4a16 checkpoint. The code reads the packed weights and the training
   scales from the file. This way is accurate. A test gave 48 of 48 tokens
   equal to the reference.

The model detects the w4a16 checkpoint. Look for the tensor
"layers.0.mlp.gate_proj.weight_packed". If the tensor is present, the model
uses the training scales for all int4 matrix products.

The int4 mode reads fewer bytes than the int8 mode. Thus the int4 mode is now
the fastest mode. The C kernel converts the nibbles to float32 with SIMD
instructions. The kernel does not write the values to a buffer first. An
earlier kernel did that and was about four times slower.

The int4 kernel reads four weight rows for each x block. The x row uses four
bytes for each value, so x traffic controls the time. The four-row loop gives
32 GB/s to 38 GB/s for one matrix. The one-row loop gave 21 GB/s to 34 GB/s.

### The w4a16 checkpoint

The checkpoint google/gemma-4-12B-it-qat-w4a16-ct is 10.26 GB. It has 1334
tensors. The dtypes are I32 (328 tensors), I64 (328), and BF16 (678). Each
Linear layer has three tensors:

    weight_packed   I32   (rows, cols / 8)   eight 4-bit values in each int32
    weight_scale    BF16  (rows, cols / 32)  one scale for each group of 32
    weight_shape    I64   (2,)               the original shape

The quantization config gives format "pack-quantized", group_size 32,
num_bits 4, symmetric true, observer "memoryless_minmax".

These are the exact scales from the quantization-aware training. The model
reads them and uses them.

The checkpoint does not quantize the embedding table. The model reads the
embedding table as bfloat16. The model reads the scales as float32. Thus the
memory is more than the 5.6 GB of the packed weights. The measured memory is
9.0 GB.

The packed order in the file is different from the order in the C kernel. The
function ops.convert_w4a16 changes the order. In the file, byte i of the int32
holds column 2i in the low nibble and column 2i + 1 in the high nibble. The
value is nibble - 8.

The function writes 32 bytes for each block of 32 values. Byte j holds value j
in the low nibble and value j + 16 in the high nibble. The function also
converts the scales to float32.

Use this command to run the w4a16 checkpoint in the int4 mode:

    PYTHONPATH=. ../gemma4-12b-qat-pytorch/.venv/bin/python scripts/session.py \
        --snapshot <w4a16 snapshot directory> --dtype int4

### Loop experiments that did not help

The int8 loop was tested with three changes:

    change                                    result
    software prefetch of the weight row       slower (7 GB/s against 19 GB/s)
    two rows in one loop, non-temporal hint   slower (12.5 GB/s)
    eight scalar accumulators                 slower

The compiler loop with "#pragma omp simd" was faster than all three. The
hardware prefetcher already reads a sequential stream. A manual loop prevents
the vectorization of the code.

The one change that did help was the AVX-512 baseline library. It changed the
int8 loop from 19.7 GB/s to 27.8 GB/s.

A later change also helped. The function dot_i8_f32 gives the conversion and
the multiply as AVX-512 or AVX2 instructions. It replaces the loop that the
compiler vectorizes. A test on one buffer gave 1.15 ms against 1.30 ms. The new
function is 11 percent faster. The code is in bf16_linear.c.

The four-row loop is also in the code. The loop reads one x block and four
weight rows. Thus the x traffic falls by four times. A test in one library and
one process gave this result:

    shape              one row    four rows   gain
    down 3840x15360    1.045 ms   1.010 ms    1.04x
    o    3840x8192     0.576 ms   0.545 ms    1.06x
    gate 15360x3840    0.963 ms   0.977 ms    0.99x

The gain is small. The weight stream from the memory controls the time, not
the x stream. The full model showed no clear gain. The code stays for the small
gain. Use cops.set_rows4(False) to select the one-row loop.

The token-vectorized int8 GEMM was tested with a tile shape of 16 rows and 32
tokens. The result was 343 GFLOP/s at 32 tokens on AVX-512. A tile of 8 rows
and 32 tokens gave 181 GFLOP/s. That tile is still the fallback when the
K-vectorized tile and the multi-level GEMM are off.

A K-vectorized tile then replaced it on both targets. The new tile keeps the
result of four rows and four tokens in a vector. One instruction converts 16
weights (8 on AVX2), and each converted vector serves several tokens. A test
gave this result:

    shape                 tile 16x32    K-vectorized    gain
    gate 15360x3840       174 GB/s      167 GB/s        0.96x
    down 3840x15360        65 GB/s      207 GB/s        3.20x
    o    3840x8192         47 GB/s      131 GB/s        2.80x

The full model was 1.60 times faster for a prompt of 256 tokens. The error is
also smaller, because the tile sums 16 values (8 on AVX2) in a vector. Use
cops.set_gemm_kv to select the older tile for a test.

The script peak.c measures the limits of the machine. The result was:

    test                        1 thread    6 threads
    pure fused multiply and add    210         602 to 676 GFLOP/s
    convert int8 and add           106         294 to 379 GFLOP/s

The machine was loaded during the test. The load average was 6 to 11 on six
processors. Thus the six threads gave only about half of the value of six
single threads. The K-vectorized tile reached 263 GFLOP/s for mlp.down_proj.
That is 40 percent of the six thread value. On a quiet machine the limit is
about two times larger.

The K-vectorized tile reads the weight block again for each group of four
tokens. At 256 tokens that is 64 passes. For mlp.down_proj the pass reads
3.8 GB of weights and 16 MB of x. The tile gives about 25 GB/s against the
41 GB/s of a plain read. Thus the tile waits on the memory, and a change to the
inner loop cannot help.

The float weight panel removes the int8 convert. But it reads one row block at
a time. The x data is then read again for each row block. The result was 0.41
times the speed for mlp.down_proj at 256 tokens. The default is off.

The token-vectorized tile reads the weights only 8 times at 256 tokens. But
its weight convert costs more. A packed weight layout was built to give both.
The pack puts the 16 weights of one column next to each other. One instruction
then converts all 16 values.

But the pack belongs to one row block, and the x data is read again for each
row block. The result was 0.40 times the speed for mlp.down_proj at 256
tokens. The default is off. Use cops.set_gemm_packed for a test.

A packed copy of all the weights with square blocks of 16 rows and 16 columns
was tested last. One block is 256 bytes and fills four cache lines. The result
depended on the number of tokens:

    shape                 T=64      T=256
    gate 15360x3840       1.23x     0.81x
    down 3840x15360       1.28x     0.33x
    o    3840x8192        2.36x     0.36x

The square block is faster for a short prompt. But the tile uses 32 tokens for
each step, and the x block is then larger than the L2 cache. A block of 16
tokens did not correct the loss at 256 tokens. The chunked prompt pass uses
256 tokens for each step, so the K-vectorized tile stays the default. The
packed copy is off. Set NP_GEMMA_PACKED=1 for a test.

The lesson from all five attempts is the same. The row block on the outside
keeps the weight data in the cache. The token block on the outside keeps the x
data in the cache. The two cannot both be on the outside.

A multi-level GEMM then solved the problem in a different way. It keeps the
weights, the x data, and the output in the cache at three levels at the same
time. It reads each weight one time. The result was 1.26 times the speed at
128 tokens and 1.32 times at 256 and 512 tokens. The plan is in
PREFILL_GEMM_PLAN.md. The code is in gemma_int8_gemm_ml.

Two cache experiments did not help the full model:

    change                                    result
    token block on the outside                the same speed
    K split, one region for each K chunk      0.85 times the speed
    K split, one region and a barrier         0.82 times the speed

The K split helps one wide matrix. mlp.down_proj went from 52 GB/s to 114 GB/s
at 256 tokens. The gate matrix went from 94 GB/s to 114 GB/s. But the model has
328 matrices, and many are small. The barrier and the tile store cost more than
the cache win. The default is off. Use cops.set_gemm_kc for a test.

The int8 mode is memory bound in the full model. The script profile_int8.py
gives the bytes and the bandwidth of each matrix. One matrix gives 40 GB/s to
46 GB/s. A single pass over all the matrices gives 42 GB/s.

A plain read of one large array gives 41 GB/s to 43 GB/s with six threads.
Thus the kernel works at the memory speed of the machine. No bandwidth gap is
left to close.

The machine had a heavy load from other users during some tests. The same code
then gave 17 GB/s to 24 GB/s. Measure again when the load is small. The command
uptime shows the load.

Use six OpenMP threads. The machine gives twelve logical processors. More
threads help only for one stream larger than about 500 MB. The matrices of the
model are smaller. More threads then give a lower speed. Test both values with
OMP_NUM_THREADS.

The variable OMP_WAIT_POLICY=ACTIVE keeps the threads awake between the 328
kernel calls of one token. The value improved the median time in a test on a
loaded machine. The value did not change the best time.

Test the kernels with this command:

    PYTHONPATH=. $PY scripts/bench_kernels.py

Test the int8 and int4 prompt GEMM with this command. Set NP_GEMMA_ARCH=avx2 to
measure the AVX2 library.

    PYTHONPATH=. $PY scripts/bench_prefill.py

Measured result for one 15360x3840 matrix and one token:

    numpy block dequant bf16   0.070 s
    numba fused bf16           0.007 s
    C fused bf16               0.002 s   (AVX-512, four rows)
    numpy dequant int8         0.107 s
    C fused int8 (float x)     0.007 s
    C integer int8             0.008 s
    blas float32               0.009 s
    C float32                  0.011 s

For one matrix, the C bf16 kernel is the fastest. For the full model, the
memory bandwidth controls the time. The int8 mode reads half the bytes of the
bf16 mode. Thus the int8 mode is the fastest mode.

The C int8 kernel uses one scale for each row by default. The code also gives a
group mode. Use quantize_int8(w, group) with a smaller group for a smaller
error. A small group makes the multiply slower. A group of 32 columns makes a
long chain of additions. For this reason, the per-row scale is the default.

## Flash attention

The plain attention path builds the whole score matrix for one chunk. The
shape is (kv heads, tokens, heads per group, keys). At a long context that
matrix is large. At 8192 keys one global layer holds about 134 MB of scores,
and the softmax reads and writes it a second time. The path also computes the
scores that the causal mask hides.

The flash kernel keeps the scores of one row block and one key block at a
time. It reads only the keys that the block can see, so a sliding layer reads
the window and not the whole context. It holds the key transposed. Thus the
score of a row over a block of keys is a vector over the keys, and it needs no
horizontal sum. The value stays in the natural layout. The online softmax
keeps a running maximum, a running sum, and the running weighted sum.

The kernel has three versions:

    version   key width   note
    c         scalar      The reference. Slow. Use it to check the other two.
    avx2      8 keys      The fallback for a CPU without AVX-512.
    avx512    16 keys     The default when the build gives AVX-512.

Set the version with NP_GEMMA_ATTN_IMPL=c, avx2, or avx512. Leave it unset to
take the best version that the build gives. Set NP_GEMMA_FLASH=1 to use the
kernel. Set NP_GEMMA_FLASH=ref for the NumPy reference. Leave it at 0 for the
batched matmul path.

Check the kernel against the NumPy reference:

    PYTHONPATH=. python scripts/check_flash_c.py

Measure the kernel alone. The batched matmul path uses OpenBLAS, and the model
gives OpenBLAS eight threads, so the table uses eight BLAS threads:

    shape                        batched   avx512
    256x1024 global  hd=512         29 ms     37 ms
    256x1024 sliding hd=256         14 ms     12 ms
    256x8192 global  hd=512        160 ms    279 ms
    256x8192 sliding hd=256         18 ms     20 ms

The sliding layers win, because the window caps the work. The global layers
lose, because OpenBLAS tiles the score matrix better than a register tile of
eight rows. On the full model the two effects nearly cancel. A 2048 token
prompt took 30.8 s with the batched path and 27.2 s with the kernel. A 4096
token prompt took 60.1 s and 58.3 s.

The greedy tokens were equal. The kernel allocates no score matrix, so it fits
a very long context better.

## Key and value cache

The cache holds one key tensor and one value tensor for each layer, and it grows
with the sequence. A sliding layer never reads past its window, so its buffer
does not need to grow:

    layer type   rows kept                          at 16384 tokens
    sliding      2 * window + one prompt block      2303
    global       the whole sequence                 16384

A query at position p sees back to p - window + 1. The first query of a new
block therefore sees back to start_pos - window + 1, and the code drops every
row older than that. The drop is safe, because the window mask hides those rows
for every query of the block. The code used to compact only for a decode step,
so a prompt pass grew a sliding buffer to the full sequence.

Measured cache size. The server runs the float32 cache by default:

    tokens    float32   with the int8 copy   sliding rows
    2048       0.92 GB   1.18 GB              2048
    4096       1.11 GB   1.42 GB              1535
    8192       1.28 GB   1.64 GB              1791
    16384      1.62 GB   2.07 GB              2303

The total now grows only with the five global layers. Before the change a
sliding layer held one row per token. At 16384 tokens the 25 sliding layers
alone held 25 * 16384 * 16 KB, or about 6.7 GB. The whole cache was about 7.4
GB, against 1.62 GB now.

The cache also held an int8 copy of every key and value for the fused
attention. The float32 path never reads that copy. The code now builds it only
when NP_GEMMA_ATTN is 1. Then the default float mode does not spend the memory
or the quantization time. That copy is the difference between the two columns.

The copy is now int16, with a float query. The int8 copy with an int8 query
moved a logit by up to 2.6 against the reference. The int16 copy moves a
logit by at most 0.0014, and the float cache by 0.0003 (PERF_PLAN.md,
"Accuracy against the reference"). The
int16 copy is about two times the size of the int8 copy in the table. The
server now uses it by default.

Check the bound and the output:

    PYTHONPATH=. python scripts/check_kv_window.py 2048 4096 8192 16384

## Prefill experiments

Four ways to make the prompt pass cheaper were tried. Three change how the
machine works; the fourth removes work. The measurements use a 4096 token prompt
unless the text says otherwise.

**Layer-major schedule.** The code normally runs every layer for one block,
then the next block. A layer-major pass runs one layer for the whole prompt,
then the next layer. Thus the model loads a layer one time, and every expert
sees the whole prompt.

At the same block of 256 tokens it was 58.6 s against 61.0 s, a gain of 4
percent. A block of the whole prompt was 65.7 s, which is 12 percent slower,
because the working set no longer fits the cache. Use
Model.prefill_layer_major.

**Weight prefetch.** The int8 tile asks the prefetcher for the next group of
weights and activations. It is slower: 0.87 to 1.00 of the speed without it. The
prefetch instructions cost more than the stall they hide. Use
cops.set_int4_prefetch(1).

**Wider token block.** The wide tile reads 32 tokens for one weight decode
instead of 16. The result is the same to the last bit and it is 0.57 to 0.80
of the speed of the 16 token block. The row block falls from 8 to 4 to make
room in the registers. Then each activation load and each weight broadcast
serves half the rows. Use ops.linear_int4_q8_wide.

A test of the VNNI instruction says the tile spends more time on the
instructions that do not multiply than on the multiply itself. That result
suggested the wider block. The wider block then lost, so the instruction count
is not the whole story: the register pressure and the smaller row block cost
more.

**Prefix reuse.** A conversation sends the whole history again on each turn. The
session cache keeps the keys and values of the shared prefix, so a later turn
reads only the new tokens. Measured on a 3776 token conversation:

    turn   prompt   prefill
    1      3776     59.19 s
    2      3796      1.17 s
    3      3816      1.20 s
    4      3836      1.08 s

Four turns cost 62.7 s against about 237 s for four cold prompt passes, so the
reuse saves 74 percent. That is larger than every kernel change in this file put
together, and it is the reason the server keeps a session.

Measure the four with scripts/bench_prefill_schedule.py, scripts/bench_tile.py,
and scripts/bench_prefix.py.

## Weight cache

The int8 conversion reads the full model and quantizes six billion parameters.
The command stores the result in ~/.cache/np_gemma/weights/<key>/. The
directory holds two files:

    manifest.json   the dtype, the shape, and the offset of each tensor
    data.bin        the raw bytes

The first int8 load writes the cache. Later loads read the cache. The cache key
covers the source path, the source size, the source time, and the dtype. Thus
the code never uses a stale cache.

The default directory is in the home directory. The home directory is a local
disk. Do not put the cache in the project directory when the project is a
network mount. Set the variable NP_GEMMA_CACHE to use a different directory.

Set the variable NP_GEMMA_CACHE_RAM=1 to copy the cache into one block of
anonymous memory. The block starts at a 2 MiB boundary. The system then gives
2 MiB pages. One large page covers 512 small pages. Thus the read needs fewer
TLB entries.

The copy gave 1.09 times to 1.24 times more speed in a test. The copy uses the
same memory as the file page cache. The count is larger for a short time
during the load. The default is the memory map.

Measure the two caches with this command:

    PYTHONPATH=. $PY scripts/bench_ram_cache.py --snapshot "$SNAP"

Measured on this machine:

    first int8 load (read the model, quantize, write the cache)   402.8 s
    later int8 load (read the manifest)                             5.4 s
    first pass after a later load (page in 11.9 GB)             about 12 s

The cache removes the quantization work. The first pass still reads the cache
from the disk. Put the cache on a fast local disk for the best result. The
network mount gave about 115 MB/s, so the 11.9 GB cache needed about 104 s for
the first pass. The local NVMe disk gives about 1 GB/s, so the first pass needs
about 12 s. A warm decode step is 0.72 s per token.

## Compare with Hugging Face

The file check_trace.py compares each intermediate tensor with a trace from the
real model. The reference trace is in
../gemma4-12b-qat-pytorch/traces/ref-france-pos3.

    PYTHONPATH=. $PY scripts/check_trace.py         --config "$SNAP/config.json" --weights "$SNAP/model.safetensors"         --trace ../gemma4-12b-qat-pytorch/traces/ref-france-pos3

Add --layers 1 for a quick test. Layer 0 needs only 0.5 GB of the 24 GB file.

The file check_cache.py compares the key and value cache with a batch prefill.

    PYTHONPATH=. $PY scripts/check_cache.py --config "$SNAP/config.json"         --weights "$SNAP/model.safetensors"         --trace ../gemma4-12b-qat-pytorch/traces/ref-france-pos3 --layers 1

The file gen_ids.py writes greedy token ids. Compare the ids with the HF file
../gemma4-12b-qat-pytorch/scripts/hf_gen_ids.py.

    PYTHONPATH=. $PY scripts/gen_ids.py --snapshot "$SNAP" --dtype bf16         --prompts "Count from 1 to 10." --max-new-tokens 24 --out ids.json

## Gemma 4 E4B

`google/gemma-4-E4B-it-qat-mobile-ct` is the 4.5B "effective" dense model. It
is a different runtime from the 12B, so it has its own module and its own
scripts. The new idea in this model is Per-Layer Embeddings (PLE). Every
decoder layer gets its own small embedding for each token. The layer adds it
to the residual stream as a second signal.

Two published files of this model work with this runtime:

    file                                 form          size
    gemma-4-E4B-it-qat-mobile-ct         SafeTensors   3.73 GB
    gemma-4-E4B-it-qat-q4_0-gguf         GGUF Q4_0     5.15 GB

The first file is the compressed-tensors "pack-quantized" form. It is not the
w4a16 form. One file holds four kinds of weight:

    weight                     bits  strategy
    the token embedding
      and the output head        2   one scale for each row
    the per-layer table          2   one scale for each group of 256
    the decoder projections      4   one scale for each row
    the per-layer gates          8   one scale for each row

`np_gemma/ct.py` reads that form straight from the memory map. It infers the
bit width from the packed shape and the logical shape, so it does not need the
quantization section of the configuration file. `scripts/check_ct.py` shows
that the decode matches the `compressed_tensors` library exactly, for all four
kinds, and that `row()` agrees with the whole-matrix decode.

### The architecture

    layers                      42
    global attention layers     5, 11, 17, 23, 29, 35, 41
    sliding window              512
    head size                   256 in a sliding layer, 512 in a global layer
    query heads                 8
    key and value heads         2
    key and value sharing       the last 18 layers reuse layers 22 and 23
    attention scale             1.0; the query norm already gives unit RMS
    query and key norm          yes, with a scale
    value norm                  yes, without a scale
    per-layer width             256
    vocabulary                  262144
    logit soft cap              30
    tied output head            no

Three details differ from the 12B model:

1.  The head size changes with the layer. The reference calls this
    `per_layer_config`. A sliding layer uses 256 and a global layer uses 512.
2.  The last 18 layers have no `k_proj`, no `v_proj`, and no `k_norm`. They
    reuse the key and the value that layer 22 (sliding) or layer 23 (global)
    stored. The file still holds a `k_proj` and a `v_proj` for those layers.
    They are unused and this runtime does not read them.
3.  `attention_k_eq_v` is false, so the key and the value stay separate in
    every layer. The 12B model reuses the key as the value in a global layer.

Two constants are rounded to bfloat16 before use, because the reference casts
them to the weight dtype. The embedding scale `sqrt(2560)` becomes 50.5. The
per-layer input scale `2**-0.5` becomes 0.70703125. Without that rounding the
output drifts.

The model also differs in its chat template. When thinking is off, the 12B
starts the answer with an empty `<|channel>thought\n<channel|>` block. The E2B
and E4B models do not. `render_chat` and `apply_chat_template` take the
argument `empty_thought_block` for this; pass False for E4B.

### Run it

Copy the checkpoint to the local disk first. The HuggingFace cache of this
project lives on an sshfs mount. The int4 mode reads the weights on every
token, so the storage under the file sets the speed. Copy the files, not the
symlinks:

    mkdir -p ~/.cache/e4b-mobile-ct
    cp -L <the snapshot>/* ~/.cache/e4b-mobile-ct/

Set the paths:

    cd numpy-gemma
    PY=../gemma4-12b-qat-pytorch/.venv/bin/python
    SNAP4B=~/.cache/e4b-mobile-ct

Show the tensor layout of the file. This step reads the header only:

    PYTHONPATH=. $PY scripts/e4b_inspect.py "$SNAP4B/model.safetensors" --layer 0 --layer 24 --prefixes

Generate text in the int4 mode. This mode reads every packed weight on each
token. Thus set the BLAS thread count to one. With more than one thread the
BLAS pool fights the OpenMP regions of the kernel, and the run is about eight
times slower. The script warns when the setting is wrong.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=6 OMP_WAIT_POLICY=ACTIVE \
        PYTHONPATH=. $PY scripts/e4b_generate.py --snapshot "$SNAP4B" \
        --mode int4 --prompt "The capital of France is" --max-new-tokens 8

The Python interface is small:

    from np_gemma.ct import CompressedTensors
    from np_gemma.e4b import E4B, E4BConfig, E4BCache

    ct = CompressedTensors(snapshot + "/model.safetensors")
    cfg = E4BConfig.load(snapshot + "/config.json")
    model = E4B(ct, cfg, mode="int4")
    cache = E4BCache(cfg)
    hidden = model.forward(prompt_ids, cache=cache, start_pos=0)
    print(model.generate(prompt_ids, max_new_tokens=8))

The mode chooses how the weights are kept:

    mode      what the memory holds                     one token reads
    f32       every weight as float32, about 15 GB      18.6 GB
    stream    one layer                                 2.2 GB of packed words, decoded every time
    int4      the packed 4-bit and 2-bit words, 2.2 GB   2.2 GB of packed words, decoded by the kernel

The mode "int4" is the one to use. The lines below give the numbers.

The same runtime reads the Q4_0 GGUF file of this model. Put the file on the
local disk and set the path:

    GGUF4B=models2/gemma-4-E4B-unsloth-UD-Q4_K_XL/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf

Use the QAT file of Unsloth (see "The Unsloth Q4_0 files of the E4B and the
E2B"). All its tensors are Q4_0.

The file of Google holds Q4_0 weights, Q6_K embedding tables, and one F16
matrix.
The block layout of Q4_0 is the layout that the int4 kernel of the 12B model
already takes. Thus the mode "int4" needs no new kernel. The class `E4B` takes
either source and selects the kernel from it. The check compares the GGUF run
with a HuggingFace reference:

    PYTHONPATH=. $PY scripts/check_e4b_gguf.py --gguf "$GGUF4B" \
        --config-snapshot "$SNAP4B"

The config snapshot gives the dimensions and the tokenizer. The weights come
from the GGUF file.

### The int4 kernel

The checkpoint packs a weight row into int32 words. Element k of a row starts
at bit k * bits from the start of the row. The value carries a bias of 2 **
(bits - 1). One float32 scale covers the whole row, because these matrices use
the "channel" strategy:

    out[t][r] = scale[r] * sum over k of x[t][k] * q[r][k]

`gemma_ct_linear` in `np_gemma/csrc/bf16_linear.c` reads those words where the
file put them. It unpacks them in the registers, converts to float32, and uses
a fused multiply and add. It never writes a float32 copy of a weight, and it
never builds a second copy of the file. So the layout of the file is the
layout of the runtime, which is what makes the memory map enough.

The kernel has an AVX-512 version, an AVX2 version, and a plain C version, in
the same shape as the other kernels in that file. A 4-bit row takes two words
for each 16 values. A 2-bit row takes one word. Both fill a 512-bit register
with one fused multiply and add.

`scripts/check_ct_kernel.py` checks the kernel against the decoded weights. It
gives the kernel the one-hot vector e_k. The kernel then returns one column of
the matrix. The script does that for every column of a group of rows. Thus the
check tests every bit position of the row. The 4-bit and the 2-bit forms both
agree exactly.

A block of random tokens agrees to 1e-6 relative. The difference is the
float32 rounding. The kernel applies the scale after the sum, and the NumPy
path applies it to each group.

Two notes:

*   The kernel takes a channel scale, one value for each row. The per-layer
    embedding table uses a group scale of 256, and the two per-layer
    projections use 8 bits, so those three keep the decoded path. They are
    small: the 8-bit pairs are 55 M parameters against 4.56 G for the packed
    matrices.
*   `OPENBLAS_NUM_THREADS=1` matters. The model makes about 300 kernel calls
    for each layer pass, and each one opens an OpenMP region. A BLAS pool with
    six threads fights those regions: the measured decode is 2.5 s a token
    against 0.33 s. This is the same effect the section "Start here" notes for
    the int8 kernel of the 12B model.

### The elementwise kernels

`gemma_rms_norm` normalizes the last axis of a tensor. A decode step calls it
211 times for each token, on rows of 256 to 2816 values. The cost of one call
is now nearly the same for every row length. Thus the fixed cost of the call is
the larger part:

    cols            1     256     704    2816
    microseconds  7.58   7.71    7.86    8.42

That measurement uses buffers that are already in place, so it holds the kernel
and the call and not the Python. A call of `gemma_gelu` on one value gives the
floor of a C call on this machine: about 5.9 microseconds. The work of the
normalization is therefore about 0.9 microseconds for a row of 2816 values.

The first form of the kernel summed the squares in one chain. The latency of an
addition is about 4 cycles, so that loop used one value for 4 cycles and the
multiply units waited. The kernel now uses four accumulators at 512 bits, which
takes the multiply and the add of 64 values in each step. The work of one row
fell from 7.1 to 2.5 microseconds.

The sum now adds the values in a different order, so the result differs from a
left to right sum in the last bits. `scripts/check_rms_norm.py` gives the size
of that difference against a float64 reference. It is 1.1e-07 or better over 14
shapes, and the rounding of the true value to float32 alone is 3 to 5e-07. The
generated token ids of the 12B, the 26B, and the E4B do not change.

A decode step has one row, and the work of that row is below the cost of a
thread team. For fewer than 8 rows the kernel therefore stays out of the OpenMP
runtime. That step alone saved about 1 microsecond for each call.

### The cost of a call to C

A decode step of the 26B model made about 600 calls to the C library. Each call
passes pointers, and the cost of a call is almost all in the pointers. The
measurement below uses an empty C function, so the times are the cost of the
call alone:

    the call                                      microseconds
    ctypes, no argument                                   0.24
    ctypes, each int argument                             0.10
    ctypes, each pointer argument                         1.73
    one read of ndarray.ctypes.data                       1.5

Thus a call with three pointers costs about 5 microseconds before the kernel
starts, and the read of `.ctypes.data` is the larger part of that. A run of
`gemma_rms_norm` with the buffers already in place gives 6.0 microseconds when
the code reads `.ctypes.data` for each call. It gives 1.6 microseconds when the
code keeps the addresses. The Python wrapper above the call costs a further 4
to 5 microseconds.

`.cache/addr_reuse.py` counts the buffers that one function hands to the C
library more than one time. One decode step of the 26B model hands over 1412
arrays. Only 112 of those are a second use of a buffer by the same function.
The rest are new activations, or a weight that a different function uses.

    function            extra reads    the buffer
    qkv_norm_rope                54    the cosine and sine tables
    rms_norm_multi4              29    the scratch of the module
    gelu_mul_int4                29    the scratch of the module

A first measurement gave 633 extra reads. That number was wrong. The count
used the identity of each array, and an array that is no longer alive lets a
new array take the same identity. The script now holds a reference to every
array that it counts.

The three buffers above now keep their address with the buffer. The scratch
carries it, and the rope table cache carries it next to the table. The count
of extra reads is 0.

That saves about 0.17 ms of a step, or 0.2 per cent. It is too small to
separate from the noise of the machine.

The other 1300 reads are not a repeat inside one function. To remove those, a
buffer must keep its address across calls, and the model must own a pool of
buffers for that. A pool carries a real danger: a caller that keeps a result
while the next call writes into the same buffer.

Two changes follow.

**The environment is read one time.** The functions of `ops.py` asked for
`NP_GEMMA_KERNEL`, `NP_GEMMA_INT4_Q8_GEMV`, and three more variables on every
call. A lookup of the environment costs about 3 microseconds, and a decode step
made 360 of them. Every value is now read at the import of the module. Set
these variables before the import.

**Two kernels run in one call.** The C library now gives four fused entry
points. Each one is a composition of kernels that already exist, so the result
does not change:

    gemma_qkv_norm_rope    the three norms and the two rotations
    gemma_rms_norm_multi4  the norm of a row, then up to four int4 matrices
    gemma_gelu_mul_int4    gelu(g) * u, then one int4 matrix
    gemma_moe_gemv_gelu    the gate and up projection, then the GELU

The fused form is 3.8 per cent faster at the decode of the 26B model. Four
interleaved pairs of `tg128` gave 12.38 tokens a second without the fusion and
12.85 with it. The generated token ids are equal for the 12B, the 26B, and the
E4B, and for a prompt of 1024 tokens. Set `NP_GEMMA_FUSED_STEP=0` for the
separate calls.

The count of calls for each token falls from about 600 to about 420. The
remaining calls are the four large matrix kernels, which do most of the work,
and the norm calls that the fusion does not reach.

### The Q4_0 GGUF source

`np_gemma/gguf.py` reads the GGUF file with NumPy only. It maps the names of
the Gemma 4 blocks onto the names of this runtime. The per-layer embeddings
need these mappings:

    GGUF name                       runtime name
    blk.N.inp_gate.weight           layers.N.per_layer_input_gate.weight
    blk.N.proj.weight               layers.N.per_layer_projection.weight
    blk.N.post_norm.weight          layers.N.post_per_layer_input_norm.weight
    per_layer_model_proj.weight     per_layer_model_projection.weight
    per_layer_proj_norm.weight      per_layer_projection_norm.weight
    per_layer_token_embd.weight     embed_tokens_per_layer.weight

The GGUF file has no output head. The token embedding is the output head,
because the two matrices hold the same values. The mobile-ct file proves that
point: its `lm_head` weight and its `embed_tokens` weight are equal bit for
bit. Thus the GGUF runtime reads the token embedding table for the head.

Three tensors differ from the mobile-ct file:

    tensor                        GGUF                   mobile-ct
    the token embedding           Q6_K                   2 bit, row scale
    the per-layer table           Q6_K                   2 bit, group scale of 256
    the per-layer model map       F16                    4 bit, row scale

The Q6_K tables are more accurate than the 2-bit tables, but they are larger.
One token reads 2824.8 MB from the GGUF file against 2443.2 MB from the
mobile-ct file. The output head alone is 550.5 MB against 167.8 MB. The GGUF
file keeps the token embedding in Q6_K, which is the common choice for a Q4_0
file. The head is the largest single cost of the decode.

The int4 kernel takes the Q4_0 blocks in place. `GGUF.int4_packed` returns a
view of the file map, so the mode "int4" copies no weight. `GGUF.tensor_bytes`
gives the stored size of one tensor without a read of the data;
`scripts/profile_e4b_gguf.py` uses it for the byte table of a decode step.

### Check it against the reference

The check script compares the E4B file against a HuggingFace reference in two
steps. Thus the two models never hold memory at the same time:

    PYTHONPATH=. $PY scripts/e4b_trace.py hf --snapshot "$SNAP4B" \
        --prompt "The capital of France is" --out .cache/hf_e4b.npz
    PYTHONPATH=. OMP_NUM_THREADS=6 $PY scripts/e4b_trace.py np --snapshot "$SNAP4B" \
        --trace .cache/hf_e4b.npz

The first step writes the hidden state after each of the 42 layers, the
per-layer embeddings, and the logits. The second step prints the largest
difference at each layer.

One warning about the reference.
`Gemma4ForConditionalGeneration.from_pretrained` does not work for this
checkpoint in transformers 5.17. The compressed-tensors loader keeps
`weight_packed` and `weight_scale` as parameters, and it removes `weight`.
Then the shared `_init_weights` asks for `module.weight`, so the load stops.
If the call is patched to continue, the load finishes but the model returns
zeros. The dequantize step never runs for those modules.

The check script therefore builds a plain `Gemma4TextModel`. That class has no
quantized modules. The script fills it from `np_gemma.ct`. Thus the reference
is the forward pass of HuggingFace itself. That is the purpose of the check.

### Does the memory map make sense?

`scripts/bench_e4b_mmap.py` answers the question. The answer depends on where
the file lives, so the script reports the mount as well.

Measured on this machine. The benchmark ran on the local copy in
~/.cache/e4b-mobile-ct, with OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=6, and
OMP_WAIT_POLICY=ACTIVE.

    file                                          3.734 GB
      the two embedding tables                    0.872 GB   read by row
      the matrices and norms                      2.224 GB   read in full per token
    the same weights as float32                  32.536 GB

    local ext4, /dev/nvme0n1p2
      first read of the text model                 0.50 s     6.17 GB/s
      second read                                  0.50 s     6.26 GB/s
      the per-token weights, second read           0.35 s     6.28 GB/s
      memory map, one touch for each 4 KiB page    0.17 s     4.5 M pages/s

    sshfs, the HuggingFace cache inside the project
      first read of the text model                29.5 s      0.10 GB/s
      second read, same file handle                0.53 s     5.80 GB/s
      the per-token weights, second read           0.38 s     5.90 GB/s

    decode the per-token weights with NumPy        21.6 s    18.58 GB of float32

    mode      prefill s/token   decode s/token   private memory
    f32                 0.834             1.75    15.0 GB
    stream              0.628            20.35     1.6 GB
    int4                0.126             0.26     0.44 GB

The decode figure is the steady state, from the second token on. The first
decode step of the f32 mode also loads the output head. That matrix is 2.68 GB
as float32 and 168 MB packed. Thus the mean of the first four steps is 2.80 s
in that mode. The output head is the largest single matrix in the model.

The conclusions:

*   **Copy the file to the local disk.** The project cache is on an sshfs
    mount to jackal.local. That mount reads at 0.10 GB/s, so a mode that reads
    the weights on each token cannot work there. The local ext4 disk reads the
    same file at 6.2 GB/s, sixty times faster, and it also holds the page
    cache. The int4 mode went from 2.57 s a token on
    sshfs to 0.26 s on the local copy. The storage causes most of that
    difference.

*   **The map is fine for a row.** A token reads 640 bytes of the token
    embedding, 2688 bytes of the per-layer table, and 86 bytes of scales. The
    map reads 3.4 KiB for a token. A model that reads the tables whole reads
    2.2 GB. This is the strongest argument for the map, and it is the reason
    the two tables stay in the file in every mode.

*   **The map is not a way to avoid the projections.** The model reads every
    decoder projection once for each token. The stream mode therefore decodes
    18.58 GB of float32 again for each token. That is 21.6 s of NumPy work, and
    the mode measures 20.35 s a token. A memory map does not remove that work.
    It only moves where the source bytes live.

*   **The float32 copy was the real problem.** The f32 mode holds 15.0 GB of
    private memory and reads 18.58 GB for each token. The packed 4-bit and
    2-bit matrices are 2.113 GB, and the int4 kernel reads them in place. That
    is 8.8 times fewer bytes for each token. The decode takes 6.7 times less
    time and the prefill 6.6 times less. The mode needs 34 times less private
    memory.

*   **The kernel is not the limit, and neither is the storage.** The int4
    mode reads the per-token weights at 10.5 GB/s on nosey. The same kernel in
    one tight loop reads at 27.3 GB/s, which is 73 per cent of the machine
    rate. On jackal the same loop reads at 46.2 GB/s, which is 77 per cent of
    the machine rate. See the next section for the rest of the time.

### Why the decode is slower than the memory rate

One token reads 2443.2 MB from the mobile-ct file and 2824.8 MB from the GGUF
file. The floor for one token is the byte count divided by the read rate of
the machine. Two machines give these numbers:

    machine          cores   read rate   decode    floor   ratio
    nosey, W-2133        6   37.6 GB/s   0.26 s   0.065 s   4.0
    jackal, W-2295      18   59.8 GB/s   0.10 s   0.047 s   2.1

`scripts/profile_e4b.py` measures the mobile-ct file.
`scripts/profile_e4b_gguf.py` measures the GGUF file. Each script splits one
decode step into the matrix kernels, the output head, and everything else.

The GGUF file on jackal, 18 threads, ten steps:

    step                0.0862 s   median
      the kernels       0.0566 s   66 per cent
      the output head   0.0121 s   14 per cent
      everything else   0.0174 s   20 per cent

The table gives the mean of the ten steps:

    group                           calls      MB   seconds    GB/s
    mlp gate, up, and down            126  1857.9    0.0411   45.20
    output head (Q6_K)                  1   550.5    0.0111   49.57
    attention q and o                  84   289.0    0.0091   31.76
    per-layer projection and gate      84    31.0    0.0061    5.08
    per-layer model projection           1    55.1    0.0009   61.90
    attention k and v                  48    41.3    0.0014   29.49
    TOTAL                             344  2824.8    0.0698   40.47

The call column counts the matrices. The fused kernel runs two or three of
them in one call, so the model makes 254 calls for one token.

The floor for these bytes at 59.8 GB/s is 0.0472 s. The kernels are 1.5 times
the floor. The same measurement for the mobile-ct file on nosey, six threads,
gives 2443.2 MB in 0.2439 s. The rate is 10.02 GB/s, which is 3.8 times the
floor. The bfloat16 copy of the per-layer model projection takes that call
from 7.7 ms to 1.2 ms on the same machine.

The machine varies by about 8 per cent between two runs of the same command.
Treat a single step value as a guide and not as a measurement.

Two changes took the step from 0.1055 s to 0.0965 s. The measurement
interleaves the settings, and each value is the mean of two runs:

    change                            step      kernel
    the start                       0.1055 s   0.0677 s
    a bfloat16 copy of the
      per-layer model projection    0.1017 s   0.0622 s
    one call for the query, the
      key, and the value            0.1044 s   0.0665 s
    both                            0.0965 s   0.0598 s

The attention change of the section "The prompt pass" also helps the decode.
It removes the `np.repeat` and the NumPy softmax from every layer. The effect
is small at a short context, and large at a long one.

#### Three ideas that the profile rules out

**The pause between kernel calls is not the cost.** The model runs work
between two kernel calls that the OpenMP pool does not join. A test walks 16
matrices of 14.7 MB, which is more than the cache of the machine. A pause of
0.05 ms before a call costs 1.01 times, and a pause of 1 ms costs 1.10 times.

A second test puts a norm and an attention product between the calls. That
costs 1.00 times. So a fused call for the whole layer saves almost nothing.

**The storage is not the cost.** The packed words come from the file mapping.
A copy of the same words in ordinary memory with large pages gives 14.32 GB/s
against 14.01 GB/s for the mapping. That is 1.02 times.

#### What the cost is

**The inner loop is the cost.** One thread reads plain memory at 13.4 GB/s.
One thread reads a packed 4-bit matrix at about 4.9 GB/s. So the unpack and
the conversion cost about 2.7 times more than the read alone. The 16 matrices
of the tight loop give 46.2 GB/s on jackal, which is 77 per cent of the machine
rate. Each group of 16 values costs about ten instructions: load the word,
take the two nibbles, convert to float32, and multiply and add.

**A short run for each thread is also a cost.** The small matrices lose the
most. On jackal the feed-forward matrices reach 42.6 GB/s inside the model,
against 46.2 GB/s in the tight loop. The per-layer projection is 0.37 MB for
each layer and reaches 3.5 GB/s. The kernel opens an OpenMP region for each
call, and 0.37 MB does not pay for that region.

**The tanh of the GELU was the largest cost outside the kernels.** The old
`gemma_gelu` called the scalar `tanhf` function of the C library for each
value, at about 20 ns for each value. The E4B model needs 440000 values for
each token, so the function cost 17.2 ms of a 125 ms token. The new kernel
uses an AVX-512 form of tanh with a degree-6 polynomial, and the cost falls
to 1.5 ms. The decode went from 0.120 s to about 0.10 s. This change helps
every model in this project.

`scripts/check_gelu.py` measures the error against float64. The error is the
same as the error of the scalar function.

**The per-layer model projection was the largest single cost.** The matrix is
F16 in the GGUF file and BF16 in the mobile-ct file, and 55 MB in both. The
mode "int4" kept a float32 copy, and the multiplication used one BLAS thread
because the kernel needs `OPENBLAS_NUM_THREADS=1`. The rate was 6.5 GB/s, at
8.4 ms for each token. The model now keeps a bfloat16 copy and uses
`cops.linear_bf16`. The rate is 58 to 62 GB/s, at 0.9 ms.

The change gives 7.5 ms, which is the largest single gain of this decode. A
small weight keeps the float32 path, because the start of the kernel costs
more than the read.

**One kernel call can run several matrices.** The query, the key, and the
value projection share the input row of the attention block. The gate and the
up projection share the input row of the feed-forward block. One call with
`gemma_int4_multi4` runs each group, so the model opens 254 OpenMP regions for
a token in place of 344. That change gives 1.1 ms alone and 3.3 ms together
with the bfloat16 change.

**VNNI gives 10 to 14 per cent on the prefill.** The int8 tile of the 12B
model uses `vpdpbusd` when the CPU gives AVX-512 VNNI. A CPU without VNNI uses
an emulation with `maddubs`. The measured E4B prefill on jackal, 18 threads:

    tokens   VNNI      no VNNI   ratio
         14   0.331 s   0.377 s   1.14
         64   1.480 s   1.647 s   1.11
        256  10.696 s  11.811 s   1.10

The int8 tile also beats the float tile, by 1.2 to 1.4 times at 18 threads.
For one token the int8 tile is slower, because the token lanes stay empty.

### The prompt pass

A prompt pass reads the same 2825 MB of weights one time, but it uses each
weight for every token. Thus the prompt pass is compute bound. The useful
measure is the number of arithmetic operations for each second.

`scripts/profile_e4b_prefill.py` splits one prompt pass. The measurement is on
jackal with 18 threads, after a warm-up:

    tokens     before      after    GFLOP/s
        14    0.336 s    0.266 s       516
        64    1.376 s    0.767 s       818
       256   10.493 s    2.920 s       859
       512   35.217 s    6.646 s       755
      1024        --    14.489 s       692

The column "before" is the state at the start of this measurement. The matrix
kernels were 23 per cent of a 256-token pass, and the glue was 73 per cent.
The two attention products alone were 6.64 s of 9.92 s.

#### The attention product did not reach the BLAS library

`np.einsum("thd,shd->hts", q, kk)` ran at 4.8 GFLOP/s. The index order does
not let NumPy use a matrix multiply, so the call used the slow NumPy loop. The
same product with `np.matmul` runs at 86 GFLOP/s, which is 18 times faster.
The 12B model already used `np.matmul` for this reason.

The code now uses the batched matrix multiply of the 12B model. That form also
removes the `np.repeat` of the key and the value. It replaces the NumPy mask
and the NumPy softmax with one C call to `ops.softmax_mask`. A prompt of 256
tokens then took 3.86 s in place of 10.51 s.

#### The C flash kernel for the prompt

`NP_GEMMA_FLASH=1` sends a prompt of more than one token to the C flash kernel.
That kernel walks only the keys that the mask leaves visible, and it uses the
OpenMP pool in place of one BLAS thread. E4B measures the kernel ahead at
every prompt length. The default is now 1 for every model.

    tokens   matmul    flash   ratio
        14   0.274 s  0.283 s   0.97
        64   0.763 s  0.755 s   1.01
       256   3.235 s  2.860 s   1.13
       512   7.590 s  6.646 s   1.14
      1024  18.014 s 14.489 s   1.24

Each value is the mean of two runs, and the two settings run in turn. The
tokens agree at every length. The logits of the two paths have cosine 0.9996
at 256 tokens and 0.9967 at 1024.

The C kernel also agrees with the HuggingFace reference a little better than
the matmul path does. The cosine at layer 41 is 0.9996 for the kernel and
0.9994 for the matmul.

#### What is left

After the changes, a prompt of 256 tokens takes 2.89 s:

    part                seconds   share
    matrix kernels       2.32 s   80 per cent
    the glue             0.20 s    7 per cent
    the rest             0.37 s   13 per cent

The int8 tile is the cost now, at about 900 GFLOP/s. A sweep of the settings
shows that the current ones are already the best. The token block of 16 wins
at 14 tokens, and the float path is 1.8 times slower at 256 tokens.

The remaining items are small. The rope table of a prompt is now made two
times in place of 42, and that change also helps the decode. The per-layer
model projection costs 0.11 s for 14 GFLOP.

A sliding layer cannot drop keys inside one prompt block, because the first
query of the block sees key zero. The code keeps that trim for a decode step
at a long context. The attention of one such step with 512 keys falls from
0.151 s to 0.013 s.

### Compare with llama.cpp

llama.cpp is the reference implementation for a quantized GGUF model. The
comparison uses the same three GGUF files, the same machine, and 18 threads.
The build is `llama.cpp/build-vnni` with AVX-512 VNNI. The command is:

    llama-bench -m FILE -p 512 -n 128 -r 2

`scripts/bench_gguf_models.py` gives the same two numbers for this runtime:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=18 OMP_WAIT_POLICY=ACTIVE \
        PYTHONPATH=. $PY scripts/bench_gguf_models.py --gguf FILE \
        --prompt 512 --gen 128 --reps 2

The two runtimes run in turn, for three rounds. The table gives the mean. The
earlier run carried a machine load of 18 and the run below carried a load of 2.
Thus the two runs do not agree in the last digit of each value. Both runtimes
gain from the lighter load, so the ratio moves less than the value. The table
below is the later run:

    model              runtime        pp512     tg128
    gemma-4-12B Q4_0   llama.cpp      42.4 t/s   7.46 t/s
    gemma-4-12B Q4_0   numpy-gemma    29.0 t/s   5.53 t/s
    gemma-4-26B Q4_0   llama.cpp      81.0 t/s  18.43 t/s
    gemma-4-26B Q4_0   numpy-gemma    76.9 t/s  12.20 t/s
    gemma-4-E4B Q4_0   llama.cpp     106.8 t/s  16.17 t/s
    gemma-4-E4B Q4_0   numpy-gemma    73.3 t/s  11.26 t/s

    model              pp512   tg128
    gemma-4-12B        1.46x   1.35x
    gemma-4-26B        1.05x   1.51x
    gemma-4-E4B        1.46x   1.44x

The runs before that one gave other ratios. For the 12B they were 1.53 and 1.43,
then 1.46 and 1.38. For the 26B they were 1.09 and 1.57, then 1.09 and 1.55. For
the E4B they were 1.53 and 1.47, then 1.47 and 1.46. The runs agree on the
ratios to about 0.05. The clock of the machine was about 2.3 GHz in every run;
see "Memory bandwidth".

Three points follow from the table:

1.  The prompt pass of the 26B model is at parity. The mixture-of-experts
    prefill of this runtime is competitive with llama.cpp.
2.  The 12B model has the smallest margin. It is a dense model, so each token
    reads 6.48 GiB. The rate of llama.cpp is 53 GB/s against a machine rate of
    59.8 GB/s. This runtime reaches 38 GB/s. Both sit near the limit.
3.  The decode of the 26B model is the largest gap.

#### The A4B decode

The 26B model is a mixture of experts with 4B active parameters. One token
reads 2390 MB:

    part                   bytes
    the experts            801 MB    8 of 128 experts for each of 30 layers
    the attention          681 MB    the query, key, value, and output maps
    the output head        605 MB    Q6_K
    the dense feed-forward 301 MB    the gate, the up map, and the down map

`scripts/profile_gguf_decode.py` gives the stage report and `glue26.py` in the
cache directory gives each call. At a context of 512 tokens:

    call                  calls   seconds   share   rate
    experts                  60   21.5 ms    21%   37 GB/s
    projections             120   16.7 ms    16%   30 GB/s
    output head               1   16.0 ms    15%   38 GB/s
    query, key, value        30   11.3 ms    11%   50 GB/s
    attention                30   11.1 ms    11%
    norms                   211    5.5 ms     5%
    router                   30    3.8 ms     4%
    the Python of the loop    1   12.4 ms    12%

The machine reads at 59.8 GB/s. The model reaches 27 GB/s for each token
against 43 GB/s for llama.cpp. The size of one matrix explains most of the
difference, because the matrices of this model are small. One process gives
this curve, so the readings share one load:

    matrix                        rate
    6.7 MB merged gate and up   34.5 GB/s
   14.8 MB (the E4B gate)       42.2 GB/s
   17.8 MB, the 8 experts       38.0 GB/s

The 26B model has 90 matrices of 3.3 MB. The same kernel reads the larger E4B
matrix at 42 GB/s. A small block does not reach the rate of the memory.

Two changes follow from the profile:

*   The gate and the up map of the dense feed-forward part share the input
    row. One call for both reads 6.7 MB in place of two reads of 3.3 MB. In
    one process, over the 30 layers, the pair goes from 212.6 to 172.5
    microseconds for each layer, which is 31.5 to 38.8 GB/s.
*   The rope table is now made one time for each layer type in place of one
    time for each layer. A prompt of 30 layers makes two tables.

The attention of this model already used the fused kernel, and it is 1.28
times faster than the batched matmul here.

#### The int4 matrices in groups of 16 rows (KQ_Q4X)

Model.load_all (int4) makes a second copy of each int4 matrix in groups of 16
rows (ops.q4x_pack_model, KQ_Q4X in csrc/kquants.c). This is the layout of
the NVFP4 experts of Qwen3.8 (KQ_NVX), with Q4_0 codes and one float16 scale
for each row and block. A lane of vpdpbusd is a row. The copy is in memory
and takes the int4 bytes again (13 GB for the 26B). NP_GEMMA_Q4X=0 turns it
off.

- The prompt: the experts and the dense matrices use it with int8
  activations. The experts of a layer with 512 tokens take 18 ms, not 67.
- The default of the prompt activations was then int8 for every model
  (NP_GEMMA_INT4_Q8=1, the form of llama.cpp). The 26B now takes int16
  again, for every product of the prompt program (SPLIT_PLAN.md, "The int16
  prompt of the 26B").
- The decode: the records of one token and of a verify group read the copy
  with float32 activations. The products agree with those of the old
  kernels to 3e-6, and a verify group gives the bits of steps
  (scripts/check_mt.py). The experts go from 16.3 to 13.8 ms for each
  token, and the step from 56 to 51 ms.
- The parts of a step (np_gemma/parts.py) read the rows of the copies. Their
  copies of the experts are KQ_Q4X too, and they give the bits of one part. The
  cold experts of the GPU steps are on the CPU with KQ_Q4X.
- The GPU (NP_GEMMA_GPU_Q4X=1, the default): the hot experts and the
  experts of a prompt pass use the copy too. The kernel of a step reads a
  group of 16 rows with one warp. The prompt pass gives int8 activations to
  the tensor cores (k_moe_gemm_q4x), also when NP_GEMMA_GPU_TC_MOE is 0. The
  26B with 2 GB of hot experts: pp2048 429 to 444 t/s, decode 46.8 and
  46.7 t/s. The gain is small, because the experts are not the limit on the
  GPU. The logits agree with those of the CPU as well as those of the old
  kernels (scripts/check_gpu_split.py passes).
- On the GPU, the dense products of a prompt pass of the 26B also take int8
  x (k_gemm_q8). NP_GEMMA_GPU_I8_MOE=0 turns it off. The attention stays float32, because
  float16 has overflow in the global layers. pp2048 goes from 1010 to
  1628 t/s.
- A prompt of the GPU now goes in chunks of 2048 tokens. Then the copy of
  the cold experts is hidden behind the other work of the layers, also with
  2 GB of hot experts. The limits of a prompt pass are then the attention
  (53%) and the dense products.
- The attention of a prompt pass of the 26B now runs on the tensor cores
  (k_flash_qc_tc). The kernel is k_flash_f32h for the int16 cache. It copies
  the int16 rows and their scales with cp.async. The query heads of a key
  head share them.
- The queries and the keys have a norm, so float16 has no overflow. The
  attention of a group of 2048 goes from 1325 ms to 128 ms. Its
  result has the bits of k_flash_tc<0> (a prompt of 8192 tokens).
- Then the copy of the cold experts is the limit again: FETCH_WAIT takes
  38% of a group. Chunks of 4096 do not fit with the default hot experts.

The 26B on the GPU, default hot experts:

    prompt       float32 attention   tensor cores
    4096         1312 t/s            1943 t/s
    8192         1022 t/s            1929 t/s

#### The decode of the 26B on the GPU

An nsys trace of the decode (600 tokens of context, default hot experts)
gave these changes:

- The hot experts (k_hot_gu, k_hot_dn) had one warp for each KQ_Q4X group
  of 16 rows. Half of the blocks of the grid had no group. Now a block
  of 8 warps takes one group, and each warp takes one eighth of the
  columns. The gate and up rows go from 70 to 42 us for each layer, and the
  down rows from 31 to 23 us. The step goes from 15.4 to 14.2 ms.
- The attention of a step used k_attn_part, which has three phases and
  keeps the scores in memory. The new kernel k_attn_fdt is k_attn_fd (one
  pass) for the shapes of the 26B. These are head_dim 256 with 2 query heads for each key
  head, and 512 with 8. The
  attention of a step goes from 1.2 to 0.67 ms at 600 tokens. At 8192
  tokens the step goes from 16.4 to 14.7 ms.

The step is then about 14 ms. The GPU works about 11 ms of it. The rest is
waits:

    the GPU waits for the cold experts of the CPU    about 1.3 ms
    the launch of the segment after each join        about 0.9 ms
    between the steps (the head, the logits)         about 1.5 ms

About 2 of the 8 experts of a layer are cold. The CPU takes about 60 us for
one (3.35 MB at the rate of the memory). The dense matrices read 926 MB for
each token at 81% of the rate of the GPU memory.

#### The step as one graph, and the fused norms

The step handed the experts of the CPU over with boundary records
(GP_TO_HOST, GP_CPU_JOIN, GP_TO_DEV). The host ran them between the graphs
of the step, so it launched a graph after each join. Now the step uses
flags in pinned memory, and it is one graph:

- GP_SIGNAL: a kernel writes the input of the CPU part to pinned memory.
  Then it writes the value of the run (the slot seq) to the flag in.
- GP_CPU_TASK: no kernel. After the launch, the host waits for the flag in,
  runs the CPU program, and writes the flag out.
- GP_AWAIT: a kernel waits for the flag out, then copies the output of the
  CPU part (loads with no cache). Its last operand is the count of the cold
  experts. With 0, it writes zeros and does not wait.

A copy node of a graph costs more than a kernel, so the kernels copy the
data. NP_GEMMA_GPU_FLAGS=0 gives the boundary records.

The step of the GPU uses the fused forms of layer_form. The norms and the
add of the end of the attention go in GP_ADD_NORM. The norms, the adds, and
the scale of the end of the layer go in GP_FFN_OUT.

GP_FFN_OUT also gives the input norm of the next layer. A layer then has 2 norm kernels, not about 10. The
CPU keeps the separate records. NP_GEMMA_GPU_FUSED=0 turns it off.

With a fixed set of hot experts, the logits of the four forms have the
same bits. The default set depends on the free memory of the GPU, so two
runs can differ. The 26B, 7 GB of hot experts, 600 tokens of context:

    form                        forward   step with the head
    boundary records            12.10 ms  14.78 ms
    flags                       11.72 ms  14.44 ms
    fused norms                 11.00 ms  13.58 ms
    flags and fused norms       10.76 ms  13.32 ms

The barrier of the parts of a CPU step (np_gemma/parts.py, GP_XBAR) now
uses flags too. Each part has a flag on a cache line of its own, and it
stores the count of its barriers there. It then waits until the flags of
the other parts have that count. Before, all the parts added to one shared
count.

A step of 2 parts crosses 150 barriers. On jackal (one NUMA node) it
goes from 42.0 ms to 40.8 ms. The bits stay the same (scripts/check_parts.py).
gemma_xbar_stats gives the time that each part waits.

The runner still waits for the GPU work before each CPU part (about 200 us
for each layer). The CPU part (about 137 us, about 2 cold experts) is longer
than the GPU work that runs with it.

#### The mixed groups of a prompt on the GPU

Each group of a prompt copied all the cold experts of each layer to the GPU.
The fast attention made that copy the limit. A mixed group
(SplitCompiler.moe_group_mix) copies only some cold experts. The CPU computes
the other cold experts during the GPU experts:

1. Before a group, ModelGPU.plan_mix selects the cold experts to copy for
   each layer: the ones that the routers of the group before selected most.
   The first group uses the file of counts. The copies go on two layers
   ahead, as before.
2. GP_HOT_SPLIT_MT gives the pairs of the experts that are not on the GPU
   to the CPU. GP_MOE_GPU gets -1 for them. The CPU computes its
   pairs with the KQ_Q4X experts and int8 activations, as the prompt of the
   CPU does. It uses a helper thread (GP_CPU_START). It starts before the wait
   for the copies, and it quantizes only the rows with a CPU pair.
3. The count for each layer comes from a model of the time of a layer:

       max(copies, other work + max(GPU experts, CPU part))

   The CPU part costs a fixed time, a time for each expert (the read of its
   weights), and a time for each pair. Where the model is flat, the plan
   takes the most copies, as a margin.
4. Each mixed group measures the model again (ModelGPU.calibrate_mix). The
   copy thread times its copies. Events on the stream time the waits for
   the copies and for the CPU part, and the work between them. The helper
   thread times the CPU part. A least-squares fit gives the three costs of
   the CPU part. Thus the plan follows the machine and other load on the CPU.

The plan of the Qwen models (after the router of each layer) does not suit
the 26B. A group of 2048 tokens uses 47 of the 53 cold experts of a layer.
Also, that plan cannot copy during the other work of the layer. A simulation gave 7%
for it and 20% for this form.

The 26B, default hot experts, README text. Each prompt came after another
text, so the plan of its first group comes from that text:

    prompt       all copied    mixed, fixed model   mixed, calibrated
    1024         811 t/s       1645 t/s             1810 t/s
    2048         1550 t/s      1949 t/s             2219 t/s
    8192         1578 t/s      1917 t/s             2138 t/s

The calibrated plan copies about 14 of the 53 cold experts of a layer. The
best fixed count was 15. With 2 GB of hot experts it copies about 37 of 108.
Then pp8192 goes from 1164 to 1784 t/s (the best fixed count, 40, gave 1762).

With six other busy processes on the CPU, it copies about 17. On the chat
text in groups of 512, the KL to float32 stays at 0.0022 to 0.0024. The CPU
part uses the CPU during a prompt, so other load on the CPU slows it.

The table gives the effect of the int8 activations on the 26B. The text is a
chat prompt and the answer of the model (1488 tokens):

    form                             ppl     KL       top token   prompt
    float32 (CPU)                    1.714   -        -           46 t/s
    CPU, int16 (INT4_Q8=16)          1.713   0.0000   99.9%       67 t/s
    CPU, int8 (default)              1.726   0.0026   98.0%       186 t/s
    GPU, float32 experts             1.713   0.0003   99.7%       518 t/s
    GPU, int8 experts (KQ_Q4X)       1.716   0.0012   99.5%       771 t/s
    GPU, int8 experts and dense      1.724   0.0020   98.9%       1214 t/s
    GPU, and float16 attention       1.723   0.0024   98.5%       1404 t/s

KL is the mean KL(float32 || form) over the 64 most probable tokens. On the
text of this README (a prompt with no chat turns), the model is not sure of
many tokens: 41% of the top tokens have a probability less than 0.3. There
the int8 forms agree for only 83% of the top tokens, but the perplexity does
not change (261.5 and 261.8; llama-perplexity gives 261.2). Where the top
token has a probability more than 0.7, int8 agrees for all tokens. No one
product causes the difference: int8 in the input of any one product type
gives about 88%.

Thus measure agreement on text of the kind that the model reads in use.

The 26B on the CPU (llama-bench method, 18 threads; llama.cpp
in the same session):

    runtime                         pp512       tg128
    numpy-gemma before KQ_Q4X       71 t/s      16.5 t/s
    numpy-gemma                     166 t/s     19.2 t/s
    numpy-gemma, NP_GEMMA_FLASH=1   231 t/s     19.1 t/s
    llama.cpp                       105 t/s     19.6 t/s

The attention of the prompt was then the largest part. The flash kernel
(NP_GEMMA_FLASH=1) is now the default. It takes 0.8 s of a prompt of 512
tokens, and the batched matmul 1.5 s. A prompt of 16384 tokens: 105 t/s
with the flash kernel, 51 t/s with the batched matmul, 74 t/s for
llama.cpp.

#### Two ideas that the measurement does not support

**A larger expert read does not help.** The 8 experts of one layer are 8
blocks of 2.23 MB. One test read one contiguous region of 17.8 MB in place of
the 8 blocks of the same total size. The rates are 37.7 and 38.8 GB/s. The
reason is below: the kernel is limited by its instructions, so a longer read
does not raise the rate.

A second test used 16 experts in place of 8. The read went from 17.8 MB to
35.7 MB, and the rate went from 38.9 to 39.9 GB/s. The order of the experts
also does not matter: 8 scattered experts and experts 0 to 7 both give
39 GB/s.

**A chunked prompt does not help either.** The C flash kernel walks only the
keys that the sliding window leaves visible. It already skips the keys that a
chunk can drop. The kernel is ahead of the batched matmul at every length:
1.08 times at 512 tokens, 1.25 times at 1024, and 1.48 times at 2048.

#### What llama.cpp does differently

llama.cpp repacks a Q4_0 tensor when the model loads. One block holds the
scales of 8 rows and the nibbles of those 8 rows in pieces of 8 bytes. The
nibbles become signed values. The buffer type is "CPU_REPACK". The multiply
then uses an int8 kernel that covers 8 rows for each pass, with the activation
as Q8_0.

This project uses a float32 kernel over 4 rows for a decode step. It also has
an int8 kernel, but that one puts 16 *tokens* in the lanes of a tile. For one
token the lanes hold one token and 15 empty ones. The kernel is then 2.2 times
slower than the float kernel. It gives 17.3 GB/s against 38.6 for the maps of
the 26B. It has the wrong shape for a decode step.

The int8 form of llama.cpp was put into this project and measured. The result
is that the shape is **not** the answer. The new kernel
`gemma_int4_q8_gemv` gives 1.0 to 1.2 times the rate of the float kernel. It is
not the 1.5 to 1.9 times that an earlier test gave.

Two smaller tests belong with this list:

*   **The work between the calls is not the cost.** 30 small maps read in a
    loop give 32.3 GB/s. The same loop with a norm and an add between the calls
    gives 26.6 GB/s, and the difference is the time of the norm itself. Thus
    the read keeps its rate.
*   **llama.cpp does not use a prefetch instruction.** The repacked kernel has
    no `_mm_prefetch` call, and it uses the AVX2 form rather than VNNI.

**An earlier reading of the two kernels was wrong.** Two faults made the int8
form look better than it is:

*   The first test did not empty the cache between the calls. It gave rates
    above the rate of a pure read for the same buffer. The data came from the
    cache, not from the memory.
*   The second test compared the int8 form with a float kernel that the author
    wrote for the test. That kernel was about two times slower than the float
    kernel of this project, `dot4_i4_f32`. A comparison against it says nothing
    about the code that runs.

The test `.cache/gemv_real.c` corrects both faults. It holds a copy of
`dot4_i4_f32`. It writes 512 MB between the calls. The speed uses the bytes of
the packed matrix and the bytes of the scale. The result at the map sizes of
the 26B model is:

    size         read   float   int8 8 rows   int8, no scale
     1.36 MB    38.4    25.0      30.4           33.4
     2.73 MB    51.5    30.8      29.3           37.0
     4.09 MB    48.3    39.1      39.1           42.7
     5.45 MB    61.4    41.2      42.7           46.5
     7.93 MB    50.0    42.9      48.8           49.3
    63.44 MB    56.5    52.6      52.9           55.5

The rates are GB/s. The column "int8 8 rows" is the kernel of this project.
The column "int8, no scale" leaves out the group scale, and it gives the upper
limit of the integer multiply. The noise of one row is about 10 per cent, so
one row can move by more than the difference between two columns.

The int8 form is 0.95 to 1.21 times the float form. The mean is about 1.04.

**The group scale is the reason.** The machine has hardware counters, so the
two kernels were measured directly. `perf stat` ran each inner loop with one
thread and a matrix of 5.45 MB, which stays in the 24.75 MB last level cache.
That setting removes the memory from the question and shows the work of the
instructions alone. The counters give:

    kernel            cycles/repetition   instructions   IPC   port0  port1  port5
    float, 4 rows             1,778,139      3,515,030  1.98    83%     5%    83%
    int8, 8 rows              1,631,647      3,317,532  2.03    55%    55%    60%
    int8, no scale              766,396      1,960,555  2.36    62%    63%    66%

The counter `uops_dispatched_port.port_N` gives the port columns. A port at 100
per cent is a port with no free slot in any cycle.

**The scale is the cost.** The int8 kernel without the group scale is **2.32
times** faster than the float kernel. With the scale it is only 1.09 times
faster. The scale therefore takes back the whole advantage of the integer
multiply.

The reason is the width. For each row and each group the int8 kernel runs four
steps at 256 bits. The steps are the subtraction of the nibble bias and the
change from int32 to float32. They also include the multiply by the scale of
the group and the fused multiply and add.

The float kernel applies its scale in the fused multiply and add that it
already runs. That operation is 512 bits wide, one time for the whole group.
Thus the scale work of the int8 kernel is about four times the
scale work of the float kernel for each value.

A second measurement agrees. The int8 kernel reaches the rate of a pure read
at the large sizes, and the float kernel reaches 86 per cent of it. That
difference, about 1.14 times, is the whole gain of a decode step.

**A note on the model.** `llvm-mca` reads the assembly of a loop and predicts
the ports and the cycles. Give it the two group loops:

    loop               instructions   cycles/group   cycles for 8 rows
    float, 4 rows                50          17.10               34.20
    int8, 8 rows                108          32.28               32.28

That is 1.06 times for the int8 form, and the counters give 1.09 times. The
model and the machine agree on the comparison.

The model does not agree on the port numbers. It says 95 per cent for the float
loop and 85 per cent for the int8 loop. The counters say 83 and 55 per cent.
The reason is that the model assumes that every load comes from the first level
cache. The real kernel waits on the memory, and the wait spreads the work over
more cycles, so each port is less busy. **Use the model for a comparison of two
kernels, not for the absolute port numbers.**

An earlier version of this section gave 1.50 times for the int8 form, from a
port table of 93 per cent. Those numbers were wrong. The tool took the wrong
block of assembly. For the float kernel it read the horizontal reduction at the
end of the kernel in place of the group loop. The reduction is shorter than the
loop, so the comparison had no meaning. The numbers above come from the markers
`LLVM-MCA-BEGIN` and `LLVM-MCA-END`, which name the loop itself.

The decode of the 26B model shows the same result. Each of the three int4
kernels has an int8 form, and each one is a little faster:

    stage                   float    int8    change
    mixture of experts       20.9    20.2     -3%
    four matrices, one call  17.2    15.8     -8%
    one matrix per call       9.5     8.8     -7%

The total step time does not show the gain. The four int4 kernels are about
half of a decode step. A cut of 6 per cent of them is therefore about 3 per
cent of the step. The machine noise between two runs of the model is plus or
minus 8 per cent. Five interleaved pairs of `tg96` gave a median of 11.14
tokens a second for the float form and 10.90 for the int8 form. The two are
the same within the noise.

The int8 form is therefore **off by default**. Set `NP_GEMMA_INT4_Q8_GEMV=1`
to use it. The code is in `gemma_int4_q8_gemv` and its three entry points
`gemma_int4_q8_gemv_x`, `gemma_int4_q8_multi4`, and `gemma_int4_q8_moe_gemv`.
The checks in `scripts/check_int4_q8.py` cover all of them.

**The interleaved layout is also not the answer.** A test of the two layouts
with the same access gave 1.0 to 1.2 times.

**What remains.** llama.cpp is 1.57 times faster than this project at the
decode of the A4B. The int4 kernel is not that difference. The stage report of
`.cache/kern_time.py` gives the parts that are left. The output head reads
605 MB of Q6_K data for each token. The 211 norm calls cost 4 ms of a 79 ms
step. Those are the next places to look.

#### The E4B decode at a long context

llama-bench measures the generation after a prompt of 512 tokens, so the
context is 513 tokens. The context changes the result:

    context     before     after
      20 tok   0.0862 s   0.0824 s
     512 tok   0.1308 s   0.0871 s

The profile showed that the work outside the matrix kernels grew by 31 ms at
the long context. Two measured parts of that growth were the attention
products (16 ms) and the key and value cache (6 ms).

**The attention product did not use the cache layout.** The batched matmul
needs a transpose of the key for each layer. At a context of 512 tokens that
copy costs more than the arithmetic. `gemma_attn_decode_f32` is a new C
kernel. It runs the scores, the softmax, and the output for one query token in
one call. It reads the cache in place.

The gain for one layer:

    layer type             head_dim   matmul    fused
    sliding                     256   274 us   151 us
    global                      512   673 us   405 us

**The cache copied its whole history.** `E4BCache.append` ran a `concatenate`
for each layer, which copied 63 MB for each token at a context of 512. The
cache now makes its buffers one time and writes each new key into its place.
The two buffers of a layer also keep a stride, so the kernel reads the part in
use without a copy. A copy of the key and the value for each layer costs
98 MB for a token at the same context.

**The two orders cost 2.4 times.** A cache of (keys, kv_heads, head_dim) makes
the hardware prefetch jump over the other head for each key. The cache now
keeps (kv_heads, keys, head_dim), so the keys of one head follow each other. A
measurement of the dot product alone gives 69 us for the first order and 29 us
for the second.

**The norm and the rope now use two calls.** Each layer gave the query, the
key, and the value their own norm, and then turned the query and the key with
their own rope. That is five calls and five OpenMP regions for each layer, and
the NumPy rotation also builds a second array for each call. `ops.qkv_norm`
and `ops.rope_apply` do the same work in two calls, in place.

The two fused kernels agree with the path they replace to the last bit. The
first form of the rope kernel did not. A fused multiply and add rounds one
time fewer than the two multiplies and the add of the NumPy form. The model is
sensitive to that difference over a long context: one token of a 1024-token
prompt changed. The attribute `fp-contract=off` on the kernel removes the
difference.

The decode step loses 4 to 5 per cent from this change, and the number of
calls for a token falls by 72. The prompt pass does not change.
`NP_GEMMA_FUSED_QKV=0` selects the older path for a comparison.

The generation at the context of the benchmark went from 8.52 to 11.20 tokens
for each second. `NP_GEMMA_ATTN=0` selects the older matmul path for a
comparison.

The machine also carries a `scripts/serve.py` process for the 26B model from
an earlier session. It uses about 160 per cent of one core, and two rounds of
the same measurement differ by 5 to 10 per cent.

## Qwen3.8-Flash-Next

The model is nvidia/Qwen3.8-Flash-Next-NVFP4 (the llama.cpp name is
qwen4exp). It has 48 layers of Gated DeltaNet and QSA attention, 512 experts
(10 for each token), and an MTP layer. `scripts/convert_nvfp4_gguf.py` makes
one GGUF file of this runtime (132 GB). The experts stay NVFP4 in groups of 16
rows, one layout for the CPU and the GPU. QWEN38_PLAN.md has the plan and the
history. HANDOFF_QWEN38.md has the state, the profiles, and the next steps.

The speed. The method is that of llama-bench: random tokens,
one warm-up, 3 reps (`scripts/bench_qwen4.py`). The machine has a Xeon
W-2295 (18 cores, 66 GB/s) and 188 GB of RAM. Its RTX 5060 Ti has about 8 GB
free, on PCIe Gen3 x8. Other programs ran on the machine.

    runtime                              pp512   pp2048  pp4096   tg128   tg512
    this runtime, GPU (0.5 GB hot)       327     496     493      22.8    23.2
    this runtime, CPU (dense q8)         93.1    -       -        7.61    -
    llama.cpp, GPU (-ngl 99 -ncmoe 48)   101     101     -        18.3    -
    llama.cpp, CPU                       27.9    -       -        5.0     -

The rates are tokens a second. llama.cpp uses the Q4_K_XL GGUF file of
Unsloth.

The long prompts depend on the free memory of the GPU. The cache of a long
context takes the room of the buffer of the expert copies.

    prompt           GPU memory                     dense   tok/s
    32768 tokens     about 8 GB free                q8      253 (a buffer of 0 to 3 experts)
    32768 tokens     13 GB free                     bf16    351
    131072 tokens    13 GB free (cache 5.6 GB)      q8      424

The buffer now keeps room for at least 64 experts (a group of 1024 at 32K:
2.8 s, not 3.8 s). The dense mode is q8 by default. The "auto" mode takes
bf16 when the GPU has much free memory, and bf16 is slower here.

- The GPU holds the dense part and 0.5 GB of hot experts. In the decode,
  the CPU computes the other experts. A prompt runs in mixed groups: the GPU
  copies the experts with the most tokens, and the CPU computes the others
  at the same time.
- MTP (3 drafts) on the CPU gives about 11.6 tok/s in place of 7.5. On this
  GPU it gives about the rate of the plain decode.
- A test of 32001 tokens of the source of this project ran on the GPU. The
  prompt ran at 319 tok/s and the decode at 19.1 tok/s. The answers were
  correct.

Run the checks and the measurement:

    G=models2/Qwen3.8-Flash-Next-NVFP4-GGUF
    python scripts/convert_nvfp4_gguf.py models/Qwen3.8-Flash-Next-NVFP4 \
        $G/Qwen3.8-Flash-Next-NVFP4-bf16.gguf
    python scripts/check_qwen4_st.py --path $G/Qwen3.8-Flash-Next-NVFP4-bf16.gguf \
        --tok $G/tokenizer.json --layers 4 --gpu --hot-gb 0.5
    python scripts/bench_qwen4.py -m $G/Qwen3.8-Flash-Next-NVFP4-bf16.gguf --backend gpu \
        --hot-gb 0.5 -p 512,2048,4096 -n 128,512 -r 3

A test of a long context has a prompt of about 128K tokens of the source
of this project. Then it asks questions with a long answer (up to 4096 new
tokens).
The answer streams to the terminal, and the rates go to the end of
long_qwen4_out.txt.

    python scripts/long_qwen4.py                          # 128K on the GPU
    python scripts/long_qwen4.py --tokens 32768 --gen 1024
    python scripts/long_qwen4.py --question "What does csrc/moe.c do?" --gen 512

The OpenAI compatible server uses the HTTP part of np_gemma/server.py. It
renders the chat template of the model (chat_template.jinja), with the tools.
It gives the <think> part as the reasoning. It gives the tool calls of the
model as the tool_calls of the API:

    python scripts/serve_qwen4.py --ctx 98304 --port 8081     # http://127.0.0.1:8081/v1

A request sets the think part with "reasoning_effort" (none, low, medium,
high) or "thinking"; --thinking gives the default (medium). The usage of a
response gives the prompt tokens that the cache held
(prompt_tokens_details.cached_tokens).

The security of the servers (serve.py and serve_qwen4.py):

- Each request needs the API key: "Authorization: Bearer KEY" or the
  header x-api-key. The key comes from --api-key, else NP_GEMMA_API_KEY,
  else "change-me" (set your own). --api-key "" turns the check off. /health needs no
  key.
- The server sends CORS headers only for --cors-origin. Without it, a web
  page cannot call the server from a browser.
- A body above --max-body-mb (64 MB) gets 413, and a negative
  Content-Length gets 400. The server does not read such a body.
- The raw answers (last-answer.txt, empty-*.txt) go to files only with
  --debug DIR, in DIR, readable by the owner only.
- max_tokens below 1 gets 400. serve_qwen4.py raises a request of 8192
  tokens or more (--max-tokens-floor-min) to --max-tokens-floor (32768).
  A smaller request (a title, a summary) keeps its limit. The headers
  X-Max-Tokens-Applied and X-Max-Tokens-Requested give a change.
- A stream parses a long answer again only every 1 + n / 2000 tokens. A
  parse of 30000 tokens takes 33 ms, and it holds the GIL that the model
  thread needs.
- In serve_qwen4.py, a CUDA error that leaves the GPU unusable ends the
  process with exit 70. Then scripts/serve_forever.sh starts the server
  again (after any exit that is not 0).

The option --mtp N gives N drafts of the MTP layer for each
round of Qwen3.8 on the GPU. The prompt also fills the cache of the MTP
layer. A draft stays only when the sample of its row picks it, so the text
has the distribution of the settings. With the hot experts fixed, a greedy
answer of 106 tokens is the same with and without --mtp 3:

    setting                       plain        --mtp 3
    greedy, thinking off          21.9 tok/s   27.8 tok/s (68% of drafts)
    temperature 0.7               -            19.5 tok/s (40% of drafts)

### The methods of the 26B on the Qwen models

The changes to the 26B on the GPU in this round, and the Qwen models:

    method                         26B    Qwen3.6          Qwen3.8
    the step as one graph (flags)  yes    yes (new)        yes (new)
    skip when no expert is cold    yes    yes (new)        yes (new)
    mixed groups of a prompt       yes    yes (new)        yes (before)
    groups of 2048 rows            yes    yes (new)        yes (new)
    the plan tuned                 yes    new default      as before
    one-pass decode attention      new    as before (fd)   QSA
    prompt attention, tensor cores new    as before        QSA
    int8 dense products (prompt)   new    as before        as before
    fused norms of a step          new    GP_ADD_RMS       GP_ADD_RMS
    hot experts, a block a group   new    (rows)           no (see below)

- The flag form is in QwenGPU._moe_split, so the steps and the small
  groups (the MTP verify group) of both models use it. The verify of an
  MTP round of Qwen3.8 went from 103 to 84 ms. A verify group still gives
  the bits of steps.
- The mixed groups moved from Qwen4GPU to QwenGPU. Qwen3.6 went from 547 to
  960 tok/s at pp2048 (groups of 1024 rows copied all the cold experts).
  Groups of 2048 rows: Qwen3.8 pp8192 from 402 to 491 tok/s, Qwen3.6 from
  754 to 932.
- The costs of the plan (GP_MOE_PLAN) now come from desc when the host
  sets them. A fit of the measured costs gave a worse plan: the copies come
  from pageable memory, so they use the CPU too. A search on the sum of the
  waits found the best cost of a copy for Qwen3.6 (about twice the default).
  It made Qwen3.8 slower.
- The default of the K quants now has that ratio: Qwen3.6 pp8192 1019 to
  1032 tok/s, pp2048 1010 to 1073. NP_GEMMA_GPU_MIX_CAL=1 turns the search
  on.
- The hot experts of Qwen3.8 (KQ_NVX) keep one warp for each group. A block
  for each group (as for the 26B) did not change the decode. At 1 GB of hot
  experts, the cold experts on the CPU limit it. The block also changed the
  order of the sums. Then an expert on the GPU did not give the bits of the
  CPU, and the tokens changed with the set of hot experts.
- The runner records all the graphs of a program before it launches one
  (gg_exec). The first use of a kernel can make the driver wait for the
  GPU. A graph on the GPU can wait for a task of the CPU, and the host runs
  the tasks after the launches. Thus a record after a launch can wait for
  a task that waits for the record.

### Images on Qwen3.6 and Qwen3.8

Qwen3.6-35B-A3B and Qwen3.8-Flash-Next have the same image encoder, the
Qwen3-VL ViT. It has 27 layers of 1152, 16 heads of 72, patches of 16, a
merge of 2 x 2, and no deepstack layers. Only the output width is different
(2048 and 2560). The file np_gemma/vision_qwen.py reads the mmproj GGUF of
Qwen3.6. For Qwen3.8 it reads the tensors model.visual.* of the checkpoint
(BF16 in the NVFP4 files too). The graph follows transformers
(qwen3_5_moe and qwen4_exp; the two vision graphs are the same):

- The image: bicubic to sides that are multiples of 32 (smart_resize), and
  values 2 (x / 255) - 1. The patches go in the order of the 2 x 2 blocks.
- The patch linear: the Conv3d of two equal frames is the sum of its two
  kernels. Then the 48 x 48 position table, bilinear with align corners.
- Each layer: LayerNorm, q k v with bias, the 2D RoPE (theta 1e4),
  attention with scale 1 / sqrt(72), LayerNorm, and the FFN with gelu tanh.
- The merger: LayerNorm, the 4 rows of a block as one row of 4608, and two
  linears with gelu (erf) between them.

The encoder is a program with two new records: ENC_LNORM (LayerNorm with
bias) and ENC_GELU (the tanh or the erf form). ENC_ROPE2D of gemma4v serves
the RoPE. For it, the rows of q and k of each head get the order of that
record (quarters 0, 2, 1, 3). The scores do not change, because q and k
get the same order.

The text side:

- The template writes <|vision_start|><|image_pad|><|vision_end|>. Then
  media.expand_qwen makes one <|image_pad|> for each row of the image.
  The rows take the place of the embeddings. The tokens of an image are
  causal.
- M-RoPE (media.mrope_positions): a token of an image has the positions
  (p, p + row, p + column). The text after the image continues at p +
  max(rows, columns), not at the index of the row. QwenCache.set_rope keeps
  the positions of each row of the cache, and the rows after the prompt
  continue from the largest one + 1. Qwen.rope takes positions (3, t), with
  sections [11, 11, 10]. A frequency pair j takes the row if j % 3 == 1,
  the column if j % 3 == 2, else the time. A prompt of text only keeps the
  old path: the same bits as before.
- Qwen3.8, QSA: the indexer gives each block of 4 keys the RoPE of its
  first row. After an image, that row has an M-RoPE position. Thus the
  record QSA_SELECT takes the positions of the rows (operand 22). They are
  (rows, 3) int32, or 0 for text only, on the CPU and on the GPU.

The checks:

    scripts/check_mm_qwen.py (against transformers)
      Qwen3.6 weights (334 tensors)        equal to the GGUF
      Qwen3.6 image rows, CPU program      mean 1e-5, max 1e-4
      Qwen3.6 image rows, GPU program      mean 2.3e-3 (x in float16 on the tensor cores)
      Qwen3.8 image rows, CPU / GPU        mean 1e-5 / 3.1e-3
      M-RoPE positions (get_rope_index)    equal; cos and sin 4.4e-7
    scripts/check_qwen_mm_prompt.py (CPU program against GPU, test-1.jpeg)
      Qwen3.6                              KL 0.009, same top token
      Qwen3.8                              KL 0.0006, same top token
    QSA block keys after an image (2624 tokens)
      CPU and GPU against the keys of transformers    3e-7
      the same without the positions                   549 of 549 blocks wrong

Both models read the front page (the headline, the date) and give the year
to a second question after the answer. Q8_0 weights in this encoder give
rows with 4% to 9% error (mean), so the Qwen encoder keeps BF16.

                                  CPU program   GPU program
    encoder, 936 patches          1.6 s         0.16 s
    Qwen3.6 prompt, 259 tokens    3.1 s         1.1 s
    Qwen3.8 prompt, 259 tokens    58 s          4.9 s

The server scripts/serve_qwen4.py loads Qwen3.6 or Qwen3.8, as the
architecture of the GGUF says. The option --mmproj gives the encoder: the
mmproj GGUF of Qwen3.6, or models/Qwen3.8-Flash-Next-NVFP4. The options
--image-budget (1024 tokens) and --media-dir are as in serve.py. The encoder runs on the thread of the
model. The cache compares the keys of the tokens, so another image of the
same size is not a hit. On Qwen3.6 an image and a question take 2.2 s,
then the answer comes at about 31 tok/s.

The cache of a chat. The DeltaNet state cannot go back, so
the server reuses the cache only from its end or from a snapshot. Before,
it kept one snapshot, at the end of the last prompt. But the template of
the next turn writes the earlier answer in another form than the model
wrote it:

- Qwen3.6 drops the think part of the earlier answers. It keeps them only
  with preserve_thinking, or after the last user message (a tool loop).
- Qwen3.8 keeps them by default (preserve_thinking undefined is true), but
  only with the reasoning_content that the client sends back.

Thus each follow-up read the whole chat again. Now the server keeps the
last 4 snapshots. One of them is after the last <|im_start|>assistant\n of
each prompt. A prompt uses the longest one that fits. The option
chat_template_kwargs.preserve_thinking goes to the template. A chat with an
image and three follow-ups (the new tokens against the prompt):

                                 Qwen3.6                  Qwen3.8
    thinking off                 cached 325/349, 345/373  370/391, 401/419
    thinking on                  cached 325/376, 374/409  351/421, 419/458
    thinking on, reasoning back  cached 325/376, 374/409  462/481, 550/566

A follow-up with thinking off now takes 0.4 to 1.1 s on Qwen3.6.

Video on Qwen3.6 and Qwen3.8. The processor of transformers
(Qwen3VLVideoProcessor, Qwen3VLProcessor) does this:

- Frames: 2 for each second, at least 4 and at most 768 (np.linspace over
  the frames, rounded). An odd count gets the last frame again.
- Size: smart_resize over all the frames. With cap_pixels_per_frame (the
  reference of qwen-vl-utils; the default of transformers from v5.22), a
  frame has at most max_video_tokens tokens.
- Pairs: two frames make one temporal patch. The Conv3d then sees two
  different frames, so the patch linear of video has the two kernels
  (1536 values), not their sum. The patches of a pair see only each other.
- Text: the <|video_pad|> of the template becomes <T seconds><|vision_start|>
  pads <|vision_end|> for each pair. T is the mean time of its two frames.
  M-RoPE treats each pair as an image.

np_gemma/vision_qwen.py follows it. video_frames decodes only the sampled
frames (PyAV), and encode_video runs the program of a pair for each pair.
The server takes video and video_url parts. The option --video-budget
gives the most tokens of a pair (128), and --video-frames the most frames
(32). The script check_mm_qwen.py --video compares with transformers. The
clip has 9 s: 18 frames, 9 pairs of 9 x 13 tokens.

                                  Qwen3.6               Qwen3.8
    frame indices, grid, pixels   equal                 equal
    rows, CPU / GPU program       mean 1e-5 / 1.8e-3    mean 1e-5
    prompt ids, M-RoPE positions  equal                 equal
    encoder, GPU / CPU            0.6 to 0.9 s / 7 s    - / 6.7 s

The answers place each slide at its time (1 s, 4 s, 7 s). The last slide
starts at 6 s, and the answers say so.

Qwen3.8 fills the GPU: 5.3 GB of weights and 3 hot experts in each layer,
3.7 GB of cache at 98K, and about 4 GB of programs and copies. The desktop
takes 3.3 GB of it too. Thus the tests of video found three faults of
the GPU memory of the Qwen servers, and a client found a fourth. They were there for text too; prompts of
many sizes and media only made them come sooner:

1. The buffer of the copies of the mixed groups (mix_ring) takes all the
   free memory but MIX_KEEP. A program of a new size then had only that.
   Now a new program frees the buffer first (QwenGPU.free_ring), and the
   next mixed group makes it again with what is left.
2. Each size of group kept its program for good. When a program does not
   fit, the programs used least recently now go (_evict, with the device
   copies that only they use). Before a buffer of the copies with less
   than MIX_RING_MIN experts, programs go as well.
3. A cudaMalloc that failed and that the code handled left its error in
   the runtime. The next run then reported "cudaGetLastError(): out of
   memory" with 1.7 GB free. gg_clear_error now clears it. Also, a new
   program records and uploads its CUDA graphs when it is made
   (gg_prepare), not at its first run.

4. The scratch of QSA_SELECT (Qwen3.8) had nbmax keys for each query, with
   nbmax from the size of the context. At 98K a mixed group of 2048 rows
   took 403 MB, in each program, when bind() ran. A turn of 8500 tokens of
   a client with 27 tools then failed. Now nbmax follows the rows of the
   run ((pos + t) / 4 + 1), and one buffer serves all the programs
   (QwenGPU._alloc makes room for it).

The GPU classes of the Gemma models (ModelGPU of the 12B and the 26B,
E4BGPU) had fault 2 too. A 12B server with MTP failed at a prompt of 21674
tokens of an agent harness ("cudaMalloc of 16777216 bytes failed"). It
held a program for each size of group and then had 11.8 GB of the GPU.
Now they use gpu.ProgramLRU. When a new program, a larger cache, or the
program of the drafter does not fit, the group programs used least
recently go. A new program also records its graphs when it is made.

In a test, a dummy buffer left 4 GB of the GPU free. Six prompts of 173 to
4123 tokens with MTP then evicted programs and made them again. Their
tokens were the same as those of a run with all the memory.

After the fixes, 3 rounds of the tests (long text, video, video and image,
chats; 60 requests) pass at 98K on both models. Before, a prompt of 1800
tokens failed on Qwen3.8 at 98K with no media at all. By default
(--mmproj-gpu auto) the encoder of Qwen3.8 stays on the CPU. Its 1.3 GB on
the GPU have no room next to the programs of the model. The encoder of
Qwen3.6 stays on the GPU. The server encodes a small image at the start,
so its weights are there before the buffers of the model.

--mmproj-gpu reserve puts the encoder of Qwen3.8 on the GPU in
memory reserved before the model sizes its hot experts
(QwenEmbedder.reserve_gpu): the weights and the program of an image of
max(--image-budget, 1120) tokens, 1.58 GB. A smaller image has a smaller
program, and a buffer then holds the rest of the room between media
requests, so the programs and the buffer of the copies of the model cannot
take it. A request gets at most that many tokens. On the 3090 at 98K: 60 hot
experts in each layer in place of 72, the 8K prompt 688 tok/s (675 without),
an image on the GPU in 0.2 to 0.3 s (2.2 s at 1120 tokens; the CPU 2.2 s at
any size), with the buffer of the copies and the MTP layer left in place.
Also: a GPU encoder program of a new size used to keep the buffers of the
old sizes (in the mirror of all sizes) until release; _Runner now frees
them, and frees the old program before the new one takes its memory.

### Images, video, and audio on the E4B and the 26B

The E2B, the E4B, and the 26B have real encoders (gemma4v; gemma4a on the
E2B and the E4B). The file np_gemma/gemma4_encoders.py reads them from the
mmproj GGUF and follows transformers:

- Vision: patches of 16 x 16, and a 2D RoPE of theta 100 in each layer.
  The E4B has 16 layers of 768 and the 26B 27 of 1152. The attention of
  all patches runs on the C flash kernel. Then the average of each 3 x 3
  block, and the std_bias and std_scale of the 26B.
- Audio: the mel, two convs of stride 2, and 12 Conformer layers (local
  attention of 12 keys with relative positions, the conv module).
- The linears of the E4B clamp their input and output (use_clipped_linears).

The 26B has no audio encoder (audio_config is None); it takes images and
video. The E4B text side gives a soft token the per-layer row of the pad
token (id 0), as transformers does. The E4B is causal; on the 26B the
tokens of an image see each other.

The checks against transformers (scripts/check_mm_gemma4.py, the
unquantized QAT safetensors):

    E4B weights (931 tensors)          equal to the GGUF (the converter changes undone)
    E4B image rows                     1.8e-4 and 4.6e-6 (relative)
    E4B mel, audio rows                exact, 4.4e-6
    26B weights (356), image rows      equal; 7.3e-5 and 3.6e-5
    E4B end to end, int4 on the GPU    image KL 0.007 (38/40), audio 0.0020
                                       (48/48), text alone 0.020

The 26B vision tensors come from the shard of 50 GB by range requests
(1.15 GB). A full reference of the 26B is too large, so llama.cpp
(llama-mtmd-cli, the same GGUF files) is the reference there. The thought of
both on the test image gives the same headline and date and the same first
draft sentence.

Three faults of the tools came out:

1. The mmproj of unsloth for the 26B comes from the model before QAT: its
   vision weights differ by up to 45%. Use the mmproj of
   google/gemma-4-26B-A4B-it-qat-q4_0-gguf with the QAT GGUF.
2. With eager attention, transformers hides the keys that an audio query
   sees. The mask is additive there, and the audio attention masks where
   attention_mask.logical_not() is True. The references use sdpa.
3. The C flash kernel (cops.attn_prefill) gives nonsense for a head of 72
   values (the 26B vision). The encoder pads the head to 80 with zeros.

The encoders are programs (np_gemma/program.py), as the steps of the text
models are. The records ENC_* run in C on the CPU, in one OpenMP region
where the rows split over the threads. When the model is on the GPU, they
run as one CUDA graph there.

NP_GEMMA_MEDIA_GPU=0 keeps them on the CPU, and
NP_GEMMA_ENC_PY=1 runs the NumPy layers (the reference). A program has the
size of its input: the patches of an image, or the rows of a clip. The last
four sizes stay compiled, and the GPU programs share one copy of the
weights. The records:

- ENC_LINEAR: x times a bfloat16 or float32 W, for any sizes (the 26B FFN
  has 4304). It applies the clamps of a clippable linear and a bias. The
  float32 matrices of the mmproj are upcasts of bfloat16, so they become
  bfloat16 with no loss.
- ENC_RMS, ENC_ADD (with a scale), ENC_GELU_MUL, ENC_SILU, ENC_MUL_VEC,
  ENC_GLU: rows and values.
- ENC_ROPE2D: the axial 2D RoPE of gemma4v, in place.
- ENC_ATTN: the attention of all patches. On the CPU it calls the flash
  kernel of the prompt in the region of the program
  (gemma_attn_prefill_region). The tasks of the AVX-512 and AVX2 versions
  are now functions of their own. On the GPU a warp takes a query and a
  head, with tiles of 32 keys.
- ENC_DWCONV and ENC_LOCAL_ATTN: the causal depthwise conv and the local
  attention of gemma4a (12 keys, the relative position term, the soft cap).

The mel and the subsample convs of the audio stay on the host (the front
end, about 70 ms).

The script check_mm_gemma4.py compares with transformers, with and without
--gpu. The E4B image rows agree to 1.7e-4 and 7.1e-6 (GPU 2.9e-4 and
4.9e-6), and the audio rows to 2.9e-6 (GPU 2.8e-6). The 26B image rows
agree to 3.4e-5 and 3.7e-5 (GPU 4.0e-5 and 3.9e-5). The script check_mm_prompt of
the E4B gives image KL 0.007 and audio 0.0020, as before. The checks
check_flash_c and check_program pass after the change of the flash kernel.

On the CPU the linears read W in groups of 16 rows (KQ_BF16X16 and
kq_x16f_body, now with an AVX2 core too). Each column is one fma for each
of 16 tokens (6 on AVX2). The products of an image take half the time or
less: E4B 1736 to 836 ms, 26B 5294 to 1961 ms (AVX-512, 18 threads), and
5684 to 2935 ms (AVX2, 6 threads). NP_GEMMA_ENC_X16=0 keeps the rows.

On the GPU the linears with bfloat16 W run on the tensor cores. The kernel
k_enc_gemm_tc is k_gemm_bh for any cols of 8k values; x becomes float16
with the input clamps. The rows then differ from transformers by a few
percent at most (mean 1.7e-3 on the E4B, 5.2e-3 on the 26B). The logits of
the E4B do not change: image KL 0.0064 (39/40), against 0.0070 (38/40) with
the float32 kernel. NP_GEMMA_GPU_ENC_TC=0 keeps the float32 kernel.

                               NumPy      CPU program   GPU program
    E4B image (280 tokens)     3.9 s      2.7 s         0.35 s
    E4B clip of 17 s           2.6 s      0.74 s        0.14 s
    26B image (280 tokens)     11.7 s     6 to 8 s      0.92 s

The 26B on the server: an image and a question take 4.3 s. The 9 s video
takes 12.1 s (192 s with the CPU encoder), and a new question on it 0.7 s.

### 8-bit weights in the encoders

Gemma4Embedder(q8=True) gives each linear of the encoders with cols of 32k
values Q8_0 weights (cops.kq_to_q8_0). The only exception is the 26B FFN
down matrix (4304 cols), which keeps bfloat16. On the CPU the weights are
in groups of 16 rows (KQ_Q8X16). An encoder linear then takes the int8
records of the GGUF products: ENC_CLAMP (the input clamps), KQ_QUANT and
KQ_LINEAR, and ENC_BIAS_CLAMP (the bias and the output clamps). The server
takes --mmproj-q8 (auto: on for the E2B and E4B).

KQ_Q8X16 now also has an AVX2 kernel (the i5). It multiplies |x| by w with
the sign of x (vpmaddubsw, then vpmaddwd), so no int16 sum saturates.
Before it, the AVX2 path of Q8_0 took 98.7 s for an image.

Two faults came out:

1. The kernels k_enc_* did not start with PDL_START. In a CUDA graph a
   kernel can start before the kernel before it ends, and must wait with
   griddepcontrol.wait. Thus ENC_BIAS_CLAMP read the rows of KQ_LINEAR
   before they were all written.
2. The widened guard of KQ_Q8X16 also took kq_tiles, which exists only
   with VNNI. The AVX2 library then did not load.

The E4B with Q8_0 encoders (check_mm_prompt, int4 on the GPU):

                                 bfloat16          Q8_0
    image KL (top token)         0.0064 (39/40)    0.0063 (38/40), CPU encoder 0.0073
    audio KL                     0.0020 (48/48)    0.0020 (48/48), CPU encoder 0.0020
    the weights of the linears   912 MB            485 MB
    image, CPU (AVX-512, 18)     1.58 s            1.33 s
    image, CPU (AVX2, 6)         6.27 s            5.76 s
    image, GPU                   0.37 s            0.43 s
    audio, CPU (AVX-512, 18)     0.50 s            0.32 s

The rows of Q8_0 differ from transformers by 1% to 4% (mean), the CPU more
than the GPU. The answers do not change.

### Video on the Gemma 4 12B

The 12B has no video encoder. A video is a series of frames. Each frame is
an image with a budget of 70 soft tokens, as in the video processor of
transformers:

- 32 frames: the indices 0, N/32, 2N/32, and so on of the N frames,
  truncated to integers (np_gemma/unified.py frame_indices).
- Before each frame, its time as mm:ss and a space. Then <|image>, a
  <|video|> (id 258884) for each soft token, and <image|>. A space joins
  two frames (media.video_text). The time is the frame index over the
  frames per second.
- The tokens of one frame see each other. The time text between two frames
  ends the block, so a frame does not see a later frame.
- The video has no audio track in the model. Send the sound as an audio
  part.

PyAV (the package av 19.0, with its own FFmpeg) decodes the video. It is in
the venv now. The library transformers decodes a video path only with
torchcodec. Thus the reference (scripts/hf_mm_reference.py --video) decodes
it with the PyAV reader of transformers and gives the frames to the
processor.

The test clip is 9 s (216 frames at 24 frames per second): three images of
llama.cpp, 3 s each. The command is in scripts/check_mm_prompt.py. On it:

    the prompt ids of the server (the chat template, the times, the frames)
        the same as transformers: 2325 ids, 32 frames of 63 tokens
    float32 (the same weights), the soft rows of the reference   KL 0.00000, 44/44
    int4 on the GPU, our soft rows                                KL 0.0178, 41/44
    the prompt pass on the GPU                                    1.82 s
    the server: the video and a question, 94 tokens out           7.0 s
    the same video, a new question                                0.3 s

The server takes video_url (or video): a data URI, or a path under
--media-dir. The options --video-frames and --video-budget change the 32
frames and the 70 tokens of a frame.

A cut of the cache after a turn often found the rows of a window gone. The
MTP drafter reads two layers on the host, and those layers drop rows at
each step. Then the Session read the whole history again: 5 s for each
question on the same video.

Now a layer with a window keeps KV_KEEP more
rows when it drops rows (NP_GEMMA_KV_KEEP, 1024 by default). A cut back of
up to KV_KEEP tokens is then always possible. The script check_gpu_session
(the 26B) now reuses the cache for the cut of 1300 too, with the same bits.
The script check_mtp gives the ids of the plain decode.

### Images and audio on the Gemma 4 12B

The 12B (the "unified" model) has no image encoder and no audio encoder,
as MULTIMODAL_PLAN.md tells. The file np_gemma/unified.py reads the mmproj
GGUF of google:

- An image becomes patches of 48 x 48 pixels (70 to 1120 of them, 280 by
  default). Each patch goes through LayerNorm, a linear, LayerNorm, the
  rows of two position tables, LayerNorm, an RMS norm, and a projection.
- Audio at 16 kHz becomes frames of 640 samples (40 ms). Each frame goes
  through an RMS norm and a projection.

The GGUF keeps the 6912 values of a patch in the order channel, row,
column. transformers uses row, column, channel.

np_gemma/media.py puts the soft tokens of each placeholder into the prompt
ids (<|image> + n x <|image|> + <image|>). Model.prefill(media=spans) puts
the soft rows in place of the token rows. The tokens of one image see each
other in every layer. The text of the mask function of transformers says
that the global layers stay causal. But generate() and forward() of transformers
both give the logits of the mask in every layer, as llama.cpp does.
NP_GEMMA_BIDIR_ALL=0 keeps the global layers causal.

The checks (scripts/check_mm_unified.py, scripts/check_mm_prompt.py, and
the references of scripts/hf_mm_reference.py):

    soft rows of the same pixels           7.7e-5 (relative)
    image, float32 (the same weights)      KL 0.00000, 40/40 top tokens
    audio, float32                         KL 0.00000, 48/48
    image, int4 GGUF, our soft rows        KL 0.013, 40/40 (int4 on text: 0.014)
    the same, the Unsloth 12B file         KL 0.00055, 40/40
    audio, int4 GGUF, our soft rows        KL 0.0014, 47/48
    image with no mask                     KL 0.77, 29/41

The server takes --mmproj, --image-budget, and --media-dir. A request sends
image_url (a data URI, or a path under --media-dir; detail low gives 70
tokens, high 1120) and input_audio (base64). The server fetches nothing
from the network. A session matches the soft tokens only of the same media
(Session.common). Thus a chat that sends the same image in each turn
reuses the cache. The 12B on the GPU (the dense weights), with MTP:

    an image (280 tokens), 60 tokens out      8.8 s (a new chat, the prompt on the CPU)
    the same image, a new question            0.9 s (272 tokens reused)
    a clip of 17 s (436 tokens), 91 out       11.5 s (the prompt on the CPU)

The prompt pass of a prompt with media runs on the GPU when the model does.
ModelGPU.group puts the soft rows in x. The attention records of a large
group (GP_ATTN_QC_MT) take the array lim (operand 16): the last key of each
query less pos. Four kernels read that record: k_attn_qc_mt,
k_flash_qc_mt, k_flash_tc, and k_flash_qc_tc. They use lim for the causal
limit and for the last key of a tile.

Without media lim is the identity, so text gives the same bits. A chunk of
a prompt does not split an image. On the CPU the
prompt with media runs as a program (np_gemma/prompt.py), with int8 or
int16 activations as a prompt without media: the soft rows go into x, and
the int16 flash kernel (gemma_attn_prefill_qc) takes the last key of each
query (limit). The Python path takes the same kernel, so the two have the
same bits (NP_GEMMA_MEDIA_GPU=0 runs the prompt on the CPU on a GPU too).

    the 12B, int4, an image of 266 tokens     prompt 0.37 s (the CPU: about 6 s)
    KL to transformers                         0.0145, 40/40 (no mask: 0.757)
    a clip of 436 tokens                       prompt 0.39 s, KL 0.0014
    the server: an image and a question       1.9 s (before: 8.8 s)
    the server: a clip, 91 tokens out         3.0 s (before: 11.5 s)

check_gpu_prompt and check_gpu_split of the 26B pass.

### A cut of a chat cache on the GPU

A chat client sends the whole history in each turn. The Gemma 4 template
drops the thought part of an earlier answer, so the history can differ from
the cache before its end. The Session then cuts the cache back to the common
prefix (KVCache.truncate). On the GPU of the server this gave an illegal
memory access, and then each later request failed. The faults:

1. The GPU copy (GPUKV) learned of a cut only from sync(). While the GPU
   holds the cache, the host rows lag, so sync() did not see the cut.
2. A layer with a window keeps only the rows from base. A cut before base
   made the next prompt pass write before the buffer (the access fault).
3. A token at n sees back to n - window + 1. A cut that kept n but not
   those rows passed the test (n >= base), and the logits then had errors
   of up to 14.

Now Model.truncate_cache cuts the GPU copy (GPUKV.truncate) and the host
cache. Each one first checks that every layer with a window holds the row
n - window + 1. If a layer does not, the Session starts again from an empty
cache.

The script scripts/check_gpu_session.py runs a prompt of 8717 tokens
and 48 steps. It then cuts the history back by 1700, 1300, 300, 40, and 0
tokens, and adds a question. A reference Session makes the same rows with no
cut. The logits of 24 tokens have the same bits for all five cuts. Before
the fix, the cut of 1700 gave the access fault and the cut of 1300 gave the
errors.

The script fixes the hot experts (NP_GEMMA_GPU_HOT_DYN=0). It also turns off
the mixed groups (NP_GEMMA_GPU_MIX=0). Otherwise the place of an expert
follows the text, and the logits of two runs differ by about 2.

A cut that starts again costs a prompt pass of the history. The GPU keeps
between one and two windows of rows (1024 to 2048). Thus the model reads a
history again when it changes more than about 1000 tokens before its end.

The server now writes the trace of a failed generation to its log.

### A long prompt of the 26B on the GPU

After a prompt pass of about 8000 tokens or more on the GPU, the decode of
the 26B on the GPU gave nonsense. The same prompt on the CPU, a GPU prompt
with a CPU decode, and a CPU prompt with a GPU decode gave the right text.
The fault was in GPUKV:

1. A layer with a window that dropped its oldest rows (prepare) set
   host_end to the first row that it kept.
2. The host cache had no rows of the prompt. Thus sync() before the next
   step read the host end as a truncate, and it cut end to that value.
3. The next drop then moved base with no rows. The rows of the window no
   longer matched their positions.

prepare no longer changes host_end (detach starts at base). The same fault
also came in a long decode on the GPU, at the second drop. The script
scripts/check_gpu_prompt.py runs a prompt of 11778 tokens on the GPU. It
then compares 32 GPU steps with the CPU program on the same rows: 31 of 32
top tokens agree (20 before the fix).

The server (scripts/serve.py) has --max-context. A request with a longer
prompt gets an error 400, and max_tokens gets the room that is left. The
option --mtp now turns MTP on with --gpu too (NP_GEMMA_MTP=0 turns it off). The
26B with --gpu hot --gpu-experts-gb 1.5 --max-context 81920 answered a
prompt of 81495 tokens in 93 s.

### The Gemma 4 26B on an AVX2 CPU

The target is a CPU with AVX2 and no AVX-512: a Core i5-8500 (Coffee Lake,
6 cores, 32 GB, no GPU). The test here runs the code of that CPU on the
Xeon: NP_GEMMA_ARCH=avx2 selects the AVX2 library, and the kernels that
check the CPU at run time take their AVX2 form too. 6 threads, the method
of llama-bench (scripts/bench_llama_method.py), the Xeon at 50% of its
clock:

    runtime                              pp512   tg64
    numpy-gemma before                   19.6    6.5
    numpy-gemma now                      51.3    14.7
    llama.cpp (build-avx2, -t 6)         26.2    12.4

The changes:

- The AVX2 library did not build (no -mf16c), so such a CPU ran the NumPy
  fallback. It builds again.
- llama.cpp repacks the Q4_0 matrices in groups of 8 rows (q4_0_8x8). The
  KQ_Q4X copies of this runtime are the same idea (groups of 16 rows), and
  only a CPU with VNNI made them. An AVX2 CPU now makes them too. Their
  AVX2 kernel (kq_q4x_rows) takes int8 x, and a 32-bit lane is a row.
  One vpmaddubsw of the codes and 4 values of x gives the products of 8
  rows.
- The decode of the 26B on AVX2 takes int8 x (i4q_begin: one thread
  quantizes the x rows of a record). The Q6_K head takes int8 x too. The
  decode on AVX-512 keeps float32 x.
- The prompt uses the KQ_Q4X copies on AVX2 (they needed AVX-512 before),
  and kq_quant_part has an AVX2 form with the same bits.

A step of one token and a verify group still give the same bits
(check_mt.py). The int8 x changes the values. Against the float32 x of
the decode on AVX-512, over 255 steps of the chat text: KL (top 64)
4.9e-3, and 97.6% of the top tokens agree.

The experts alone give 2.0e-3. The dense matrices give the rest. About
half comes from the query and gate side, and half from the output and
down side. llama.cpp has the same int8 form. NP_GEMMA_Q4X=0 gives the
float32 decode (about half the speed).

The memory: the copies take 13 GB more (RssAnon 16.5 GB). The pages of
the GGUF file that the copies replace are clean. Thus the kernel can drop
them. llama.cpp keeps a repack of 13.1 GB in the same way. The load
takes 24.5 s, most of it for the copies.

### MTP on the GPU and the hot experts

In some runs of check_qwen4_gpu.py, MTP with the MTP layer on the GPU did
not give the tokens of the plain decode. Only the first 2 of 48 were the
same. The test looked for a fault of the synchronization. It found none:

- Each hot slot held the bytes of its expert. The slot table on the GPU
  was that of the host. The test looked before each verify group and each
  run of the MTP layer.
- With a synchronization of the GPU before each copy of an expert, the
  difference stayed. Thus no copy overlapped a kernel.
- The hot experts stayed fixed with NP_GEMMA_GPU_HOT_INS=0 or HOT_DYN=0.
  Then 6 runs of MTP on the GPU, a run on the CPU, and 2 plain runs gave
  the same 48 tokens.
- The tokens differed only at the two nearest ties of the plain decode:
  token 2 (margin 0.142) and token 18 (margin 0.076).

HotCache moves experts between the GPU and the CPU, and a hot expert gives
values a little different from a cold one. A run of MTP moves other
experts than the plain decode, so a near tie can go the other way. The
check now keeps the hot experts fixed for the MTP part, and it fails when
a token differs.

### The UD files of the E2B and the E4B

The files unsloth/gemma-4-E2B-it-GGUF and gemma-4-E4B-it-GGUF (UD-Q4_K_XL)
mix Q4_K, Q5_K, Q6_K, Q8_0, and IQ4_XS. The E4B class knew only Q4_0 (the
QAT files). The changes:

- gguf.py reads IQ4_XS (ggml type 23). On 64 blocks of 3 tensors, the
  values are those of dequantize_row_iq4_xs of ggml, bit for bit.
- The E2B file gives a feed-forward size for each layer (6144, then 12288).
  text_config gives a list then.
- E4B.kq gives a K quant or Q8_0 matrix to the GGUF products (GP_KQ_LINEAR
  on the CPU and on the GPU). An IQ4_XS matrix becomes Q8_0: the table
  values of the codes, and a float16 scale d (ls - 32) for each 32 values.
- The forms of the E4B take these matrices (linear, int4_multi4), and a
  small matrix gets a bfloat16 copy.
- The tiles of csrc/kquants.c now take rows of up to 1024 parts (KQ_S). The
  down matrices in Q6_K (10240 values) went from 4.4 to 81 GB/s.
- The GPU head of a K quant token table (Q5_K) is one GP_KQ_LINEAR.

Against the f32 mode (each weight dequantized), on 512 tokens of a chat
text:

    model   ppl f32   ppl int4   KL       top token
    E2B     11.744    11.932     0.0021   98.2%
    E4B     5.387     5.374      0.0006   98.0%

The decode against llama.cpp (llama-bench, 18 threads; tok/s):

    model   CPU    llama.cpp CPU   GPU    llama.cpp GPU
    E2B     13.9   24.4            99.7   178
    E4B     9.0    13.9            62.2   98

A GPU step of the E4B takes 15.9 ms. The products take 10.8 ms: 2.52 GB at
233 GB/s, half of the rate of the memory. About 660 small records take
about 5 ms. The kernel of the products of one token has a warp for each
row and few loads in flight. The hot experts of the 26B had the same
problem.

Three changes to the products of one token on the GPU:

- gpu.py _fuse_kq joins the GP_KQ_LINEAR records on the same x. One
  GP_KQ_MULTI does up to 5 matrices (q, k, v; gate, up).
- A row of 8192 values or more (the down matrices) gets 4 warps
  (k_kq_linear_split). A shorter row keeps one warp.
- kq_row_i8 takes int8 x (dp4a) for Q4_K, Q5_K, Q6_K, and Q8_0, as it did
  for Q8_R. The CPU products of these types use int8 x too. Thus the GPU
  now gives the values of the CPU to 5e-7 (5e-3 with float x).
  NP_GEMMA_GPU_I8X=0 gives the float x again.

The decode (tok/s, the same 64 tokens in each case):

    model   before   joined, split   int8 x
    E2B     99.7     102.6           107.9
    E4B     62.2     64.6            68.5

With int8 x, the Q6_K down matrices go from 337 to 470 GB/s. The kernel
reads 1 byte of x in place of 4. A small Q4_K matrix alone is
slower, because of the extra launch that quantizes x. A group shares that
launch.

Two changes to the small kernels of the step:

- k_add_norm_v keeps the values of a row in registers (float4), so it
  reads o, x, and w once. Alone in a graph, a record with two norms went
  from 5.5 to 2.6 us. The order of the sum of squares changed. Against the
  CPU, on 255 decode steps of the chat text, the KL (top 64) is 9.6e-4 and
  the top token agrees at 98.4%. With the old kernel, it was 1.08e-3 and
  97.6%.
- k_add_norm_v and k_gelu_mul_rows_v also write the int8 x when the next
  record is a product with int8 x. That record then does not quantize x.
  The launches that quantize x went from 135 to 35 for each step. These
  values are the same bits as those of k_kq_quant_x.

NP_GEMMA_GPU_AN_V=0 and NP_GEMMA_GPU_QX_FUSE=0 give the old kernels. The
decode (tok/s):

    model      old kernels   new kernels
    E2B        103.8         114.3
    E4B        70.6          72.1
    Qwen3.6    48.7          58.4 (check_qwen_gpu, 73.2 with a warm HotCache)

The Qwen3.6 step uses ADD_NORM too. It gives the same 128 tokens as the CPU.

#### The int8 x of the CPU products

On the CPU, a K quant product of the E4B step ran at 34 GB/s, but alone it
ran at 55 to 75 GB/s. The first product after GP_KQ_QUANT was slow. The gate
matrix took 350 to 920 us, and the up matrix after it took 270 us. The
threads of the team wrote the parts of the int8 x (omp for, 80 parts of 32
values). Then each thread of the product read lines that many cores wrote,
and each product lost about 200 us. The time of GP_KQ_QUANT did not show
it.

Now one thread quantizes an x of 512 parts or less (kq_quant_body,
kq_quant_rows_body, ma_quant_body). The quantization uses AVX-512 and gives
the same bits (cvtps rounds as lrintf). NP_GEMMA_QUANT_SINGLE sets the
limit, and 0 gives the old form. The CPU decode (tok/s):

    model                 old    new    llama.cpp
    E2B UD-Q4_K_XL        14.0   22.6   24.4
    E4B UD-Q4_K_XL        8.8    13.2   13.9
    Qwen3.6 UD-Q4_K_M     17.0   17.9   -

The E4B step went from 94 to 61 ms. The products now run at 52 GB/s. The
text of the decode is the same. The int4 path of the QAT files (Q4_0)
reads x as float, so it did not have the problem.

#### The GPU products against llama.cpp

MTP_PLAN.md has the details. In short:

- A lane of the one-token products reads 16 bytes (32 values) of each
  superblock, not 4 bytes. A warp then has 4 times the bytes in flight.
- The gate, the up matrix, and the GELU are one kernel (k_kq_glu_i8).
- The GPU applies the soft cap and finds the greedy token (GP_SOFTCAP,
  GP_ARGMAX).
- The token rows of the Q4_K and Q5_K tables come from C (gemma_kq45_rows),
  with the bits of NumPy.
- The rope tables of all the positions stay on the GPU.

The decode (tok/s, e4bgen, 32 tokens; llama.cpp from the table above):

    model   GPU    llama.cpp GPU   CPU    llama.cpp CPU
    E2B     131.1  178             24.5   24.4
    E4B     87.4   98              14.4   13.9

check_mtp.py (four prompts) gave 88.1 plain for the E4B, and llama.cpp
89.9 with the same prompts. The E2B on the GPU is still behind. Its
matrices are smaller, so the host part of a step (about 1 ms) is a larger
part of it.

### The Unsloth Q4_0 file of the 26B

Unsloth gives a file of the 26B with the name "UD-Q4_K_XL"
(unsloth/gemma-4-26B-A4B-it-qat-GGUF). All its tensors are Q4_0, the token
table (token_embd) too. The file of Google keeps the token table in Q6_K.

The weights of the Unsloth file agree with the unquantized QAT release of
Google (google/gemma-4-26B-A4B-it-qat-q4_0-unquantized). The difference is
0.16 to 0.18 per cent of the norm in each tensor that we compared. That is
the error of the float16 scales.

The Q4_0 file of Google has a difference
of 5 to 8 per cent from that release. Its Q6_K token table has 1.4 per cent.
The scales of its feed-forward matrices are 3/4 of the scales of the Unsloth
file. The file of Google thus does not have the grid of the release.

The test used three chat prompts with no media. For each prompt,
scripts/hf_mm_reference.py ran the release in float32 and made a greedy
answer of up to 128 tokens (models2/mm-refs/26b-chat1.npz to 26b-chat3.npz).
Then scripts/check_mm_prompt.py --model 26b --gguf FILE gave these results
over 360 positions:

    file              runtime   mean KL (top 64)   top token
    Google Q4_0       CPU       0.027              349/360
    Unsloth Q4_0      CPU       0.0010             359/360
    Unsloth Q4_0      GPU       0.0001             359/360

The runtime gave a Q4_0 token table to the bfloat16 head before. That head
reads 1.48 GB for each token, and the decode on the CPU fell from 17.4 to
14.4 tok/s. Now a Q4_0 token table is the int4 head (Model._embed_q from
gguf.int4_packed, the blocks of the file in place). It reads 0.42 GB, and
the Q6_K head of the file of Google reads 0.61 GB. The values do not change.
The products of two or more rows on the CPU take int8 x, as the layers do.

On the GPU, gg_q4_head (k_q4_head) is the head for a Q4_0 table. Lane l of a
warp reads two bytes of a block, four blocks at a time, and x in float2
pairs. One to four rows have their own kernels. The time of the head of the
26B on an RTX 5060 Ti:

    rows of x    1        2        3        4        16
    Q4_0         1.00 ms  1.33 ms  1.71 ms  2.06 ms  9.65 ms
    Q6_K         1.61 ms  2.59 ms  3.09 ms  3.98 ms  11.90 ms

We also tried a lane for each value of a block (the order of k_q6k_head).
That order reads x in one run, but it took 1.48 ms for one row.

The speed (tok/s; 18 threads, pp512 and tg128; the GPU with the hot experts,
scripts/bench_decode.py --gpu hot):

    file           llama.cpp CPU    numpy-gemma CPU   numpy-gemma GPU
    Google Q4_0    110 / 18.7       271 / 19.4        99.8
    Unsloth Q4_0   103 / 20.1       269 / 20.1        98.5

We also ran llama.cpp on the GPU, with the experts of 4 layers on the CPU.
It gave 1392 / 90.9 for the file of Google and 1411 / 96.9 for the Unsloth
file. The
GPU runs of numpy-gemma change by 94 to 103 tok/s from run to run.

Other differences of the Unsloth file: add_bos_token is 0, eot_token_id is
106, and the chat template is different. The prompt ids of our server are
the same for the two files.

Do the Google files have other weights? No. The GGUF of Google changed two
times after the release of 2026-06-05. The upload of 2026-07-15 changed the
chat template. The upload of 2026-07-17 has the note "validated QAT GGUF
checkpoint (280 sequence length, corrected vocabulary)".

We read 8 tensors of
the file of 2026-06-05 with HTTP range requests. They have the same bits as
the file of 2026-07-17, which is the file in models/gemma-4-26B-qat-q4_0
(sha256 3eca3b8f). Only the metadata changed. Thus no GGUF of Google has the
grid of the unquantized release.

The token lists, the merges, the token types, and the scores of the Unsloth
file are the same as those of the file of Google.

The chat template of
Unsloth is the canonical template of Google (2026-07-09) with two changes. It
has no header comment. It also accepts the arguments of a tool call as a
JSON string. The template of Google stops with an error for such a string.
But llama-server and np_gemma/chat.py change the string to a JSON object
before they use the template. The file np_gemma/chat_template.jinja has the
same bytes as the template of Google.

Unsloth changed the template of all its uploads to this template on
2026-07-17 (the commit "Added Gemma official chat template update"). Our
Unsloth files are those uploads: their sha256 values are the values of the
repositories now. An Unsloth file from before that date has an older
template. For such a file, use --chat-template-file in llama.cpp, or the
fixed file.

The add_bos_token 0 of the Unsloth 26B file has no effect in
llama.cpp. The vocabulary code sets it to true for Gemma 4
(src/llama-vocab.cpp), and llama-tokenize gives the BOS token for both files.
Thus the fixed file and the Unsloth file differ in llama.cpp only for a tool
call with the arguments as a string.

Use the Unsloth file directly. It is the 26B default of the scripts. Our
server makes the prompt with np_gemma/chat_template.jinja and adds the BOS
token itself. Thus the metadata of the file has no effect on our prompts.

A fixed file is optional. It has the tensors of Unsloth and all the metadata
of the file of Google (the template, add_bos_token 1, the tokenizer). Only
general.name changes. Make it for a program that needs the exact metadata of
Google:

    PYTHONPATH=../llama.cpp/gguf-py python scripts/gguf_swap_metadata.py \
        --meta models/gemma-4-26B-qat-q4_0/gemma-4-26B_q4_0-it.gguf \
        --tensors models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
        --out models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-Q4_0-fixed.gguf

The 658 tensors have the same bytes as those of the Unsloth file. llama-server
(--jinja) adds the BOS token and uses the template of Google. The check of
26b-chat1.npz gave KL 0.0016 on the CPU and 0.00006 on the GPU, and the GPU
decode gave 98.1 tok/s.

### The Unsloth Q4_0 file of the 12B

The 12B has the same facts as the 26B. The file of Google
(google/gemma-4-12B-it-qat-q4_0-gguf, sha256 93567e57, the upload of
2026-07-17) is in models/gemma-4-12B-qat-q4_0. We read 8 tensors of the
file of 2026-06-05 with HTTP range requests. They have the same bits as the
file of 2026-07-17.

The difference from the unquantized release
(google/gemma-4-12B-it-qat-q4_0-unquantized), as a per cent of the norm:

    tensors                      Google    Unsloth
    attention, feed-forward      4.9-5.1   0.18
    token table                  1.34      0.18

Only 15 to 18 per cent of the codes of the two files are the same. The
Unsloth file (unsloth/gemma-4-12B-it-qat-GGUF, gemma-4-12B-it-qat-UD-Q4_K_XL,
sha256 90fd44e2) is Q4_0 in all its tensors, the token table too. Its
metadata has the same differences as the 26B file: the template of Unsloth
and eot_token_id 106. Its add_bos_token is 1, as in the file of Google.

The test used the three chat prompts of the 26B test. The reference is the
12B release in float32 (models2/mm-refs/12b-chat1.npz to 12b-chat3.npz).
The results over 376 positions:

    file              runtime   mean KL (top 64)   top token
    Google Q4_0       CPU       0.026              359/376
    Google Q4_0       GPU       0.025              362/376
    Unsloth Q4_0      CPU       0.0007             376/376
    Unsloth Q4_0      GPU       0.0001             376/376

With an image (12b-image.npz, our soft rows), the Unsloth tensors gave KL
0.00055,
and the file of Google gave 0.013. That value of 0.013 is the int4 value in
the table of "Images and audio on the Gemma 4 12B". It came from the grid of
the file of Google, not from the runtime.

Use the Unsloth file of the 12B directly too. It is the 12B default of
scripts/check_mm_prompt.py. An optional fixed file of the 12B, with the
metadata of Google:

    PYTHONPATH=../llama.cpp/gguf-py python scripts/gguf_swap_metadata.py \
        --meta models/gemma-4-12B-qat-q4_0/gemma-4-12b-it-qat-q4_0.gguf \
        --tensors models2/gemma-4-12B-unsloth-UD-Q4_K_XL/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
        --out models2/gemma-4-12B-unsloth-UD-Q4_K_XL/gemma-4-12B-it-qat-Q4_0-fixed.gguf

The 667 tensors of the fixed file have the same bytes as those of the
Unsloth file. Give --model-id gemma-4-12b-it-qat-q4_0 to scripts/serve.py to
keep the model id.
The speed (tok/s, pp512 / tg128 on the CPU; scripts/bench_decode.py --gpu
dense):

    file                  CPU            GPU
    Google Q4_0           89.0 / 7.80    45.4
    Unsloth (fixed) Q4_0  88.6 / 7.83    47.4

The head of the Unsloth file on the GPU is 0.57 GB, and the Q6_K head of the
file of Google is 0.83 GB.

### The Unsloth Q4_0 files of the E4B and the E2B

The E4B and the E2B have the same facts as the 26B and the 12B. The files of
Google (google/gemma-4-E4B-it-qat-q4_0-gguf, sha256 676c3507, and
gemma-4-E2B-it-qat-q4_0-gguf, sha256 fa401b55) are the uploads of
2026-07-17. We read 8 tensors of the files of 2026-06-05 with HTTP range
requests. They have the same bits as the files of 2026-07-17.

The QAT files of Unsloth (unsloth/gemma-4-E4B-it-qat-GGUF and
gemma-4-E2B-it-qat-GGUF, UD-Q4_K_XL, sha256 df0fd4ee and e5310072) are Q4_0
in all their tensors. The file of Google keeps the token table and the
per-layer table in Q6_K and the per-layer projection in F16. Thus the files
of Unsloth are smaller: 4.22 GB and 2.62 GB, not 5.15 GB and 3.35 GB. These
files are not the UD files of the section "The UD files of the E2B and the
E4B". Those files are not QAT files.

The unquantized releases are in models2/gemma-4-E4B-qat and
models2/gemma-4-E2B-qat (google/gemma-4-E4B-it-qat-q4_0-unquantized and the
E2B). The difference from them, as a per cent of the norm:

    tensors                          Google E4B   Google E2B   Unsloth
    the matrices of the layers       4.9-5.2      4.9-5.2      0.18
    the token table                  1.33         1.38         0.18
    the per-layer table              1.38         1.36         0.18
    the per-layer projection         0.00 (F16)   0.00 (F16)   0.18

The release keeps the per-layer projection on a Q4_0 grid too.

The test used the three chat prompts of the 26B test. Each release in
float32 is the reference (models2/mm-refs/e4b-chat1.npz to e4b-chat3.npz,
and e2b-chat1.npz to e2b-chat3.npz). The results over 364 positions:

    model   file      runtime   mean KL (top 64)   top token
    E4B     Google    CPU       0.020              343/364
    E4B     Google    GPU       0.019              346/364
    E4B     Unsloth   CPU       0.0004             363/364
    E4B     Unsloth   GPU       0.00002            363/364
    E2B     Google    CPU       0.025              346/364
    E2B     Google    GPU       0.024              346/364
    E2B     Unsloth   CPU       0.0007             358/364
    E2B     Unsloth   GPU       0.00003            363/364

The references of the E4B with media (our soft rows) gave these results:

    reference         Google            Unsloth
    e4b-image.npz     KL 0.0076, 38/40  KL 0.00030, 40/40
    e4b-audio.npz     KL 0.0018, 48/48  KL 0.00006, 48/48
    e4b-text.npz      KL 0.022, 40/40   KL 0.00024, 39/40

The template of the E2B and the E4B is the canonical template of Google
without the empty thought block. The runtime gives empty_thought_block False
for these models too. The template of Unsloth has the same two changes as
for the 26B. The add_bos_token is 1 in all four files.

Two changes let the runtime use these files:

- The E4B class on the GPU took a Q6_K head (gg_q6k_head) or a K quant head
  (GP_KQ_LINEAR). A Q4_0 token table stopped the GPU path with an error.
  Now E4B._gpu_head gives the Q4_0 blocks to gg_q4_head. Against the CPU,
  scripts/check_gpu.py gave logits max |d| 0.0001 and 32/32 top tokens.
- The per-layer rows of a prompt (embed_rows, gguf.take_rows) of a Q4_0 table
  went through NumPy: 39.7 ms for 512 tokens of the E4B. The C function
  gemma_q4_0_rows takes 2.5 ms and gives the same bits. The prompt pass of
  512 tokens on the GPU went from 2494 to 2990 tok/s (E4B) and from 4158 to
  5203 tok/s (E2B).

The speed is in tok/s. The CPU gives pp512 / tg128 with 18 threads. The GPU
values come from scripts/bench_e4b_gpu.py, with the tensor cores in the
prompt pass:

    model   file      CPU             GPU pp512   GPU tg128
    E4B     Google    96.6 / 15.0     2989        100.6
    E4B     Unsloth   111.2 / 15.9    2990        108.7
    E2B     Google    180.7 / 26.5    5192        167.2
    E2B     Unsloth   225.2 / 28.2    5203        179.9

Use the Unsloth files directly. The E4B scripts use the Unsloth E4B file as
their default. The mmproj file of Google stays the mmproj file of the E4B.

## Test results

    Test                                      Result
    Layer 0, position 0                       16/16 tensors, cosine >= 0.999996
    Layer 0, position 3 (RoPE active)         16/16 tensors, cosine >= 0.99999
    Up to global layer 5 (K=V, p-RoPE)        85/85 tensors, cosine >= 0.999
    All 48 layers and the logits              671/671 tensors, cosine >= 0.999
    Final hidden state                        cosine 0.999678
    First greedy token                        HF 50429 = NumPy 50429
    Key/value cache against a batch prefill   relative L2 4.8e-07
    Tokenizer encode and decode               25/25 and 25/25 exact
    Chat template, 5 message shapes           5/5 exact
    24 greedy tokens, 2 prompts, f32          48/48 ids equal to HF
    24 greedy tokens, 2 prompts, bf16 + NumPy 48/48 ids equal to HF
    24 greedy tokens, 2 prompts, bf16 + Numba 48/48 ids equal to HF
    24 greedy tokens, 2 prompts, bf16 + C     48/48 ids equal to HF
    24 greedy tokens, 2 prompts, int8 + C     48/48 ids equal to HF
    24 greedy tokens, 2 prompts, int4 + C     48/48 ids equal to HF
     (int4 from the w4a16 checkpoint)

    Gemma 4 E4B, the mobile-ct checkpoint
    Compressed-tensors decode, 4 weight kinds   exact, 7/7 tensors
    row() and rows() against dequant()          exact, 7/7 tensors
    Hidden states, all 42 layers                max relative 0.037, no jump
    Input embedding                             one bfloat16 step
    Greedy tokens vs the reference              8/8 ids equal
    Cached decode vs the whole sequence         8/8 ids equal
    int4 mode, prefill and decode               8/8 ids equal to the reference
    Packed kernel, every column, 4-bit and 2-bit  exact, 3 matrices
    Packed kernel, block of 4 tokens            relative 1e-6
    Tokenizer and chat template vs HF           ids equal
    First output                               "The capital of France is **Paris**."

    Gemma 4 E4B, the Q4_0 GGUF file
    Name mapping, config, and tied head         all tensors found
    Layer 0, position 0                         cosine 0.999968
    Layer 20                                    cosine 0.999845
    Layer 41                                    cosine 0.999607
    Logits                                      cosine 0.999941
    First greedy token                          reference 818 = ours 818
    Greedy tokens vs the reference              8/8 ids equal
    Cached decode vs the whole sequence         8/8 ids equal
    int8 tile with VNNI, 12 token counts        relative 3.6e-07 or better
    GELU kernel against float64                 max difference 4.3e-07
    RMSNorm kernel against float64, 14 shapes   max difference 1.1e-07

The GELU check gives the same error for the AVX-512 build and the AVX2 build.
98.2 per cent of the values agree bit for bit between the two builds, and the
largest difference is 2.4e-07.

The f32 mode computes all values as float32. The real model computes most
values as bfloat16. Thus small differences occur. The cosine value stays above
0.999. The generated token ids are equal.

## Model facts

These facts are necessary. A generic transformer will give wrong output.

* The model has 48 layers. Forty layers are sliding layers. They have
  head_dim 256 and 8 key and value heads. Eight layers are global layers. They
  are at positions 5, 11, 17, 23, 29, 35, 41, and 47. They have head_dim 512
  and 1 key and value head.
* A global layer has no value projection. It uses the raw key projection for
  the value. It applies RMSNorm to the value. It does not apply RoPE to the
  value.
* RMSNorm multiplies by the full weight. Do not add 1 to the weight. The weight
  values are large. The mean is 6.6. The maximum is 193.
* The attention scale is 1.0. Do not divide by the square root of head_dim.
  The QK-norm replaces this step.
* A sliding layer uses the default RoPE with theta 1e4. A global layer uses the
  proportional RoPE with partial_rotary_factor 0.25 and theta 1e6. Only 64 of
  256 angle pairs turn. The inverse frequencies have zeros at the end. The
  rotate_half function pairs dimension i with dimension i + head_dim / 2.
* Multiply the input embeddings by the square root of hidden_size. The value is
  the square root of 3840.
* Each layer has a trained scalar. Layer 0 has 0.0544. Layer 5 has 0.365.
  Layer 47 has 0.0496. Multiply the layer output by this scalar.
* The embedding table and the output head are tied. Apply the final logit
  softcap: tanh(logits / 30) * 30.
* A GGUF QAT file of Google keeps the tied token table in the Q6_K type.
  The code reads the 210-byte blocks in place. The output head then reads 6.05
  bits for each weight in place of 16 bits. The load step does no dequantize
  of the table. For the 26B model this step was 6.6 s.
* A file with a Q4_0 table (Unsloth) gives the int4 head: 4.5 bits for each
  weight.
* The attention mask is causal. A sliding layer also masks keys that are older
  than 1024 positions.
* The tokenizer replaces each space with the character U+2581. It uses BPE with
  byte fallback. The decoder replaces U+2581 with a space, joins the byte
  tokens, and joins the parts. In thinking mode, the chat template opens the
  system turn with the think token. In normal mode, it closes an empty thought
  channel.

## Memory bandwidth

**The clock of this machine is low.** The governor is `intel_pstate` in
`powersave` mode with `energy_performance_preference` of `balance_performance`.
The cores idle at 1.2 GHz and the maximum is 4.6 GHz. One busy core runs at
about 2.3 GHz, and it stays there. A run of 13 seconds gave 2.34 GHz. A run of
0.14 seconds gave 2.30 GHz.

Thus the clock does not rise with a longer load. The base clock of the part is
3.0 GHz.

    sudo cpupower frequency-set -g performance

That command puts the clock at the maximum. It is not applied here, because the
machine has other users. Every absolute number in this document comes from a
machine at about 2.3 GHz. A comparison between two versions of this project is
not affected, because both ran at the same clock. A comparison of a compute
bound stage against the memory rate is affected, because the memory rate does
not change with the clock.

The machine gives more bandwidth than the kernels use. Measured with
scripts/membw.c (6 threads, one 4 GB array):

    read float32                35.5 GB/s
    read float32, 12 threads    42.3 GB/s
    read bfloat16               33.1 GB/s
    copy (read and write)       29.0 GB/s

The C bf16 kernel gives about 59 GB/s for one large matrix. The same kernel
gives about 17 GB/s for a group of 40 large matrices (4.7 GB). The full model
gives about 17 GB/s. The kernel reads the same bytes in all three tests. Thus
the size of the working set controls the speed. A large working set has more
translation lookaside buffer misses.

The steps below are complete:

    huge pages for the weight mapping      done (the system may say no)
    four output rows in one inner loop     done
    AVX-512 with an AVX2 fallback          done
    integer int8 kernel                    done, off by default

These steps changed the decode time from 1.46 s per token to 1.07 s per token.
The effective bandwidth changed from 12.4 GB/s to 17.0 GB/s. The machine can
give about 35 GB/s for a plain read. Thus some bandwidth remains.

## Server

The server gives the OpenAI paths. It uses only the Python standard library.

    PYTHONPATH=. python scripts/serve.py \
        --gguf models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
        --model-id gemma-4-26B_q4_0-it --dtype int4 --port 8080 --temperature 0.0

Use the Unsloth file of the 26B (see "The Unsloth Q4_0 file of the 26B"). The
--model-id option keeps the model id of the harness settings below.

The paths are:

    GET  /v1/models               List the model.
    GET  /v1/models/{id}          One model.
    POST /v1/chat/completions     A chat turn. Set stream true for the events.
    POST /v1/completions          A raw text prompt.
    GET  /health                  The status.

The server also takes the bare paths with no /v1, because some clients use
them. The chat path takes the fields that the OpenAI clients send. The fields
are model, messages, max_tokens, temperature, top_k, top_p, min_p, stop,
stream, seed, presence_penalty, frequency_penalty, and repetition_penalty. A
stream uses the server-sent-event format and ends with the data [DONE] line.
Set stream_options.include_usage to get the token counts in the last event.

The model is not thread safe and shares one key and value cache. Thus the
server answers one request at a time. The other requests wait.

A client points at http://127.0.0.1:8080/v1 . The model id is the file name of
the GGUF.

The thought channel is open by default (--thinking auto). It is closed when
--max-context is less than 32768, because the thought part takes room. Use
--thinking on or --thinking off to set it. The Gemma 4 template has one level
of thought.

A request sets the thought channel with the field thinking (true, false,
or {"type": "enabled"}), reasoning_effort, reasoning.effort, or
chat_template_kwargs.enable_thinking. The values none, off, and disabled
close it.

After a tool result, the Gemma 4 template ends the prompt in an open
thought channel (<|channel>thought and a newline). The model then writes
its thought with no opener, and closes it with <channel|>. The server
parses such an answer with the opener in front (Backend.open_channel).
Before this, the thought went out as the answer. A harness then sent it
back as the words of the model, and the model repeated its steps.

Use --debug DIR to write each turn to a file in DIR. The file has the request,
the prompt text, the thought setting, the raw answer, and the tool calls.
DIR/turns.log gets a line for each turn.

The server gives tool calls. Send the OpenAI tools field. The model then answers
with the field tool_calls and the finish reason tool_calls. Send the result back
as a message with the role tool and the field tool_call_id. The prompt uses the
canonical Gemma 4 template in np_gemma/chat_template.jinja. The file gives the
exact text without a Jinja engine.

The server also splits the thought channel from the answer. With thinking on,
the reasoning arrives in the field reasoning_content, both in the message and in
the stream delta. The answer arrives in the field content.

Check the chat template against the Jinja file. The test needs jinja2, which the
Hugging Face package gives.

    python scripts/check_chat_template.py

Check the server without a model. The script uses a fake backend.

    python scripts/check_server.py

Check it with a model. The script loads the model and asks for the capital of
France.

    PYTHONPATH=. python scripts/check_server_live.py --gguf PATH --dtype int4

### DeepSeek Harness

Register the server as a provider in the harness settings file. The
llm-pi-ai.providers dict holds one entry for each route:

    llm-pi-ai:
      providers:
        npgemma:
          displayName: Gemma 4 26B (local)
          api: openai-completions
          baseURL: http://jackal.local:8123/v1
          models:
            - id: gemma-4-26B_q4_0-it
              name: Gemma 4 26B Q4_0
              contextWindow: 8192
              maxTokens: 4096

Start the server on the machine that holds the weights. Bind every address so
that the harness can reach it over the network:

    PYTHONPATH=. OMP_NUM_THREADS=18 python scripts/serve.py --gguf PATH --dtype int4 --host 0.0.0.0 --port 8123 --temperature 0.0

The option --gpu dense needs a CUDA GPU and nvcc. It puts the weights
outside the experts on the GPU: 1.66 B weights, about 1 GB. It also puts the
output head (0.6 GB) there. The experts (22.8 B weights) stay on the CPU.

The option --gpu hot also puts the most used
experts on the GPU, up to --gpu-experts-gb. The server copies the weights at
the start. On jackal, a greedy generation of the 26B gives the same tokens
as the CPU:

    mode                    decode rate           prompt pass
    CPU only                about 18 tokens/s     about 60 tokens/s
    --gpu dense             about 40 tokens/s     about 440 tokens/s
    --gpu hot (about 2 GB)  about 45 to 52        about 440 tokens/s

With NP_GEMMA_GPU=1, the E4B model runs wholly on the GPU. The decode, the
prompt pass (tensor cores), and MTP with the drafter on the GPU
(np_gemma.gpu.GPUDrafter) run there. The script scripts/bench_e4b_gpu.py
measures them:

    E4B on the RTX 5060 Ti    this runtime     llama.cpp (CUDA)
    prompt, 1024 tokens       3107 tok/s       5178 tok/s
    prompt, int8 products     4650 tok/s       -
    decode                    94.5 tok/s       112 tok/s
    decode with MTP           128 tok/s        -

The prompt pass on the GPU copies the experts that the GPU does not hold for
each chunk of 1024 tokens. The option works for scripts/gguf_generate.py
too.

MTP works with --gpu and gives the same tokens. With the drafter on
the GPU (GPUDrafter, scripts/bench_mtp_gpu.py), the 26B gives about 47
tokens/s with MTP, against 45 for the plain decode. The gain is small,
because the verify group sends more experts to the CPU. Thus the server
turns MTP off with --gpu unless NP_GEMMA_MTP=1.

The server also takes the E2B and E4B models. It finds the kind from the
GGUF data. With --gpu dense or hot, the whole E4B model runs on the GPU.
With --mtp and --gpu, the drafter runs on the GPU too, and MTP is on by
default for the E4B:

    python scripts/serve.py --gguf gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf --gpu dense --mtp ASSISTANT_DIR

A chat turn of 197 tokens took 2.2 s, with the prompt pass and MTP. The
server keeps the cache of a chat for the next turn, as for the 26B.
See SPLIT_PLAN.md.

Then choose the provider and the model in the harness. No credential is
needed, because the daemon reads no key. The context window is a choice. The
model allows 262144, and the key and value cache is float32, so a smaller
value keeps the cache small.

The harness needs tool calls, a finish reason, and streamed usage. The server
gives all three. Check the wire path through the harness client library:

    node scripts/check_pi_ai.mjs http://jackal.local:8123/v1

### Sampling

np_gemma/sampling.py gives the Sampler class. A temperature of zero selects the
most probable token. The other settings are top_k, top_p, min_p, and the three
penalties. The seed makes a sampled run repeatable. The class keeps the count of
each token in the history for the penalties.

With MTP and sampling, a draft stays only when the sample picks it. That is
the default (mtp_accept "exact"), and the text then follows the distribution
of the settings. The option mtp_accept "in_set" also keeps a draft when the
settings allow it (Sampler.draft_ok). It needs top_k, top_p, min_p, or
mtp_floor. Without them it acts as "exact".

The Sampler keeps a draft only
when its probability is at least mtp_floor times the best probability. Give
the options to scripts/serve.py (--mtp-accept in_set --mtp-floor 0.1), or in
a request ("mtp_accept", "mtp_floor").

MTP_PLAN.md has the test. With a floor of 0.1, three drafts on the E4B gave
144.7 tok/s, and "exact" gave 127.9. The test found no drift above its noise.
The values below come after the changes of the next paragraphs.

A pick on the host now sorts only the candidates. For top_k they come from
_top_idx. For top_p and min_p they are the tokens near the best score. A pick
took 0.31 ms, not 1.56 ms,
with top_k 64. With top_p alone it took 1.0 ms, not 21 ms. The tokens are
the same with the same seed.

On the GPU, the head also gives the candidates of each row (gg_topk, a radix
select). They are the k best logits, the row max, and the sum for the
temperature. The host then copies k values, not 262144.

 Sampler.sample_sparse uses them
when they settle the result (top_k, or top_p and min_p within the
candidates). Else it takes the whole row (assistant.RowPicker). A greedy
Sampler takes the best tokens of the GPU. NP_GEMMA_SPARSE=0 turns the
candidates off. The E4B on the GPU with the settings of Gemma (tok/s):

    mode               plain    1 draft   2 drafts   3 drafts
    greedy             107.7    153.4     165.3      175.8
    sampling, before    90.8    116.6     119.4      127.4
    sampling, now      107.0    152.3     158.2      174.2
    in_set 0.1, now             169.9     185.0      214.6

The tokens are the same as with the whole rows, for each setting of the
test, on the E4B, the 12B, and the 26B.

A floor of 0.1 did not change the accuracy on GSM8K. But it makes open text
less varied, as a temperature of about 0.8 in place of 1.0 does (see
MTP_PLAN.md). Use it for tasks with one right answer.

## Limits

* The f32 mode needs about 70 GB of memory. Use the int8 mode when memory is
  small. The int8 mode needs about 12 GB.
* The C path needs a C compiler (cc, gcc, or clang). Without a compiler, the
  model uses the Numba path or the NumPy path.
* The C library uses AVX2 and FMA instructions. The target machines always give
  these instructions. The code uses AVX-512 when the CPU gives it, and AVX2
  otherwise.
* The key and value cache uses buffers of a fixed size. A sliding layer keeps
  one window. The code copies the buffer only when the buffer holds two
  windows. Thus the copy work does not grow with the sequence length.
* The int8 mode quantizes each row with one scale. A smaller group gives a
  smaller error and a slower multiply.
* The runtime gives the correct first token. The runtime and the reference can
  disagree at later tokens, because one uses float32 and the other uses
  bfloat16. The test above found no disagreement in 48 tokens.

## Next steps

1. The bandwidth is at the machine limit. The kernel gives 42 GB/s in a
   single pass. A plain read gives 41 GB/s to 43 GB/s with six threads. A
   loaded machine gave a lower value. Do not expect more speed from this path.
2. The multi-level GEMM now runs on AVX2 as well as AVX-512. AVX2 reached 289
   to 335 GFLOP/s for int8 at 256 tokens. AVX-512 reached 453 to 475 GFLOP/s.
   The measured pure FMA value is 602 to 676 GFLOP/s with six threads. The AVX2
   peak is about half of the AVX-512 peak, so the AVX2 kernel is near its share
   of the limit. A wider micro tile or a packed B panel can give a small gain.
3. The int4 path now has a prompt GEMM on both targets. AVX2 reached 308 to 337
   GFLOP/s at 256 tokens. AVX-512 reached 446 to 483 GFLOP/s. The GEMM decodes
   one row block to a float32 A panel and reuses the panel for every token
   block. The int4 decode kernel also uses a byte-lane sign decode. That change
   gave 2.6 times more speed on a wide matrix.
4. Make the integer int8 kernel accurate. Use a scale for a group of columns
   for the activations as well as the weights.
5. Add bf16 rounding after each operation. Then the float32 mode follows the
   reference more closely.
6. Port the model to C.
7. Use the vector tanh in `gemma_gelu_mul`. The mixture-of-experts path of the
   26B model still calls the scalar `tanhf` for each value.
8. Cut the number of calls for the small matrices of E4B. The 84 per-layer
   matrices cannot join one call, because each one waits for the layer before
   it. Each call costs about 16 microseconds of fixed time at 18 threads.
9. Make the int8 tile faster for the prompt. It is 80 per cent of a prompt
   pass at about 900 GFLOP/s. A copy of a weight as int8 can remove the
   unpack of each 4-bit block, at the cost of a larger weight.
10. ~~Give the decode step an int8 kernel with 8 rows for each pass and one
    token lane.~~ **Done. It does not help.** See "What llama.cpp does
    differently". The kernel `gemma_int4_q8_gemv` is correct and it is 1.0 to
    1.2 times the float kernel, but the group scale costs it the gain. The
    int8 form is off by default, behind `NP_GEMMA_INT4_Q8_GEMV=1`.
11. ~~Make `gemma_rms_norm` use more than one value for each cycle.~~ **Done.**
    The sum of the squares was one chain of additions with a latency of 4
    cycles for each value. It now uses four accumulators at 512 bits. The work
    of one call fell from 7.1 to 2.5 microseconds. The fixed cost of the call
    is now the larger part. See "The elementwise kernels".
12. ~~Cut the cost of a call to C.~~ **Done, in part.** The environment is now
    read one time, and four fused entry points run two kernels in one call.
    The decode of the 26B model gained 3.8 per cent, and the count of calls for
    each token fell from about 600 to about 420. See "The cost of a call to C".
13. Keep the address of a buffer across calls. A decode step hands about 1300
    arrays to the C library, and each read of `ndarray.ctypes.data` costs about
    1.5 microseconds. That is 2 ms of a 72 ms step. A pool of buffers that the
    model owns can remove most of it. The danger is a caller that keeps a
    result while the next call writes into the same buffer. A smaller step
    with no danger is to keep the address of a weight, which the program never
    changes.
14. Reach the remaining fusion. The norm of the input of the attention block
    and the norm of the post-attention state are still separate calls. Both
    feed a matrix kernel, so both can join it as the gate and the up projection
    now do.

## Sample run

The block below shows the command and the start of the answer. The model
output is quoted as it came from the model, so its words are not the words
of this document.

```text
$ PYTHONPATH=. $PY scripts/session.py --snapshot "$SNAP" --dtype int8 --max-new-tokens 512     --prompts "How does grouped query attention work?"
[load] 48 layers + embedding table resident (int8) in 5.4s | RSS 0.3 GB
[gen ] 512 new tokens in 731.8s (0.70 tok/s) | RSS 12.0 GB
       **Grouped-Query Attention (GQA)** is a technique used in Transformer architectures to balance the trade-off between the high performance of **Multi-Head Attention (MHA)** and the memory efficiency of **Multi-Query Attention (MQA)**.

It is most famously used in models like **LLaMA 2 and LLaMA 3**.

To understand GQA, you first need to understand the two extremes it sits between.

---

### 1. The Context: MHA vs. MQA

In standard Transformer attention, we have three sets of matrices: **Queries (Q)**, **Keys (K)**, and **Values (V)**.

#### Multi-Head Attention (MHA)
*   **How it works:** Every Query head has its own corresponding Key head and Value head. If you have 8 heads, you have 8 sets of Q, 8 sets of K, and 8 sets of V.
*   **Pros:** High representational power; the model can attend to many different types of information simultaneously.
*   **Cons:** Very memory-intensive during inference. Because every head needs its own K and V, the **KV Cache** (the memory used to store previous tokens) grows very large, limiting the maximum context length and batch size.

#### Multi-Query Attention (MQA)
*   **How it works:** All Query heads share a **single** Key head and a single Value head. If you have 8 heads, you have 8 sets of Q, but only 1 set of K and 1 set of V.
*   **Pros:** Massive reduction in KV Cache size (by a factor of 8 in this example). This makes inference much faster and allows for much larger batch sizes.
*   **Cons:** Significant drop in model quality. Because all heads are forced to look at the same keys/values, the model loses the ability to focus on diverse information at once.

---

### 2. The Solution: Grouped-Query Attention (GQA)

GQA is the "middle ground." Instead of every head having its own K/V (MHA) or every head sharing one K/V (MQA), **Query heads are divided into groups.** Each group shares a single Key and Value head.

#### How it works visually:
Imagine you have **8 Query heads**. You can group them into **2 groups** (4 heads per group
```

