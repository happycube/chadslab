# numpy-gemma — a NumPy-only Gemma 4 12B runtime

This project runs the model google/gemma-4-12B-it-qat-q4_0-unquantized.
It uses NumPy only. It does not use PyTorch. It does not use transformers.
The project is the Phase 1 baseline of the learning plan in
../Gemma LLM Runtime Learning Plan.md.

It also runs google/gemma-4-E4B-it-qat-mobile-ct, the 4.5B effective dense
model. That model comes from a compressed-tensors SafeTensors file or from the
Q4_0 GGUF file of the same model. It adds Per-Layer Embeddings. Transformers
appears only in the check scripts, where it builds the reference.

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
    NP_GEMMA_ATTN          1                       1 uses the fused attention over an int16 copy of the cache, with a float query. 0 uses the float cache. The two give about the same result. The server uses 1 (--kv-attn int16).
    NP_GEMMA_FUSED_QKV     1                       1 gives the query, the key, and the value their norm in one call, and the query and the key their rope in one call. 0 gives each tensor its own call.
    NP_GEMMA_CACHE_RAM     0                       1 copies the cache into local memory with large pages.
    NP_GEMMA_CACHE         ~/.cache/np_gemma/weights  The cache directory.
    NP_GEMMA_ARCH          auto                    avx2 or avx512 forces one C library.
    NP_GEMMA_KERNEL        auto                    c, numba, or numpy forces one kernel path.
    NP_GEMMA_INT8_INT      0                       1 uses the integer int8 kernel. That kernel is less accurate.
    NP_GEMMA_PREFILL_CHUNK 256                     The prompt pass uses blocks of this many tokens.
    NP_GEMMA_SLIDE         1                       1 drops the keys that no query in the block can see. 0 keeps every key.
    NP_GEMMA_FLASH         0                       1 runs the C flash attention kernel. ref runs the NumPy reference. 0 uses the batched matmul.
    NP_GEMMA_ATTN_IMPL     auto                    c, avx2, or avx512 forces one version of the flash kernel.
    NP_GEMMA_FLASH         0 for the 12B,         1 sends a prompt of more than one token to the C flash kernel. "slide" uses it for a sliding layer only.
                           1 for E4B
    NP_GEMMA_INT4_Q8       per model               The activations of the int4 products of a prompt pass. 1 uses int8 for every product, as llama.cpp does. 16 uses float32 for the attention and dense matrices and int16 for the experts: 99.7 per cent of the tokens agree with float32, at 1.4 times the time of 1. 0 uses float32 for every product, at 1.9 times the time of 1. Without the variable, a model with experts (the 26B) uses 16 and a dense model uses 1.
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
    NP_GEMMA_GPU           0                       1 runs the decode steps, the prompt pass, and the output head on a CUDA GPU (np_gemma/gpu.py, SPLIT_PLAN.md). The E4B model runs wholly on the GPU (decode only). The 26B model keeps its cold experts on the CPU. It needs nvcc. The first step copies the weights to the GPU.
    NP_GEMMA_GPU_HOT       the 26B counts          A file of expert counts (scripts/expert_use.py). With NP_GEMMA_GPU=1, the GPU holds the most used experts of the 26B and runs them. 0 keeps all the experts on the CPU. The default is np_gemma/data/gemma-4-26B-expert-counts.npz.
    NP_GEMMA_GPU_HOT_GB    free less 6 GB          The GPU memory for the hot experts, in GB.
    NP_GEMMA_GPU_TC        1                       1 runs the int4 products and the attention of a large group (a prompt pass) on the tensor cores, with float16 inputs. 8 gives int8 inputs to the products (the Q8_0 form): about 15% faster for the E4B, and 98.6% of the top tokens agree with float32, against 99.9%. 0 keeps float32 kernels. The 26B uses float32 kernels unless NP_GEMMA_GPU_TC_MOE=1.
    NP_GEMMA_GPU_CHUNK     1024                    The tokens of a chunk of a prompt pass on the GPU. Each chunk copies the cold experts to the GPU (about 1.9 s for the 26B), so a longer chunk is faster.
    NP_GEMMA_GPU_PREFILL_MIN 128                   A shorter part of a prompt runs on the GPU in groups of 16 tokens, with the experts on the CPU.
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

    GGUF4B=~/.cache/e4b-gguf/gemma-4-E4B_q4_0-it.gguf

The GGUF file holds Q4_0 weights, Q6_K embedding tables, and one F16 matrix.
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
every prompt length. The default for E4B is 1, and the 12B model keeps its own
default of 0.

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
* A GGUF QAT file keeps the tied embedding table in the Q6_K type. The code
  reads the 210-byte blocks in place. The output head then reads 6.05 bits for
  each weight in place of 16 bits. The load step does no dequantize of the
  table. For the 26B model this step was 6.6 s.
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

    PYTHONPATH=. python scripts/serve.py --gguf models/gemma-4-26B-qat-q4_0/gemma-4-26B_q4_0-it.gguf \
        --dtype int4 --port 8080 --temperature 0.0

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
the GGUF. Use --thinking to open the thought channel.

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
    prompt, 1024 tokens       1509 tok/s       5029 tok/s
    decode                    84 tok/s         112 tok/s
    decode with MTP           124 tok/s        -

The prompt pass on the GPU copies the experts that the GPU does not hold for
each chunk of 1024 tokens. The option works for scripts/gguf_generate.py
too.

MTP works with --gpu and gives the same tokens. With the drafter on
the GPU (GPUDrafter, scripts/bench_mtp_gpu.py), the 26B gives about 47
tokens/s with MTP, against 45 for the plain decode. The gain is small,
because the verify group sends more experts to the CPU. Thus the server
turns MTP off with --gpu unless NP_GEMMA_MTP=1.
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

