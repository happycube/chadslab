# numpy-gemma — a NumPy-only Gemma 4 12B runtime

This project runs the model google/gemma-4-12B-it-qat-q4_0-unquantized.
It uses NumPy only. It does not use PyTorch. It does not use transformers.
The project is the Phase 1 baseline of the learning plan in
../Gemma LLM Runtime Learning Plan.md.

The runtime does five tasks:
1. Read the SafeTensors weights. Convert bfloat16 data to float32 or int8 data.
2. Run the 48 decoder layers.
3. Keep a key and value cache. Generate tokens one at a time.
4. Change text into token ids. Change token ids into text.
5. Keep the weights in memory. Then later passes are fast.

## Files

    numpy-gemma/
    ├── np_gemma/
    │   ├── st.py          Read SafeTensors files. Use mmap. Convert bf16 to f32.
    │   ├── config.py      Read the configuration. Make a plan for each layer.
    │   ├── ops.py         Give rms_norm, linear, gelu_tanh, softmax, softcap.
    │   ├── rope.py        Make the default RoPE and the proportional RoPE.
    │   ├── model.py       Give KVCache and Model. Run the forward pass.
    │   ├── tokenizer.py   Give the BPE tokenizer and the chat template.
    │   ├── numba_ops.py   Give the Numba JIT kernels. Optional.
    │   ├── cops.py        Build and load the C kernels with ctypes. Optional.
    │   ├── weight_cache.py  Store the converted weights on the disk.
    │   └── csrc/
    │       └── bf16_linear.c  Give the fused bfloat16 and int8 C kernels.
    └── scripts/
        ├── check_trace.py      Compare each intermediate with a HF trace.
        ├── check_cache.py      Compare the KV cache with a batch prefill.
        ├── check_tokenizer.py  Compare the tokenizer with AutoTokenizer.
        ├── generate.py         Generate tokens from token ids.
        ├── chat.py             Generate text from a prompt.
        ├── session.py          Load one time. Then answer many prompts.
        ├── gen_ids.py          Write greedy token ids for a HF comparison.
        ├── bench_numba.py      Compare the NumPy path and the Numba path.
        ├── bench_kernels.py    Compare the NumPy, Numba, and C paths.
        ├── profile_token.py    Time the parts of one decode step.
        ├── profile_int8.py     Time each int8 matrix. Show the bandwidth.
        ├── bench_decode.py     Time each decode step. Show the warm-up.
        ├── bench_ram_cache.py  Compare the memory map and the local memory.
        ├── bench_int8_stream.py  Measure the int8 kernel for each stream size.
        ├── bench_threads.py    Measure the kernel speed against the thread count.
        ├── peak.c              Measure the peak AVX-512 speed.
        └── membw.c             Measure the memory bandwidth.

## Start here

Set the paths one time:

    cd numpy-gemma
    PY=../gemma4-12b-qat-pytorch/.venv/bin/python
    SNAP=../gemma4-12b-qat-pytorch/.cache/huggingface/hub/models--google--gemma-4-12B-it-qat-q4_0-unquantized/snapshots/b6ed86275a6a5735884e208bfed95b445a684ca2
    SNAP4=../gemma4-12b-qat-pytorch/.cache/huggingface/hub/models--google--gemma-4-12B-it-qat-w4a16-ct/snapshots/1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee

SNAP is the unquantized checkpoint. SNAP4 is the 4-bit checkpoint from the
quantization-aware training. Use the full path. The command "find" can give the
wrong model, because the cache holds two models.

Set the thread values. These values give the best speed:

    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=6

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
finds the common prefix of the new turn and the cache. It runs the forward pass
only for the new tokens. Thus a chat does not read the history again. The class
started at 24 of 48 tokens for the second turn of a two turn test. Set the
cache size with --max-len. Use --no-history for independent prompts.

Run the 4-bit mode with the w4a16 checkpoint. This mode gives 48 of 48 tokens
equal to the reference.

    PYTHONPATH=. $PY scripts/session.py --snapshot "$SNAP4" --dtype int4 --max-new-tokens 24 --prompts "The capital of France is"

Change the weight format with --dtype. The default is f32. The choices are f32,
bf16, int8, and int4.

    PYTHONPATH=. $PY scripts/session.py --snapshot "$SNAP" --dtype bf16 --prompts "Hello"

The variable OMP_NUM_THREADS gives the thread count. The default is six. Six is
the best value for the model. The variable OMP_WAIT_POLICY=ACTIVE keeps the
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
    OPENBLAS_NUM_THREADS   1                       One BLAS thread for the attention. Many threads fight the int8 kernel.
    OMP_NUM_THREADS        6                       The thread count of the int8 kernel. Six is the best value.
    OMP_WAIT_POLICY        system                  ACTIVE keeps the threads awake. The median time is better under load.
    NP_GEMMA_CACHE_RAM     0                       1 copies the cache into local memory with large pages.
    NP_GEMMA_CACHE         ~/.cache/np_gemma/weights  The cache directory.
    NP_GEMMA_ARCH          auto                    avx2 or avx512 forces one C library.
    NP_GEMMA_KERNEL        auto                    c, numba, or numpy forces one kernel path.
    NP_GEMMA_INT8_INT      0                       1 uses the integer int8 kernel. That kernel is less accurate.
    NP_GEMMA_PREFILL_CHUNK 256                     The prompt pass uses blocks of this many tokens.

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
   one time with cc. Then it loads the library with ctypes. ctypes is part of
   Python, so no new package is necessary. The C code uses AVX2, FMA, and
   OpenMP. The kernel reads the values and converts them during the multiply.
   It does not write a float32 block.
   The code builds two libraries from the same source. The first library uses
   an AVX2 baseline. The second library uses an AVX-512 baseline. The code reads
   the CPU features at run time. A CPU with AVX-512 loads the AVX-512 library. A
   CPU without AVX-512 loads the AVX2 library. Every target machine gives AVX2.
   Set NP_GEMMA_ARCH=avx2 or NP_GEMMA_ARCH=avx512 to force a library.
   The bfloat16 kernel reads four output rows in one loop. Thus the loop loads
   x one time for four rows.
   The int8 kernel reads one row for each token in a decode step. For a prompt
   the code uses a tiled GEMM. The code uses the GEMM at 32 tokens or more.
   The prompt GEMM keeps the result of four rows and four tokens in a vector.
   The code converts 16 weights with one instruction and uses each converted
   vector for four tokens. The final sum over the columns is horizontal.
   mlp.down_proj is 3.2 times faster at 256 tokens. The full model is about
   1.6 times faster. The AVX2 library keeps the older tile, because the new
   tile needs AVX-512.
   The prompt GEMM then uses three levels of cache. A micro kernel keeps 16
   rows and 16 tokens in the AVX-512 registers. An A panel and a B panel fit
   in the L2 cache. A block over the columns uses the L3 cache. Thus the code
   reads each weight one time. The multi-level GEMM is 1.26 times faster at
   128 tokens and 1.32 times faster at 256 and 512 tokens than the K-vectorized
   tile. The code uses it at 128 tokens or more. A shorter prompt keeps the
   K-vectorized tile. Use cops.set_gemm_ml(False) for a test.
   A long prompt is cut into blocks of 256 tokens. The GEMM is then always in
   its fast range. A test of the large matrices at 1024 tokens gave 1.4 times
   to 2.4 times more speed. Set the block size with NP_GEMMA_PREFILL_CHUNK.

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
add. The integer kernel is faster for one matrix. The activation quantization
causes a larger error. A test gave 29 of 48 tokens equal to the reference. The
integer kernel is not the default. Set the environment variable
NP_GEMMA_INT8_INT=1 to select it.

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
value is nibble - 8. The function writes 32 bytes for each block of 32 values.
Byte j holds value j in the low nibble and value j + 16 in the high nibble.
The function also converts the scales to float32.

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
hardware prefetcher already reads a sequential stream. A manual loop stops the
compiler from vectorizing the code.

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
tokens. The result was 343 GFLOP/s at 32 tokens. A tile of 8 rows and 32 tokens
gave 181 GFLOP/s. That tile is still in the code for the AVX2 library.

A K-vectorized tile then replaced it on AVX-512. The new tile keeps the result
of four rows and four tokens in a vector. One instruction converts 16 weights,
and each converted vector serves four tokens. A test gave this result:

    shape                 tile 16x32    K-vectorized    gain
    gate 15360x3840       174 GB/s      167 GB/s        0.96x
    down 3840x15360        65 GB/s      207 GB/s        3.20x
    o    3840x8192         47 GB/s      131 GB/s        2.80x

The full model was 1.60 times faster for a prompt of 256 tokens. The error is
also smaller, because the tile sums 16 values in a vector. Use cops.set_gemm_kv
to select the older tile for a test.

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

The token-vectorized tile reads the weights only 8 times at 256 tokens. But its
weight convert costs more. A packed weight layout was built to give both. The
pack puts the 16 weights of one column next to each other. One instruction
then converts all 16 values. But the pack belongs to one row block, and the x
data is read again for each row block. The result was 0.40 times the speed for
mlp.down_proj at 256 tokens. The default is off. Use cops.set_gemm_packed for
a test.

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
46 GB/s. A single pass over all the matrices gives 42 GB/s. A plain read of one
large array gives 41 GB/s to 43 GB/s with six threads. Thus the kernel works at
the memory speed of the machine. There is no bandwidth gap to close.

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
TLB entries. The copy gave 1.09 times to 1.24 times more speed in a test. The
copy uses the same memory as the file page cache. The count is larger for a
short time during the load. The default is the memory map.

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
* The attention mask is causal. A sliding layer also masks keys that are older
  than 1024 positions.
* The tokenizer replaces each space with the character U+2581. It uses BPE with
  byte fallback. The decoder replaces U+2581 with a space, joins the byte
  tokens, and joins the parts. In thinking mode, the chat template opens the
  system turn with the think token. In normal mode, it closes an empty thought
  channel.

## Memory bandwidth

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
2. Tune the multi-level GEMM. It reached 441 to 457 GFLOP/s at 256 tokens
   against the measured FMA value of 602 to 676 GFLOP/s. The panel pack and the
   broadcast of each x value control the rest of the time. Try a larger MC, a
   second level of blocking over the columns, and a pack of the B panel for
   each row block. The decode path uses a different kernel, so the prompt work
   did not slow the decode.
3. Make the integer int8 kernel accurate. Use a scale for a group of columns
   for the activations as well as the weights.
4. Add bf16 rounding after each operation. Then the float32 mode follows the
   reference more closely.
5. Port the model to C.

## Sample run

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

