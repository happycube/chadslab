# Plan: split a step across programs, NUMA nodes, and a GPU

## Goal

Run one decode step as several programs that work together. Each program
runs on one part of the machine: one NUMA node, or the GPU. The first use is
a machine with two or more NUMA nodes, where each node reads its own memory.
The second use is a split between the CPU and the GPU, for a model that does
not fit in the memory of the GPU.

The program of PERF_PLAN.md, phase 2, is the base. A step is already one
list of records that one call runs. This plan cuts that list into parts and
adds the operations that move data and wait between the parts.

## Why

A decode step reads every weight of the step one time. Its speed is the rate
of the memory. Two cases:

- A machine with two sockets has two memory controllers. One team of threads
  that reads all the weights from the memory of one node gets about the rate
  of one node. The remote reads also cross the link between the sockets. If
  each node holds its part of the weights and its own threads read it, the
  step can get the rate of both nodes.
- A GPU reads its memory much faster than the CPU. The RTX 5060 Ti of jackal
  gives about 448 GB/s, against about 60 GB/s for the CPU. The step of the
  26B reads 2390 MB. Thus the parts on the GPU get about seven times faster.

## The facts of jackal

    part          value
    CPU           Xeon W-2295, 18 cores, one socket
    NUMA nodes    one
    GPU           RTX 5060 Ti, 16 GB
    GPU memory    about 8 GB free; Xorg and Chrome use about 7.5 GB
    PCIe link     generation 3, 8 lanes (about 7 GB/s)
    CUDA          nvcc 12.9 and 13.1, and NVRTC

Two consequences:

- jackal has one NUMA node. It can test the correctness of a NUMA split, but
  not its speed. A machine with two sockets is necessary for the speed.
- The 26B file is 13.4 GiB. It does not fit in the free memory of the GPU,
  so the 26B needs a split. The E4B file is about 5 GB and fits alone.

## The kinds of split

### Row split: for NUMA nodes

Each node holds a part of the rows of every matrix and computes those rows.
The attention splits by heads, and each node keeps the keys and the values
of its heads. The experts split by expert: each node holds some experts.

Each output value still comes from one thread, as now. Thus a row split
keeps the bits of the program of one node. The sum of the experts is the
exception. A token can use experts on two nodes. The code must then add the
outputs of the experts in the order of the expert index, as today. It must
not add them in the order of the nodes.

After most operations every node needs the whole result, for example the
hidden state before a norm. Such a point needs a barrier across the nodes.
The data stays in memory that all nodes can read. The hidden state is 11 KB,
so a remote read of it costs little.

### Layer split: for a GPU with too little memory

The GPU runs layers 0 to k - 1 and the CPU runs the other layers. The only
data that crosses is the hidden state: 11 KB for each token, one time for
each step. This is the method of llama.cpp (the option -ngl).

### Operation split: for a mixture of experts

The GPU runs the attention, the dense feed-forward part, the router, and
the output head. The CPU runs the experts. For the 26B, the GPU part is
about 1.6 GB and fits in the free memory. The experts are 801 MB of reads
for each token, from about 11 GB of weights. A token uses only 8 of the 128
experts in a layer, so the CPU reads a small part of that memory. This is
the method of the llama.cpp option --n-cpu-moe.

Data crosses the link two times in each layer: the input of the experts to
the CPU, and their output back to the GPU. That is 30 round trips of 11 KB
for each token. The latency of a transfer, not its size, then decides the
cost. On jackal, a round trip takes 12 to 15 microseconds (see the
measurements). Thus the link adds about 0.4 ms to a step of about 25 ms.

In the 26B, the experts and the dense feed-forward part read the same
input: the residual after the attention. Thus the GPU can run the dense part
while the CPU runs the experts. Only the longer of the two is on the
critical path.

### Hot experts on the GPU

The GPU part of the operation split uses about 1.6 GB. About 5 GB of the free
memory of the GPU stays empty. An expert of the 26B is about 2.9 MB, so that
memory holds about 1700 of the 3840 experts. If the router selects some
experts much more often than others, the GPU can hold the experts that it
selects most. Then:

- the GPU runs the selected experts that it holds, and the CPU runs the
  others, at the same time;
- the time of the CPU part falls with the share of the expert reads that
  the GPU holds.

This helps only if the use of the experts is not uniform. Phase 0 measures
the use of each expert on several texts. A set of hot experts from one text
can be cold on a different text. Thus the measurement selects the set on one
text and tests it on the others. The set can also change at run time, but a
copy of an expert over the link takes about 0.4 ms. Thus a change must be
rare.

A token can select experts on both places. The sum of the outputs then adds
values from the GPU and from the CPU. Keep the order of the expert index, as
for the NUMA split. The GPU experts give other bits than the CPU experts.
Thus the MTP verify step must use the same set as the decode step.

## Measurements on jackal

Phase 0 measures these values. The test programs are
scripts/pcie_latency.cu and scripts/gpu_gemv_rate.cu.

    measurement                                              value
    round trip of 11 KB, GPU to CPU to GPU, copy and sync     14.7 us
    the same, a GPU kernel that polls pinned host memory      11.7 us
    copy of 256 MB, host to GPU, pinned memory                6.9 GB/s
    copy of 256 MB, GPU to host, pinned memory                7.2 GB/s
    read of 1.6 GB on the GPU, float4 loads                   425 GB/s
    int4 GEMV, 1.6 GB of Q4_0 blocks, 2816 columns            3.9 ms, 414 GB/s
    int4 GEMV, 55 MB (about one layer)                        0.14 ms, 395 GB/s

The copy of 11 KB takes about 1.6 us at 7 GB/s. The rest of a round trip is
latency, which is almost the same on each generation of PCIe. Thus the link
of jackal is not a problem for the decode step. It is slow for large
copies: the 11 GB of experts take 1.6 s. Thus the prompt pass must not copy
the experts to the GPU for each batch, as llama.cpp can do.

The GEMV on the GPU gets 97% of the rate of the memory. Four lanes share a
block of 18 bytes. The first form of the kernel gave each lane one block. It
got only 58 GB/s, because each read of x then touched 32 cache lines. Thus
the GPU part of the 26B takes about 4 ms, as the estimate says.

### The use of the experts

scripts/expert_use.py runs the prompt pass of the 26B on 800 tokens of four
texts: the README, Python code, C code, and notes in English. It counts the
selections of each expert. The use is far from uniform. In each layer, the
16 most used experts of 128 get about half of the selections.

The next table selects a set of experts on three texts. It gives the share
of the selections on the fourth text that the set gets. The set is global:
a layer can have more experts in the set than another layer.

    experts in the set     readme   python   C      notes
    480 of 3840 (12%)      47%      43%      44%    40%
    960 of 3840 (25%)      65%      62%      64%    56%
    1700 of 3840 (44%)     83%      80%      82%    75%
    2400 of 3840 (62%)     93%      92%      92%    88%

The texts are short, so these values are approximate. The set of 1700
experts fits in the free memory of the GPU. It gets 75% to 83% of the expert
reads. Thus the CPU reads only about a fifth of the expert bytes.

## The design

### A program with places

Each operation of the program gets a place: a node, the CPU, or the GPU. The
builder of a model gives the places with a rule. An example is "the experts
on the CPU, the rest on the GPU". Another is "rows by node". The compiler
then:

1. cuts the list of records into one list for each place;
2. puts a transfer at each edge where data goes from one place to another;
3. puts a wait where a place needs data that another place makes.

The environment of each part holds the same parameters. A call binds them
one time and writes them into each part.

### Places of the operations in a row split

Each operation of a layer gets one of three places:

- all: each part runs the operation on its own copy of the data. This is
  for a small operation, such as a norm, an add, or the router. The part
  writes only its own buffers, so it needs no barrier.
- rows: each part computes a range of the rows of the output. The output is
  one buffer that all parts can read. A barrier across the parts follows.
- one: only part 0 runs the operation, and a barrier follows. This is for an
  operation that changes shared data in place. Examples are the norm and the
  rope of the query, the write to the cache, and the attention.

A layer of the 26B then has these places:

    operation                               place
    input norm                              all
    q, k, v projections                     rows
    norm and rope of q, k; cache write      one
    attention                               one
    o projection                            rows
    norm, residual add                      all
    gate and up projections (with norm)     rows
    GELU, down projection                   rows
    router                                  all
    experts                                 rows (see below)
    norms, adds, layer scalar               all

A range starts at a multiple of 16 rows. The int4 kernels compute four rows
together, and the GELU uses 16 values in a vector. At these boundaries each
part computes each row with the same instructions as one program. Thus the
parts give the same bits.

The experts split by rows too. Each part holds rows of every expert: a range
of the gate and up rows, and a range of the down rows. The steps of a part:

1. Run the gate, the up, and the GELU for its range. Write this part of the
   activation to a shared buffer.
2. After a barrier, run the down rows of its range on the whole activation.
3. Add the outputs of the experts for its range of the output, in the order
   of the expert index.

The load of each part is then the same for each token. A split by expert does not have this
property: a token can select six experts of one part and two of the other.

The rows of a part are a copy of the weights. On a machine with NUMA, the
copy goes to the memory of the node of the part. In phase 1, the copies are
in one memory.

Phase 1 splits only a step of one token. A step of a token group writes its
outputs as (tokens, rows). A range of rows is then not one block of memory,
so the kernels need a row stride of the output first.

### NUMA: nested teams

One OpenMP region has one thread for each node, spread over the nodes. Each
of those threads opens a region with the cores of its node. The kernel
bodies of phase 2a already bind to the inner region, so they need no change.
A barrier across the nodes is an atomic counter in shared memory.

Each node must hold its weights in its own memory. The threads of a node can
write their part first, so that the kernel puts the pages on that node. A
call to mbind through libnuma is the other way. Check the place of the pages
with numastat or move_pages.

### GPU: the same program on the device

The GPU part is a list of kernel calls on the same record format. A CUDA
graph replays the list with one call. The environment lives in the memory of
the device, so a new step writes the parameters and replays the graph. This
is the same design as on the CPU: the state is part of the program.

Build the kernels with NVRTC when the model loads. Another way is nvcc at
build time, as for the C library now. CUDA stays optional. Without it, every
place is the CPU.

A transfer uses pinned host memory and an asynchronous copy. The CPU part
waits for an event of the GPU. The GPU part waits for a flag that the CPU
writes, with a stream wait on a value in host memory. Thus a step stays one
call from Python.

## Results of phase 1

The module np_gemma/parts.py compiles a step of one token into one program
for each part. The function gemma_run_parts in the C file runs the parts in
nested teams. The script scripts/check_parts.py compares each step with the
program of one part.

The compiler keeps the set of the shared buffers that the parts wrote since
the last barrier. It puts a barrier only before an operation that reads one
of them. A layer of the 26B then has five barriers, and the expert operation
has one more inside. The router needs no barrier, because it reads only the
hidden state of its own part.

The bits are the same in every case of the check:

    model   parts   cache    contexts      steps
    26B     2, 3    int16    200, 1100     12
    26B     2       float    200, 1100     12
    12B     2       int16    200, 1100     12

The next table gives the time of a step without the output head, on
jackal. The threads are bound (OMP_PLACES=cores, OMP_PROC_BIND=close). Each
value is the median of 12 steps.

    model   context   one part   2 parts   3 parts
    26B     200       43.7 ms    45.7 ms   48.4 ms
    26B     1100      55.5 ms    59.6 ms   70.0 ms
    12B     200       132.7 ms   135.8 ms  -
    12B     1100      148.5 ms   153.2 ms  -

jackal has one NUMA node. Thus the parts cannot be faster than one team
here. In this first version, the cost of the parts had two causes:

- the barriers across the parts: 180 in a step of the 26B;
- the attention, which ran only in part 0, on half of the cores. This cost
  increases with the context.

### The attention split by heads

The place heads removes the second cause. Part p runs a range of the key and
value heads and the query heads that go with them. It computes the rows of
the query, the key, and the value of its heads. It then runs their norm and
rope, writes its heads to the cache, and runs the attention of its heads.

Each operation reads only the rows that the same part wrote. Thus the
attention needs no barrier before it, and a layer of the 26B has four
barriers and the one in the expert operation. Each part uses its own scores
buffer.

The attention operations GP_ATTN_QC_H and GP_ATTN_F32_H take the row stride
of the cache as an argument. A part reads only some heads of each row. The bits are the same for all the cases of the table above, and for 3
parts of the 12B. NP_GEMMA_PART_ATTN=one gives the first version.

The time of a step, measured as before. Other work ran on jackal (load
average 12), so a difference below 1 ms is noise.

    model   context   cache    one part   2 parts   3 parts
    26B     200       float    42.8 ms    43.7 ms   44.7 ms
    26B     1100      int16    54.3 ms    55.9 ms   59.0 ms
    26B     1100      float    58.1 ms    58.7 ms   60.1 ms
    12B     200       int16    137.9 ms   137.4 ms  145.9 ms
    12B     1100      int16    159.8 ms   161.0 ms  -

At a context of 1100, 2 parts of the 26B now cost about 1.6 ms, against 4.1
ms before. 3 parts cost about 3.5 ms, against 13.6 ms before.

The cache is still one buffer for all the heads. On a machine with NUMA,
each part must hold the cache of its heads in the memory of its node.

It is important to bind the threads of the parts to the cores. For this
reason, an early run with free threads showed two parts faster than one
part. The runner of the parts binds its teams in all cases. The package now
sets OMP_PLACES=cores and OMP_PROC_BIND=close when they are not set.

For one program, a first measurement showed 15 per cent. That run set
OMP_PLACES without OMP_PROC_BIND, and jackal had other load. In a second
run, a step at a context of 200 took 44.6 ms with bound threads and 45.9 ms
with free threads. At 1100 it took 64.6 ms against 62.7 ms. A clean A/B
test on an idle machine is still to do.

The next steps for NUMA:

- a cache for each part, in the memory of its node;
- a copy of the weights of each part in the memory of its node, made by the
  threads of that node;
- the steps of a token group, which need a row stride of the output in the
  group kernels;
- the output head split by rows.

## Results of phase 3: the E4B on the GPU

The file np_gemma/csrc/gpu.cu has a kernel for each operation of the decode
step of the E4B model. The module np_gemma/gpu.py builds it with nvcc, copies the data of the
program to the GPU, and changes the addresses of the records. Each kernel
reads its operands from its record and from the environment in device
memory. Thus the first run records the step in a CUDA graph, and each later
run starts the graph with one call. Only the environment changes.

The design choices:

- The int4 kernels read the float16 scale of each block, not the float32
  copy. This removes 22% of the bytes. The loader checks that the two scales
  are equal.
- The attention of a head goes to 32 blocks, each for a part of the keys.
  A second kernel joins the parts. This is the method of FlashDecoding. One block for
  each head used only 8 blocks, and it took 5 ms at a context of 1100.
- The output head (Q6_K) and the soft cap run on the GPU too. On the CPU,
  the head took 10 to 12 ms for each token.
- The GPU copy of the cache is the true copy while the GPU runs the steps.
  A CPU pass on the same cache first copies it back to the host.

The check (scripts/check_gpu.py) runs the CPU program and the GPU on the
same true tokens:

    context   steps   hidden state, max rel. diff.   same top token   CPU      GPU
    200       32      9.3e-06                         32 of 32         62.9 ms  11.7 ms
    1100      32      1.9e-06                         32 of 32         73.5 ms  12.1 ms

The times include the output head. A greedy generation of 128 tokens with
E4B.forward and E4B.logits gives the same tokens on the CPU and on the GPU.
The decode rate is 84.5 tokens/s on the GPU, against 15.3 on the CPU and
16.2 for llama.cpp on the CPU. The first GPU step takes 2.7 s: it copies the
weights (3.2 GB with the head and the buffers) to the GPU.

The next table shows where the time goes, at a context of 200. The profile
records an event for each record. The events add about 2.7 ms to the small
operations.

    operation              count   time      rate
    int4 gate and up       42      3.2 ms    389 GB/s
    int4 down              42      1.7 ms    359 GB/s
    other int4 matrices    ...     1.8 ms    57 to 333 GB/s
    norms, adds, and other small operations   about 3 ms in the profile
    output head (not in the graph)            1.9 ms

The graph alone takes 8.8 ms, and it reads 2.29 GB. At 414 GB/s that is 5.5
ms. The small matrices of the per-layer inputs (256 by 2560) and the small
operations (about 700 kernels) take most of the rest. Fusing them is the
next work for the E4B. NP_GEMMA_GPU=1 turns the GPU on for E4B.forward. The
MTP drafter reads the cache in host memory, so NP_GEMMA_GPU=1 turns MTP off.

## Results of phase 4: the 26B with the experts on the CPU

The compiler of the split (SplitCompiler in np_gemma/gpu.py) compiles the
step of the 26B for the GPU with the float cache. The operation moe becomes
three records:

- GP_TO_HOST copies the input of the experts to pinned host memory. It also
  copies the weights and the indices that the router selected. Then it
  records an event.
- GP_CPU_JOIN comes before the first operation that reads the output of the
  experts. It waits for the event and runs a CPU program with the MOE record
  of the CPU interpreter.
- GP_TO_DEV copies the output of the experts to the GPU.

The form of a layer now puts the router and the experts before the dense
feed-forward part. The two parts read the same input, so the order does not
change the bits of the CPU program (scripts/check_program.py passes). On the
GPU, the runner launches the dense part before it runs the CPU program. Thus
the GPU computes the dense part while the CPU computes the experts. The
kernels between two records of the handoff are one segment with its own CUDA
graph: 61 segments for a step of the 26B.

The GPU keeps its own float cache (GPUKV). For a layer with a window,
the GPU drops the oldest rows, as KVCache.prepare does on the host.
The function detach() writes the new rows into the host cache with
KVCache.write, which also makes their int16 copy. A CPU step after detach agrees with a CPU step on the CPU cache.

scripts/check_gpu_split.py, with the true tokens as input:

    context   steps   logits, median |d|   same top token   CPU      GPU and CPU
    200       32      0.001                 32 of 32         56.0 ms  25.0 ms
    1100      32      0.001                 32 of 32         66.1 ms  26.1 ms

The times include the output head. At a context of 1100, one step had a
difference of 3.1 in the logits: the router selected a different expert.
A greedy generation of 128 tokens with scripts/gguf_generate.py gives the
same tokens with NP_GEMMA_GPU=1 and without it. The GPU holds 0.98 GB of
weights, 0.6 GB for the head, and the cache: 2.9 GB in all.

A step takes 25 ms, or 40 tokens/s, against 17.9 tokens/s on the CPU. This
is the value of the estimate. The experts on the CPU take most of the time.

### The hot experts on the GPU

The GPU holds the most used experts of a file of counts (pick_hot in
np_gemma/gpu.py). The set is global over the layers, up to a budget of
memory. For each layer, the step has these records:

1. GP_HOT_SPLIT writes the selected experts that the GPU does not hold (the
   cold experts), their weights, and their count.
2. GP_TO_HOST copies the input and the cold experts to the host.
3. GP_HOT_MOE computes the hot experts on the GPU, in four kernels.
4. The dense feed-forward part runs on the GPU.
5. GP_CPU_JOIN runs GP_MOE_N on the CPU: the MOE record with a count that it
   reads from memory. GP_TO_DEV copies its output back, and an add joins the
   two parts.

The file np_gemma/data/gemma-4-26B-expert-counts.npz holds the counts of the
four texts of phase 0. The test below selects the set without the counts of
the README, and it runs on the README. Thus the set did not see the text.

    budget   hot experts   context   same top token   GPU and CPU
    0        0             200       32 of 32         25.0 ms
    3 GB     896           200       32 of 32         15.0 ms
    3 GB     896           1100      32 of 32         15.4 ms
    4 GB     1195          200       32 of 32         14.1 ms
    4 GB     1195          1100      32 of 32         14.5 ms

The times include the output head. With 3 GB, a step takes 15 ms: about 65
tokens/s, against 17.9 on the CPU and 19 for llama.cpp on the CPU. A greedy
generation of 128 tokens with the default settings gives the same tokens as
the CPU.

With 4 GB, the free memory of the GPU falls to 0.8 GB. The GPU of jackal
also drives the display, so the default budget is the free memory less 4.5
GB (about 3.4 GB on jackal). The first step takes about 6.5 s: it copies the
rows of the hot experts and moves them to the GPU.

A profile of the step runs the records one at a time. In it, the CPU part
of the experts takes about 0.27 ms for each layer, 8 ms in all. It is on the
critical path. More hot experts, or a faster CPU part, make the step
shorter.

### Against llama.cpp with CUDA

The two programs ran one after the other on jackal, which had other load
(load average 8 to 10). The tool llama-bench (build-cuda) runs tg128 and
pp512 with 18 threads and flash attention. The script scripts/bench_decode.py
measures
the decode of this runtime in the same way: 128 tokens from a short context,
with the output head.

    split                                    llama.cpp tg128   this runtime
    CPU only (-ngl 0)                        13.3              14.8
    experts on the CPU (-ncmoe 30, dense)    36.8              39.2
    about 3.3 GB of experts on the GPU       43.1              47.7
    12 layers on the GPU (-ngl 12)           21.0              -

For llama.cpp, "-ncmoe 22" puts all the experts of 8 layers on the GPU
(about 3.4 GB). This runtime puts the 952 most used experts on the GPU
(about 3.2 GB). The layer split of llama.cpp gives half the rate of the
operation split. Thus this runtime does not add a layer split.

The prompt pass is different. llama.cpp gives 255 tokens/s with -ngl 0,
301 with -ncmoe 30, and 407 with -ncmoe 22. The large products of a prompt
go to the GPU, and the weights of the experts cross the link for each batch
of 512 tokens. The prompt pass of this runtime runs on the CPU: about 61
tokens/s. Phase 5 moves it to the GPU.

## The cache on the GPU in int16

The GPU keeps the cache of the 26B in the int16 form of the CPU program
(NP_GEMMA_GPU_KV=int16, the default). Each group of 32 values has a float32
scale of max |x| / 32767. The kernel of the cache write quantizes a row as
gemma_quant_group32_i16 does. The GPU keeps no float rows. detach() gives
the host the int16 values times their scales.

The size of the cache for 262144 tokens (256k):

    layers               float     int16
    5 global             10.7 GB   5.7 GB
    25 with a window     0.85 GB   0.46 GB

The global layers use the key as the value, but the cache still keeps both.
One copy can halve the global part again.

### The attention kernel at a long context

The first attention kernel took 4.9 ms for each global layer at a context of
64k: about 58 GB/s. Three changes made it 1.0 ms (about 285 GB/s):

1. A block takes a chunk of keys of one key and value head, for all its
   query heads. A global layer of the 26B has 8 query heads for each key
   and value head. Thus the block reads each row one time, not 8 times.
2. Each lane reads 16 bytes (8 values) with one load, not one value.
3. A warp takes 4 keys at a time. The loads of the 4 keys come first, and
   the sums of the lanes of the 4 keys follow each other.

A chunk has n / 256 keys, but at least 32. The join of the chunks computes
the weight of each chunk one time, in shared memory.

### The test at 64k tokens

scripts/check_gpu_long.py builds a cache of 65536 positions. It runs a real
prompt pass of 2048 tokens and copies those rows up to 65536 positions. A
real prompt pass of 64k tokens takes too long on the CPU. The CPU program and
the GPU read the same cache. The times include the output head:

    run                                  step       cache     GPU memory   same top token
    CPU program                          260 ms     -         -            -
    GPU, float cache                     33.5 ms    3.97 GB   5.90 GB      8 of 8
    GPU, int16 cache                     31.5 ms    2.11 GB   4.12 GB      8 of 8
    GPU, int16 cache, 3 GB hot experts   22.9 ms    2.11 GB   7.21 GB      8 of 8

At a context of 64k, the attention of the step takes about 6.5 ms. The
experts on the CPU take most of the rest.

## Results of phase 5: the prompt pass on the GPU

The program of a group of t tokens (the group form of the layers) now runs
on the GPU. New kernels:

- k_mt_gemv: a group of up to 16 tokens, one warp for each row of a matrix;
- k_gemm: a larger group, tiles of 64 tokens by 64 rows;
- k_flash_qc_mt: the causal attention of a large group over the int16
  cache. It uses tiles of 32 queries by 32 keys. A small group uses
  k_attn_qc_mt;
- the router of a group: the norm, a product with the matrix of the router,
  and the top experts of each token;
- GP_MOE_GPU: the experts of a large group. The pairs (token, slot) go in
  the order of the experts. Then a tile of 64 pairs of one expert reads the
  weights of that expert one time.

The weights of the experts stay in host memory. GP_FETCH copies the cold
experts of a layer to one of two device buffers. A worker thread does the
copies, so the GPU computes layer l while the link copies layer l + 1. The
hot experts of the decode step are on the GPU already, and GP_MOE_GPU reads
them there. A table gives the device address of each expert.

The compiler of a group uses the same buffers for every layer (the n-th
buffer of a shape in a layer). A program of 1024 tokens then needs 0.52 GB,
not about 9 GB.

ModelGPU.prefill runs a prompt in chunks of 1024 tokens
(NP_GEMMA_GPU_CHUNK). A part shorter than 128 tokens
(NP_GEMMA_GPU_PREFILL_MIN) goes in groups of 16 tokens with the experts on
the CPU. The copy of the experts takes about 1.9 s, which is more than the
time of the CPU for such a part. Thus a short turn of a chat stays on the
GPU, and the cache does not move.

The accuracy: the test compares 256 rows of a prompt of 512 tokens. The
top token of the GPU pass agrees with the float prompt pass of the CPU (mode
0) for 100% of the rows. The CPU pass of mode 16 agrees for 97.7%. A two-turn chat and a
prompt of 1681 tokens give the same tokens as the CPU.

The speed of the prompt pass of the 26B on jackal:

    prompt                 CPU        GPU               llama.cpp (-ncmoe 30 / 22)
    512 tokens             ~65 tok/s  266 to 317 tok/s  301 / 407 (-ub 512)
    1024 tokens            ~65 tok/s  510 to 593 tok/s  545 / 735 (-ub 1024)
    1681 tokens, chat      60 tok/s   441 tok/s         -

A profile of a chunk of 1024 tokens runs the records one at a time. The
experts take 651 ms, the other products 457 ms, and the attention 221 ms.
The copy of the experts takes about 1.9 s in all. The copy runs at the same time
as the compute, so the copy limits a chunk of 1024 tokens. A longer chunk,
or fewer bytes of experts to copy, makes the pass faster. The kernels do not
use the tensor cores yet. llama.cpp uses them, and it holds all the experts
of some layers on the GPU.

### A fault in the copy of the arrays

The first checks of a group of 40 tokens showed a difference of up to 14%
from 40 steps of one token. A new record of a small group then gave an
illegal address.

The cause was the same: Mirror registered a view of an
array (for example one row of a buffer) as an array of its own. The view of
row 0 has the start of the buffer, so the buffer got a device copy with the
size of one row. Mirror now registers the array that owns the memory. A
group of 40 tokens then agrees with 40 steps to 4e-4.

### MTP with the GPU

The verify group of MTP (2 to 16 tokens) runs on the GPU. The GPU computes
the hot experts. It sends the other pairs (token, slot) to the CPU. The
records are GP_HOT_SPLIT_MT, GP_HOT_MOE for a group, and GP_MOE_MT, which
skips the pairs of -1.

Each query of the group runs the attention of a decode step. The output
head computes the rows of the group in one pass over the head. The drafter
runs on the CPU. It reads the cache of two layers, so the GPU writes the new
rows of those layers into the host cache after each step.

The tokens are the same as the plain decode. The speed is not better:

    part of an MTP step (2 drafts)   time
    2 drafts on the CPU              14.6 ms
    verify group of 3 tokens         38.8 ms
    output head, 3 rows              3.7 ms

A step gives about 2.25 tokens in 57 ms, about 39 tokens/s. The plain
decode on the GPU gives about 47. The verify group sends about three times
the cold experts of a step to the CPU. A drafter on the GPU saves about 13
ms, which gives about the rate of the plain decode. Thus MTP is off by
default with NP_GEMMA_GPU=1. NP_GEMMA_MTP=1 turns it on.

## The E4B on the GPU: tensor cores and the drafter

scripts/bench_e4b_gpu.py measures the E4B model, which fits wholly on the
GPU. Thus no expert goes to the CPU, and the numbers show the kernels.

### The prompt pass

The E4B now runs its groups of tokens and its prompt pass on the GPU. New
kernels: the bfloat16 products of a group, and FlashAttention for the float
cache of the E4B (heads, positions, head_dim).

k_gemm_tc computes the int4 products of a large group on the tensor cores,
with mma.sync m16n8k16. The int4 values are exact in float16, and the kernel
applies the scale of each block in float32. The rows of x become float16 one
time for each product (k_to_half). A block of 8 warps takes 128 tokens by 64
rows. The kernel k_flash_tc computes the attention of a large group on the tensor cores,
as FlashAttention-2 does it.

    E4B, prompt pass   CPU        GPU float32   GPU tensor cores   llama.cpp (CUDA)
    512 tokens         75 tok/s   628 tok/s     1417 tok/s         4735 tok/s
    1024 tokens        79 tok/s   616 tok/s     1509 tok/s         5029 tok/s

Against the float32 kernels of the GPU, 99.9% of the top tokens of a prompt
of 1024 tokens agree (median relative difference 9e-4). The kernels of
llama.cpp are still three times faster.

The 26B keeps the float32 kernels for its prompt pass. Some activations of
its global layers are larger than the range of float16. Thus the products
give infinities, and 93% or fewer of the top tokens agree with the CPU. The copy
of the experts limits that pass anyway. Each program has its own switch
(GPUProgram tc). NP_GEMMA_GPU_TC=0 turns the tensor cores off for all.

A small group (at most 16 tokens, such as an MTP verify group) keeps the
float32 kernels and the decode attention of each query. Thus a verify group
agrees with the decode steps (6e-6). k_mt_gemv now fixes the token count
when it compiles (1 to 16). The general form ran at half the rate of one
token. A group of 3 tokens of the E4B went from 22.9 ms to 15.2 ms.

### The limit of the tensor cores

scripts/mma_peak.cu measures the peak rate of mma.sync on this GPU:

    inputs    sums      rate
    float16   float32   37 TFLOPS
    float16   float16   38 TFLOPS
    int8      int32     204 TOPS

The products of the prompt pass of the E4B need about 8 TFLOP for 1024
tokens. The kernel k_gemm_tc took 263 ms (30 TFLOPS).

The kernel k_gemm_tc2 copies each step with cp.async into two buffers, so
the copy of the next step runs during the compute. It uses tiles of 128
tokens by 128 rows. It makes the float16 values of the weights in registers
from the 4-bit numbers. It takes 243 ms: 33 TFLOPS, about 89% of the peak of
float16. Thus a better float16 kernel cannot give much more.

The kernel k_gemm_q8 (NP_GEMMA_GPU_TC=8) quantizes the rows of x to int8.
Each block of 32 values has a scale, as in the Q8_0 form of ggml. The instruction
m16n8k32 covers one int4 block, and its int32 sum is exact. The kernel then
applies the two scales in float32. This is the method of llama.cpp.

    E4B, 1024 tokens     products   pass         top token as float32 GPU
    float32 kernels      -          589 tok/s    -
    float16 (default)    243 ms     1527 tok/s   99.9%
    int8                 150 ms     1756 tok/s   98.6%

The int8 products take 150 ms, about 26% of the peak of int8. The
multiplication by the two scales after each block costs as much as the
products. The attention (80 ms) and the small operations (about 45 ms) now
take as much time as the products. The default stays float16, because it
stays closer to the float32 result.

### A comparison with llama.cpp in nsys

Nsight Systems recorded the prompt pass of 1024 tokens of the E4B, in
llama.cpp (llama-bench -p 1024 -fa 1) and in this runtime (int8 products).
The option --cuda-graph-trace node records each kernel in a CUDA graph. The
times are for one pass:

    part                          llama.cpp   this runtime   after the fixes
    products, with the quantize   128 ms      161 ms         156 ms
    attention                     9 ms        88 ms          16 ms
    projection of the layer input 1 ms        11 ms          11 ms
    small operations              30 ms       42 ms          42 ms
    all the GPU kernels           171 ms      302 ms         224 ms
    rows of the embeddings        on the GPU  343 ms (CPU)   7 ms (CPU)
    the pass                      200 ms      600 ms         255 ms

The products were not the cause of the difference. Two things were:

1. E4B.embed_rows decoded the Q6_K rows of the two tables of the embeddings one
   token at a time in NumPy. The GPU waited for 343 ms. GGUF.take_rows now
   gathers the rows, and the C function gemma_q6k_rows decodes them in
   parallel. It takes 7 ms and gives the same values.
2. The kernel k_flash_tc used 2 warps for each block, and it read 16 keys
   in each step with no copy ahead. Each query head read the keys again. It
   wrote the values to shared memory in columns, so the writes hit the same
   bank. The new kernel k_flash_f32h (GP_ATTN_F32H) gives the same values:
   - One block takes the 4 query heads of a key head, so it reads the keys
     and the values one time for 4 heads.
   - cp.async copies the next step of keys and values during the compute
     (two buffers for heads of 256 values; one buffer for 512).
   - The rows in shared memory have a pad, so the reads do not hit the
     same bank. The fragments become float16 in registers.
   The attention goes from 88 ms to 16 ms. NP_GEMMA_GPU_FLASH=1 selects the
   old kernel.

    E4B, 1024 tokens        before       now          llama.cpp
    float16 products        1527 tok/s   2902 tok/s   -
    int8 products           1756 tok/s   4093 tok/s   5113 tok/s

The layers with heads of 512 values take 9 of the 16 ms of the attention.

Two more changes followed:

1. The projection of the layer input has a bfloat16 matrix, and k_gemm<2>
   computed it in float32 (11 ms). The kernel k_gemm_bh computes it on the
   tensor cores. It copies the bfloat16 rows with cp.async and changes each
   fragment to float16 in registers. It takes 1.8 ms.
2. The GPU group of the E4B uses fused operations (e4b_layer_form with
   fused=True). GP_ADD_NORM does the norm of o, the add to x, and the
   multiplication by the layer scalar in one pass. GP_GELU_MUL_ROWS does
   GELU and the product with u. The small operations go from 39 ms to 22
   ms. The fused kernels do the same operations in the same order, with no
   fused multiply-add, so the values are the same as before.

    E4B, 1024 tokens    GPU kernels   pass         top token as float32
    float16 products    -             3107 tok/s   99.9% (before: 99.9%)
    int8 products       197 ms        4492 tok/s   98.6% (before: 98.6%)
    llama.cpp           171 ms        5178 tok/s   -

A first test gave 98.0% for int8. That test used an old output. The
output came from before the fix of the fused multiply-add in GP_ADD_NORM. The products and
the quantization of x now take about 26 ms more than in llama.cpp. The
other kernels take about the same time as in llama.cpp.

### The int8 products

These changes give the same values as before:

- k_quant_q8 reads 4 values with each thread (float4), not one. A record
  of GP_INT4_MULTI4_MT quantizes x one time for all its matrices (q, k, v;
  gate, up). The quantization goes from 15.6 ms to 6.8 ms (llama.cpp: 7.2).
- k_gemm_q8 takes 128 columns in each step, not 64. The 72 bytes of a row
  of w in a step then start at a multiple of 8. Thus 9 copies of 8 bytes
  take them, not 18 copies of 4 bytes. It has 3 buffers in dynamic shared memory
  and one barrier in each step.
- The grid puts the tiles of tokens on x, so the blocks that read the same
  rows of w run together.
- i2f_exact changes the int32 sums to float32 with an add, not with I2F.

The product kernel goes from about 145 ms to 137 ms (llama.cpp: 120). The
pass of 1024 tokens goes from 4492 to about 4650 tok/s.

Tests that removed a part of k_gemm_q8 (the results were wrong, only the
time counts) show where the time is. The kernel has no access to the
counters of Nsight Compute on this machine (RmProfilingAdminOnly).

    part removed                     gate and up, 42 records
    nothing                          75.6 ms
    the scale of each block          66.6 ms
    the mma instructions             58.7 ms
    the copies of x                  62.2 ms
    the copies of w                  31.9 ms
    all the compute (copies only)    39.9 ms

The copies of w cost the most. The kernel does not overlap the copies and
the compute well, and more buffers or a second block on each SM did not
help. The card also runs at its power limit (about 170 of 180 W, 2.6 to
2.85 GHz). The next test is the tile of llama.cpp: w changed to int8 in
shared memory one time for each block, not in each warp.

### The decode step, against llama.cpp

Nsight Systems recorded 64 decode steps of the E4B in each runtime. It
recorded each kernel in the CUDA graphs (llama-bench -n 64 -fa 1; this
runtime with NP_GEMMA_GPU=1). The machine had other load, so llama.cpp gave 94 tok/s.
The values are for one step:

    part                           llama.cpp    this runtime
    the step                       10.5 ms      13.7 ms
    sum of the kernels             10.8 ms      10.8 ms
    time with a kernel that runs   8.7 ms       10.8 ms
    GPU waits for cudaGraphLaunch  1.6 ms       1.5 ms
    GPU waits for the host         0.2 ms       1.0 ms

    kernels                        llama.cpp    this runtime
    int4 products and output head  8.1 ms       8.0 ms
    quantization of x (q8_1)       1.0 ms       -
    norms                          1.0 ms       1.4 ms
    attention                      0.3 ms       0.6 ms
    other small operations         0.3 ms       0.8 ms

The kernels take the same time in the two runtimes. The differences are:

1. The kernels of llama.cpp overlap on one stream (about 1100 of them in
   each step). This is programmatic dependent launch: a kernel starts
   before the kernel before it ends, and it waits for the data with
   cudaGridDependencySynchronize. It hides about 2.1 ms in each step.
   The kernels of this runtime do not overlap.
2. Between the logits of a step and the start of the next step, the host
   of this runtime takes 1.0 ms. llama.cpp takes 0.2 ms. The host work is
   the argmax of the logits in NumPy, the rows of the embeddings, and the
   tables of RoPE. It also binds the parameters of the step.
3. cudaGraphLaunch takes about 1.5 ms for a graph of about 1000 kernels in
   the two runtimes, and the GPU does not start before it ends.

Thus programmatic dependent launch (about 2 ms) and a faster host part
(about 0.8 ms) give the most. A graph with fewer kernels also makes the
launch shorter, at about 1.5 us for each kernel.

The changes, in order, with the decode rate of the E4B (128 tokens):

    change                                               decode rate
    before                                               84 tok/s
    programmatic dependent launch (PDL)                  85 tok/s
    fused forms in the decode step (add_norm, gelu_mul)  88 tok/s
    a head kernel for one row, pinned logits             92 tok/s
    add_norm2: the next norm in the same kernel          93.5 tok/s
    an argmax in C with AVX2 (0.21 ms to 0.04 ms)        94.5 tok/s
    llama.cpp, at the same time                          112 tok/s

- PDL: each kernel starts with griddepcontrol.wait, then
  griddepcontrol.launch_dependents. After the capture of a graph,
  gg_pdl_edges changes each edge from a kernel to a kernel into a
  programmatic edge. Alone, it gave little, because the kernels of this
  runtime had almost no gap between them. With fewer small kernels, it
  gives about 4 tok/s. A prefetch of the rows of the matrix to L2 before
  the wait did not help, so it is not in the code.
- A long run of records now becomes several graphs (32 records, then 160).
  Nsight Systems then showed a shorter wait for cudaGraphLaunch. But the
  rate without Nsight Systems did not change: the trace of each node of a
  graph makes the launch slower.
- The fused kernels give the same values. MTP gives the same tokens as the
  plain decode (128 tok/s with 2 drafts).

The kernels of a step now take about 9.9 ms, and the host part about 0.5
ms. The rest of the difference is in the kernels. These are the norms and
the small operations that remain, and the attention of one token (13 us
for each layer). The head takes 1.55 ms, against 1.3 ms.

### The drafter on the GPU

GPUDrafter compiles one draft step of the E4B assistant for the GPU. The
step has the four layers and the attention over the cache of the target on
the GPU. Then comes the centroid head: GP_F32_LINEAR, then GP_DRAFT_HEAD. The
second record sorts the centroids (a bitonic sort) and keeps the top 32. It
computes the logits of their 4096 tokens and writes the best token.

The drafter reads the cache of the target on the GPU, so it needs no copy
of the cache. Its weights are int4 blocks with float16 scales. A draft takes
about 0.5 ms, against about 7 ms on the CPU.

    E4B, decode of 128 tokens         rate          drafts accepted
    plain decode on the GPU           84 tok/s      -
    MTP, 1 draft, drafter on the GPU  117 tok/s     81%
    MTP, 2 drafts                     124 tok/s     76%
    MTP, 3 drafts                     121 tok/s     69%
    llama.cpp (CUDA), plain decode    112 tok/s     -

MTP gives the same tokens as the plain decode.

GPUDrafter also takes the drafter of the 26B. That drafter has a full head,
not a centroid head. The step computes the int4 product with the full head
and then finds the best token with GP_ARGMAX. The attention reads the int16
buffers of GPUKV for the two layers of shared_layers (GP_ATTN_QC). The step
binds the address of the first row in the window and the count of rows.
The script scripts/bench_mtp_gpu.py measures it:

    26B, decode of 128 tokens         rate          drafts accepted
    plain decode on the GPU           45.1 tok/s    -
    MTP, 1 draft, drafter on the GPU  47.0 tok/s    73%
    MTP, 2 drafts                     47.4 tok/s    62%
    MTP, 3 drafts                     42.7 tok/s    50%

MTP gives the same tokens as the plain decode. The gain is small, because
the verify group sends the cold experts of each token to the CPU. The
drafter on the GPU removes the cost of the drafts (about 14 ms in a step
with the CPU drafter). It does not remove the cost of the group. Thus MTP stays
off by default with NP_GEMMA_GPU=1.

### MTP of the 26B: the reuse of the experts of the first token

The verify group of MTP runs the cold experts of each of its tokens on the
CPU. The CPU reads each expert one time for all the tokens that use it.
Thus the cost is the count of different cold experts in each layer. A test
on 160 tokens of a chat answer (1.5 GB of hot experts) gave:

    tokens in the group    different experts   cold experts
    1                      8.0                 5.4 (2 GB hot)
    2                      12.5                8.9
    3                      16.0                11.7

A test (gg_set_reuse, scripts/bench_mtp_reuse.py) lets each draft token of
a group select only from the experts that are already there. These are the
experts of the first token and the hot experts. The router of the group
(k_router_top_mt) sets the other logits to -inf before the softmax. The
first token stays exact.

The value 1 + m of gg_set_reuse also lets each
draft token keep its own m best experts. The result is then not the result
of the model, so the test also runs the exact model on the new text. It
gives the share of the tokens that are the best token of the exact model.
It also gives the mean of log p(best) - log p(token).

    26B, 1.5 GB hot, 3 prompts   rate         best token   log p gap
    plain decode                 45.8 tok/s   100%         0
    exact MTP, 2 drafts          53.3 tok/s   100%         0
    reuse, 2 drafts              76.0 tok/s   81.2%        1.17 nats
    reuse + own best 2, 2 drafts 70.9 tok/s   93.8%        0.12 nats
    reuse + own best 4, 2 drafts 65.8 tok/s   98.2%        0.02 nats
    reuse + own best 4, 3 drafts 61.3 tok/s   97.9%        0.02 nats

The reuse of only the experts of the first token is fast, but the text
becomes bad. A draft token keeps only about half of its own experts, and
the text had errors ("Here are the seven planets"). When each draft token
keeps its own 4 best experts, about 75% of its experts stay. The rate is
23% more than exact MTP and 44% more than the plain decode. The exact model
selects 98.2% of the tokens itself, near the 98.6% of the int8 prompt pass
of the E4B. The text of the three prompts was correct.

The accepted draft tokens also keep their keys and values in the cache,
so a change stays in the context. The test does not turn the reuse on for
the server. For that, the verify step must turn it on and the prompt pass
must turn it off. A prompt of fewer than PREFILL_MIN tokens runs as groups
of up to 16 tokens, and those use the same router.

### The hot experts follow the text (HotCache)

The fixed set of hot experts comes from the counts of four texts. The test used
six chat answers of 256 tokens. With 1.5 GB of hot experts (448
experts), 6.5 of the 8 experts of each layer and step ran on the CPU. The best fixed set for
each answer, known after the fact, left 3.25. A simulation on the
selections of those answers compared ways to change the set as the text
goes:

    1.5 GB (448 experts)            cold experts, each layer   copies, each token
    fixed set (before)              6.52                       0
    best fixed set, after the fact  3.25                       0
    set from the prompt             4.90                       0
    LRU                             3.41                       112
    LFU, decay 0.97, one pool       3.24                       14
    LFU, decay 0.97, each layer     3.32                       12
    the same, at most 8 copies      3.40                       8

HotCache (np_gemma/gpu.py) is the last one. Each layer keeps its count of
slots. GP_HOT_SPLIT also writes the selection of the step to the pinned
array of the host. After a step, the host adds 1 to the score of each
selected expert, after it multiplies all the scores by the decay. A selected cold
expert can have a higher score than the lowest expert in the slots of its
layer. Then it takes that slot:

1. The slot table marks the old expert cold, on the stream of the programs.
2. A worker thread copies the new expert to the slot on a stream of its
   own (gg_cache_copy).
3. When the copy is done, a later step marks the new expert hot.

A large group (a prompt pass) waits for the copies, then fills its tables
of the addresses of the experts again (fill_tables). The list of copies of
the cold experts has a fixed size, so the program keeps it.

    26B, 1.5 GB hot, 255 tokens     fixed set    HotCache     best, after the fact
    CPU cache answer                48.3 tok/s   57.6 tok/s   66.3 tok/s
    story                           41.7 tok/s   59.0 tok/s   70.1 tok/s
    Rust                            51.3 tok/s   60.9 tok/s   68.5 tok/s

The tokens are the same as with the fixed set. The rows of a large group
are the same bits. MTP with 2 drafts gives 68.5 tok/s against 58.3 for the
plain decode, with the tokens of the plain decode.

Some tests that did not help:

- Pinned memory for the experts (12.8 GB, 6.6 s to copy): the same rate.
  The file map cannot be pinned (cudaHostRegister: operation not
  supported), and the link gives only 6.9 GB/s pinned against 5.4 GB/s.
- Copies in pieces of 128 KB or 32 KB: the same or slower.

The cost was the host part of HotCache. At first it took 1.4 ms in each
step. With NumPy on all the layers at one time it takes 0.5 ms. It now runs
while the GPU runs the head of the step, so it adds almost nothing. If the
set does not change after 64 tokens, the rate is 63.7 tok/s.

A second test: a cold expert goes to the GPU only from its second (or
third) use while it is cold. Its first uses run on the CPU
(NP_GEMMA_GPU_HOT_ADMIT):

    26B, 1.5 GB hot     uses before a copy   rate          copies, each token
    CPU cache answer    1 / 2 / 3            56.0 / 56.2 / 56.2   8.0 / 7.9 / 7.6
    story               1 / 2 / 3            58.0 / 58.8 / 58.9   7.8 / 7.4 / 7.1
    Rust                1 / 2 / 3            59.9 / 60.0 / 60.2   7.5 / 7.0 / 6.7

The cold experts of each layer and the tokens do not change. The copies go
down by 1% to 7% with 2 uses.

The score already does most of this work. An
expert after one use has a score of about 1. It takes the place of a held
expert only if that score is lower. Also, the limit of 8 copies for each
step is almost always full. Thus the rule changes which experts come to
the GPU more than their count. The default is 2.

A third test: the scores start from the routers of the prompt pass. Each
group counts the selections of its routers on the GPU (GP_COUNT, after each
router). After the group, HotCache.seed adds the counts to the scores. The
tokens count as the last steps. Then up to 128 slots change:

    26B, 1.5 GB hot         first 64 tokens         all 255 tokens
                            no seed    seed         no seed    seed
    CPU cache answer        -          51.4         -          53.0
    story                   49.6       52.5         57.0       57.5
    Rust                    53.6       54.6         59.8       60.2
    README, 1500 tokens     50.8       47.9         53.5       52.6

A short prompt gives a small gain at the start. A long prompt gives a loss.
Its text is not the text of the answer (a summary of the README). Also, 128
slots are empty until their copies arrive, during the first steps. Thus the
seed from a prompt is off by default (NP_GEMMA_GPU_HOT_SEED=0).

The counts of the groups stay for MTP. MTP runs verify groups, not decode
steps, so observe() never runs, and before this the slots did not change.
Now each verify group changes up to 8 slots:

    26B, 1.5 GB hot, story, a new HotCache   MTP, 2 drafts
    fixed set                                41.0 tok/s
    HotCache, with the counts of the groups  54.3 tok/s

The tokens and the accepted drafts are the same.

### A difference from one run to the next

Two runs of the same decode steps of the 26B gave results that differed by
up to 14%. The attention of a decode step added the sums of its groups of
threads with an atomic add, in the order of arrival. The last bit of the
result then changed from run to run, and the router of the 26B sometimes
selected a different expert. The groups now add their sums in a fixed
order, and three runs give the same bits. The earlier notes of a difference
of 14% for a group of 40 tokens came from this cause too.

## Verification

- A row split on NUMA nodes keeps the bits. `scripts/check_program.py`
  compares the program of one node with the program of two emulated nodes.
  On jackal, the two nodes are two teams of 9 cores on the same node.
- A GPU part does not keep the bits of the CPU, because the sum order of its
  kernels is different. Compare it with a tolerance against the CPU program,
  and with `scripts/check_hf_decode.py` against the reference. For the 26B,
  use the share of the same most probable token, as in PERF_PLAN.md.
- MTP gives the same tokens as the plain decode only when the verify group
  and the decode step use the same kernels. Keep both on the same places.

### Node-local weights (phase 2, first part)

np_gemma/numa.py finds the node of the team of each part (gemma_part_cpus
gives the CPU of each thread, with the teams of gemma_run_parts). The rows of
the int4 matrices and of the experts that a part reads are copies, made with
an anonymous mmap and mbind (MPOL_PREFERRED) before the first write. Thus
each page is on the node of the part, also when the main thread makes the
copy. move_pages checks the place of each page. NP_GEMMA_NUMA=0 turns it
off. The cache and the small weights of the place all stay where they are.

The machine: two Xeon Platinum 8268 (24 cores each, two nodes), 251 GB. The
26B Q4_0, 48 threads, a context of about 60 tokens, the median of a step
(forward) and of the output head (logits):

    form                          forward    logits
    one team                      32.0 ms    5.1 ms
    2 parts, NP_GEMMA_NUMA=0      32.8 ms    7.8 ms
    2 parts, node-local           28.5 ms    7.9 ms
    4 parts, node-local           26.2 ms    7.3 ms

At a context of 200, a step of 2 parts goes from 43.1 ms (views) to 26.6 ms
(node-local), and the 3.36 million pages of the copies are all on their
node. The bits of the parts are the same with and without the copies.

The one team is already near the parts, because the parallel first touch of
the load spreads its pages over both nodes. The output head runs after the
parts in one team and takes 2 to 3 ms more than after a step of one team.
That takes most of the gain: tg128 is 23 to 25 tokens/s in each form.

On this machine scripts/check_parts.py fails before and after this change:
the hidden state of the parts differs from that of one part by up to 7e-4
(the top token is the same in 12 of 12 steps). It passed on jackal. This
is still open.

### The output head in the parts

Each part now computes its range of the rows of the tied head after the final
norm, from a copy of those rows on its node: Q6K_LINEAR (a new record for
gemma_q6k_linear_body) for a Q6_K head, or KQ_QUANT and KQ_LINEAR for a Q4_0
head with a KQ_Q4X copy. The parts write their rows into one logits buffer.
Its pages are on the node of part 0 (numa.empty_on), so the parts copy their
results to that node when they finish. The main thread reads the logits (the
softcap, the choice of the token), and numa.pin_thread keeps it on that node;
OpenMP had already bound it to CPU 0. Model.logits gives the buffer when x is
the hidden state of the last step of the parts. NP_GEMMA_PART_HEAD=0 turns
this off.

The logits are the same bits as Model.logits on the same hidden state (12 of
12 steps), and the 256 pages of the buffer are on node 0. The median of a
token (forward with the head, then logits: the copy and the softcap):

    form                 forward    logits    sum
    one team             31.5 ms    5.0 ms    36.5 ms
    2 parts with head    29.4 ms    2.3 ms    31.7 ms
    4 parts with head    29.6 ms    2.1 ms    31.7 ms

tg128 (3 runs; clickhouse and sshfs also ran, and runs of the same form
differ by up to 30 per cent): one team 20.5 and 24.0, 2 parts 20.3 and 26.6,
4 parts 28.8 and 25.8 tokens/s.

### The router on each node, and a cache for each part

A measure of the bytes that each part reads on the other node (the arrays of
the records of each part, and the node of their pages) gave, for part 1 at
a context of 1000: the router 43.9 MB for each token (the float32
projection, 2816 x 128 for each of the 30 layers, which each part runs whole),
the cache (304 MB of 347 MB on node 0), and some MB of norms, scratch, and
shared rows. Two changes follow:

- The router: each part reads a copy of the projection and its two scales
  on its node (pk_router).
- PartKVCache (np_gemma/parts.py): a subclass of KVCache with buffers for
  each part. Part p holds the rows of its KV heads (the ranges of the place
  heads) on its node. The buffers of one position of all heads were one
  page (8 heads x 256 x 2 bytes for a sliding layer of the 26B), so a page
  cannot go to two nodes; each part needs its own buffers. The records of
  the parts read their buffer with a row of their heads only (_cache_row).
  With NP_GEMMA_PARTS of 2 or more and the int16 form, KVCache(...) makes a
  PartKVCache (KVCache.__new__); NP_GEMMA_PART_KV=0 turns it off.
- The prompt: Model._attention gives a block to
  PartKVCache.prefill_attention. One record for each part (PART_PREFILL)
  runs in the team of the part: it quantizes the rows of its heads into its
  buffers, makes their float rows, and runs gemma_attn_prefill_region for
  its query heads. The projections, the experts, and the norms of the
  prompt stay in one team.
- The other users of a KVCache (KVCache.read, read_qc, write, write_q: the
  small groups, the GPU) gather or scatter the heads with a copy.
- A layer with fewer KV heads than parts leaves a part with no head in that
  layer. The global layers of the 12B have one KV head, so with 2 parts one
  part runs all of their attention.

The checks (the 26B, a prompt of 2600 tokens, 2 parts): the hidden state of
the prompt and every row of the cache are the same as with one KVCache, and
8 decode steps of the parts give the same hidden states and logits with
either cache. The 152460 pages of the buffers of the parts are all on their
node.

### The profile of the parts (the 12B)

scripts/profile_parts.py gives the time of each operation of a decode step.
In parts, the first thread of each team adds the time of each record
(gemma_run_parts_prof, with no added barrier), and gemma_xbar_stats splits
the barriers into the wait for the team and the wait for the other part. One
team and one node use gemma_profile, which adds a barrier after each record
(about 19 ms for the step of the one team on both nodes: the many small
records each wait for 48 threads).

The 12B (Unsloth UD-Q4_K_XL), a context of 2048, 32 steps, ms a token:

    operation                  part 0   part 1   one node   rate (GB/s)
    rms_norm + gate/up            24.6     24.7       47.9   65 (parts), 67 (node)
    gelu_mul + down               13.6     13.4       24.7   59, 65
    q, k, v projections            5.7      7.9       13.3   63, 65
    o projection                   4.9      4.4        8.4   51-56, 59
    head (KQ_LINEAR)               4.2      4.4          -   65-68
    attention, sliding            16.6     17.1       10.5
    attention, global                -      6.2        7.7
    XBAR (192 a token)            14.1      5.2          -
    of it, wait for the other     12.3      3.7
    of it, wait for the team       1.1      0.9
    sum of the records            86.5     86.7      120.1
    the step (head, Python)       93.9               129.5

The counters (perf stat, uncore) agree: each socket reads 3.6 GB a token,
with almost no remote reads.

- The matrices of each part run at the rate of one node (60 to 68 GB/s for
  a socket). The placement is right, and the bandwidth is not the limit.
- The attention of the sliding layers (ATTN_QC_H, 4 KV heads of the part)
  takes 16.6 ms in each part, more than one node takes for all 8 heads
  (10.5 ms). It reads 4 MB of cache for each layer, 9.6 GB/s. The kernel of
  the heads is not parallel enough for a team of 24 threads.
- The global layers of the 12B have one KV head, so part 1 runs their
  attention (6.2 ms) and their q, k, and v rows (2.3 ms more), and part 0
  waits 12.3 ms at the barriers.
- The wait for the team (1 ms) is small. The rest of the step (7.4 ms) is
  outside the records: the bind, the embedding, the copy of the logits, the
  softcap, and the choice of the token.

The next steps for the 12B: a parallel form of ATTN_QC_H (the keys split
over the threads of the team, as ATTN_QC_MT), and the balance of the global
layers (the query heads split over the parts with the one KV head in both,
or the positions split).

### The attention of the parts, and the global layers of the 12B

- ATTN_QC_H now takes the split form of GP_ATTN_QC (the keys of a head in
  chunks over the threads of the team, gemma_attn_split_i16_rows_body), with
  a row of the cache of row_heads heads. The scratch of the split form is
  one set for each team (as_team: the part + 1 in gemma_run_parts), because
  the parts run it at the same time. The form is the choice of GP_ATTN_QC
  for the whole layer (the operand rep), so each head has its bits: with
  it, scripts/check_parts.py passes on this machine for the 12B and the
  26B (contexts 200 and 1100). It failed before, because the parts used one
  head for each thread and one team used the split form.
- head_split: with a PartKVCache, a layer with fewer key and value heads
  than parts splits the query heads. Each part computes the key and value
  head of its query heads (in its own buffers, head_out) and keeps it in
  its cache. The global layers of the 12B then give 8 query heads to each
  part. PART_PREFILL takes the query heads (h0, h1). The decode with a
  PartKVCache has the bits of the decode with one KVCache.

The 12B, a context of 2048, ms a token in each part:

    operation                 before (0 / 1)    now (0 / 1)
    attention, sliding          16.6 / 17.1      6.6 / 6.4
    attention, global              - / 6.2       6.3 / 6.7
    XBAR, wait for the other    12.3 / 3.7      11.7 / 3.6
    sum of the records          86.5 / 86.7     84.6 / 84.6
    the step                       93.9            92.1

The sliding layers gained 10 ms. The global layers have each 8 query heads
in each part, but the form of one head for each thread takes as long for
8 heads as for 16 (one thread a head, 4 MB of cache for a head at 2048), so
part 0 now spends the time that it waited before. The split form does not
take the global layers of the 12B: 16 query heads of 512 values exceed
AS_MAXQ (4096). In this run part 1 also read its matrices at 54 to 56 GB/s
against 59 to 64 for part 0 (24.7 ms in the run before), so part 0 still
waits.

pp2048 and tg128 at 2048 (tokens/s, 2 reps; the load from a local copy):

    2 parts (PartKVCache)        104.4, 121.3      10.18, 11.70
    one team on node 0           115.6, 121.5       8.24,  8.24
    one team on both nodes       117.2, 138.3      10.16,  8.77

AS_MAXQ is now 8192, so the split form takes the 16 query heads of 512
values of a global layer of the 12B, in one team and in the parts. The AVX2
form of the scores (as_scores_avx2, 8 accumulators) takes the heads of an
item in blocks of 8, with the same values for each head. The values of
those layers change (the split form in place of one head for each thread);
the parts keep the bits of one team: scripts/check_parts.py passes for the
12B and the 26B, and for the 12B with NP_GEMMA_ARCH=avx2.

    operation                 before (0 / 1)    AS_MAXQ 8192 (0 / 1)
    attention, global            6.3 / 6.7         3.0 / 3.0
    XBAR, wait for the other    11.7 / 3.6         4.7 / 9.0
    sum of the records          84.6 / 84.6       79.3 / 79.8
    the step                       92.1              89.7

A download ran during the benchmark of this change (about 30 MB/s to the
disk), and the matrices of part 0 read at 52 to 59 GB/s in this run; all
three forms were slower than in the run before:

    2 parts (PartKVCache)         88.3,  99.7       9.66,  9.71
    one team on node 0           105.2, 110.3       7.36,  7.39
    one team on both nodes       113.9, 129.3       9.00,  8.83

### Locked clocks, and the links between the sockets

With turbo, the clock of each socket moved between 2.6 and 3.5 GHz in a run
(the count of active cores, the AVX-512 license, and the power limit; the
thermal counters showed no throttle). scripts/lock_clocks.sh locks the core
and the uncore clocks (sudo scripts/lock_clocks.sh lock [CORE_MHZ]
[UNCORE_MHZ]; restore puts the settings back).

The links: the UPI counters (uncore_upi_N, event 0x1, the clock of each
link; a clock of 1.3 GHz is 10.4 GT/s) give links 0 and 1 at 10.4 GT/s on
each socket, and link 2 at a clock of 9.6 GT/s with no data (the data flits,
event 0x2, are 0 on both sockets in a run of one team). The machine thus
has 2 links: 2 x 20.8 GB/s raw in each direction, about 28 GB/s of data.

The 12B at a context of 2048 with the core and the uncore at 2.0 GHz (the
clocks stayed at 2000 MHz, the lowest sample 1816):

    form                         pp2048 (tok/s)    tg128 at 2048
    2 parts (PartKVCache)          73.6,  79.7       8.96,  8.23
    one team on node 0             90.9,  97.3       7.79,  7.80
    one team on both nodes         77.9,  92.2       6.33,  6.45

The parts: 98.0 ms a token, records 86.7 / 86.5 ms; the matrices read at
52 to 61 GB/s (37 to 45 for the o projection); the wait for the other part
6.6 ms in part 0 and 11.3 ms in part 1, with the clocks fixed. Each part
waits for the other at some barriers, so the wait is not one slow part:
the time of an operation differs from one part to the other at each
barrier.

### The paired split (NP_GEMMA_PART_PAIRED=1)

The output projection splits by the columns of the query heads of each
part (head_split), and the down map by the columns of the rows of the gate
and the up map of each part. Each part multiplies only the values that it
computed itself, so neither product needs a barrier before it. Each part
writes its sum of the whole output into its row of a shared buffer, and
after one barrier each part adds the rows in the order of the parts (ADD),
so the parts keep the same x. The copies of the columns are KQ_Q4X on the
node of the part (PartCompiler.mat_cols). The place cols marks these
operations. A layer has two barriers in place of four (96 a token for the
12B, not 192). The experts of the 26B keep the rows.

The sums change the order of the additions. Against one team over 16
steps (a context of 1100): the 12B 5.5e-4 of the largest value of the
hidden state, 0.054 in the logits; the 26B 1.7e-5 and 0.010. The top token
is the same in 16 of 16 steps, and 32 tokens of a greedy run are the same,
for both. The default stays the split by rows, with the bits of one team
(scripts/check_parts.py passes).

The 12B at a context of 2048, the clocks at 2.0 GHz:

    form                         records (part 0 / 1)   wait for the other   step
    rows (four barriers)             86.7 / 86.5           6.6 / 11.3        98.0
    paired (two barriers)            84.9 / 84.7           4.8 / 12.2        95.4

    form                         pp2048 (tok/s)    tg128 at 2048
    2 parts, paired                71.7,  82.1       7.72,  8.11
    2 parts, rows                  67.0,  54.7       8.05,  8.46
    one team on node 0             92.9,  97.3       7.81,  7.82

Half the barriers did not shorten the wait. The sum of the waits of the two
parts is 17 ms a token in both forms, so it is not the skew of each
operation; it is one steady difference: part 0 computes for 78.7 ms a token
and part 1 for 71.8 ms (the records less XBAR) on the same work. Node 0 is
about 9 per cent slower in this run: its matrices read at 54 to 57 GB/s,
those of node 1 at 58 to 61. Node 0 holds CPU 0 (the Python thread, most
interrupts) and the other processes of the machine. The next steps: a run
of one team on node 1 against one on node 0, the interrupts and the other
processes away from the cores of the parts, or a split of the rows by the
measured rate of each node.

### The balance: the shares of the rows from a measure

The first steps of a new program of the parts run with the time of each
record (NP_GEMMA_PART_BALANCE, 8 after 2 to warm up). The compiler gives
each record a kind: rows (an operation split by rows, the experts, the
output head, and with the paired split the down map), fixed (the heads, the
norms, the other operations), or xbar. Part p took F_p fixed and R_p rows
at the share s_p; with u_p = R_p / s_p the shares s'_p = (T - F_p) / u_p,
T = (1 + sum F_p / u_p) / sum 1 / u_p, end the parts at the same time
(balance_shares). The wait inside MOE_PART (the barriers less the XBAR
records) leaves R. The program is then compiled with those shares (ranges
with weights, cuts at multiples of 32) and the copies of the old ranges go.
NP_GEMMA_PART_WEIGHTS gives the shares with no measure. The same measure
serves a GPU paired with a CPU: a part reports F and R, and the shares then
differ much more.

scripts/check_parts.py with the shares 0.45 and 0.55 passes for the 12B
and the 26B: a split by rows gives the bits of one part at each cut.

The 12B at a context of 2048, the clocks at 2.0 GHz:

    form      measure (F; R, ms a step)     shares         records     wait      step
    rows      23.2, 23.2; 51.0, 51.5        the same       79.6/79.0   4.7/5.2   91.2
    paired    28.6, 30.4; 42.4, 45.9        0.530/0.470    81.9/81.3   6.4/8.8   90.8

    form                         pp2048 (tok/s)    tg128 at 2048 (the measure in rep 0)
    2 parts, rows                  86.8,  87.6       6.59, 10.49
    2 parts, paired                87.0,  75.6       6.71, 11.22
    one team on node 0             93.4,  96.4       7.59,  7.78

In this run the two nodes were near the same rate (the run before had node 0
9 per cent slower), so the rows kept the same shares. With the balance the
records of the parts are the same, and each part still waits 5 to 9 ms a
token: the wait of each barrier goes to one part or the other, so it is the
spread of each operation, which shares cannot move. The rate of a node
changes from one run to the next (the other processes, the interrupts),
which is a reason for a measure at the start; a measure from time to time
in a long run is a next step.

### The barriers: one team barrier, the prefetch in the wait, the sums

- gp_xbar has one team barrier, not two: after it, the first thread stores
  the count of the part (release), and each thread waits for the flags of
  the other parts itself (acquire). Each thread counts its barriers
  (gp_xcnt, 0 at the start of a run).
- The prefetch (NP_GEMMA_PART_PREFETCH): an XBAR record carries the weights
  of the next operation of the part (_xbar_prefetch: the first copy that the
  records after the barrier read). During the wait, each thread prefetches
  its share of them into L2 (up to 512 KB).
- The paired sums: GP_ADD is an omp for of the team (it was one thread),
  and each part copies its sum into the copy of the sums on the node of each
  other part before the barrier (COPY), so the add reads its node only.

The bits do not change: scripts/check_parts.py passes for the 12B and the
26B, and the paired split gives its values of before (the 12B, 5.53e-4).
The wait for the team stays about 1 ms a token (rows) and 0.4 ms (paired),
so the second team barrier cost little. The prefetch, off and on in turn
(the 12B, a context of 2048, the shares 0.49 and 0.51, 48 steps, the
records of part 0 / 1 in ms a token):

    prefetch off      89.3 / 89.3      93.0 / 92.5
    prefetch on       85.5 / 85.0      90.7 / 88.8

The prefetch takes 2.3 to 4.3 ms a token (3 to 4 per cent). A run against
the run before shows no change, because the rate of the machine moves by as
much from one run to the next; only a measure in turn shows it.

### The prompt pass on the two nodes (scripts/profile_prefill.py)

The 12B, 2048 tokens in blocks of 256, the clocks at 2.0 GHz, the own time
of each function and the traffic of each socket (perf stat, uncore):

    form                     tok/s        q4x_linear   flash     the rest   DRAM S0/S1   remote reads S0/S1
    one team (48 threads)    118.5, 100.4   10.8 s     3.7 s     3.8 s      119/97 GB    260M/252M lines
    one node (24 threads)     97.5,  89.9   13.7 s     4.7 s     2.9 s      195/3 GB     ~0

- The products (_q4x_linear, KQ_Q4X with int8 x) are 53 to 60 per cent of
  the time. On one node they run at about 1.8 T int8 products a second,
  about 15 per cent of VNNI on 24 cores at 2 GHz, with about 9 GB/s of DRAM:
  not the bandwidth. 48 threads make them only 1.07 to 1.27 times faster.
- The KQ_Q4X copy of one team on the nodes of the threads of a static
  schedule (each group of 16 rows on the node of the thread that reads it,
  mbind MPOL_MF_MOVE) made the DRAM of the two sockets the same (112/114
  GB) but not the remote reads (201M/352M lines), and the pass was not
  faster; a step of one team was slower (8.49 and 8.27 tok/s against 9.04
  and 8.90 in turn). So the remote reads are not the weights; they are
  most likely the int8 x of each product, which every thread reads and
  threads of both nodes wrote. The placement was taken out.
- Blocks of 512 (102.1, 112.5 tok/s) and 1024 (85.5, 104.0) did not differ
  from 256 beyond the spread of the runs.
- The prompt of the parts (NP_GEMMA_PARTS=2): PART_PREFILL took 6.4 s, the
  attention of one team (cache.write, cache.read, flash_prefill) about
  4.5 s.

The next steps for the prompt: a copy of the int8 x on each node (each team
quantizes x for itself), a kernel of the products nearer to the rate of
VNNI (the weights of a group unpacked one time for all the tokens of a
block, or a layout for the prompt as the 8x8 Q4_0 of llama.cpp), an
attention that reads the int16 cache with no float copy of it, and the
prompt as a program of the parts (one barrier for each product, no Python
between the operations).

### The prompt pass: the kernel, the attention, the program

The four changes that the measure of the prompt pass gave (the 12B, the
clocks at 2.0 GHz). Each keeps the bits.

1. The products (kq_q4x_gemm in csrc/kquants.c). The counters of one
   thread (perf stat, 3 s in the loop) showed 2.3 instructions a cycle, the
   two vector ports of 512 bits half busy (1.12 a cycle), and the back end
   full (resource_stalls 35 per cent), with few misses: kq_q4x_rows kept the
   float sums of its 16 tokens in an array that the compiler put on the
   stack (36 zmm loads and stores of the stack), and it unpacked the codes
   once for each 16 tokens. The new kernel unpacks the codes of a group of
   16 rows once for a block of tokens (a scratch of the thread in L2), and
   keeps a tile of 2 groups and 6 tokens in registers (12 chains of
   vpdpbusd, 24 sums, no spill). The items are a block of tokens and a pair
   of groups, with blocks of the same size, so a matrix of few rows (the
   down map, 240 groups) still gives each thread the same work. The int32
   sum of a block is exact in any order and the float operations are those
   of kq_q4x_rows, block after block: the same bits
   (scripts/bench_q4x_gemm.py --check against the kernel of one token).

       matrix, 256 tokens       kq_q4x_rows (24 threads)   new (24)    new (48)
       3840 x 15360 (down)          9.49 ms                6.48 ms     5.13 ms
       15360 x 3840 (gate, up)      8.20 ms                5.26 ms     3.06 ms
       3840 x 4096 (o)              2.91 ms                1.78 ms     1.07 ms

   The old kernel gained 1.17 times from the second node; the new one 1.3
   to 1.7 times, with about 44 MB/s of remote reads (the x of a block stays
   in L2).

2. A copy of x on each node: not needed. With the new kernel the remote
   reads of the products are small (above).

3. The attention over the int16 cache (gemma_attn_prefill_qc,
   ops.flash_prefill_qc, and the attention of PART_PREFILL). The team
   dequantizes the keys while it transposes them, and the values of the
   visible rows only, into a scratch of the team, then runs the tasks of
   gemma_attn_prefill. KVCache.read made a float copy of the whole cache
   for each block (new arrays, a page fault for each page), and the
   transpose of the keys ran on the 8 key heads only, or on one thread in
   a team (a nested region). The same bits for each of the 48 layers of the
   12B; a block of 256 at a context of 1100 takes 603 ms in place of 739.

4. A prompt block as one program (np_gemma/prompt.py): the step form of t
   tokens with the kernels of the Python path (KQ_QUANT and KQ_LINEAR on
   the KQ_Q4X copies, RMS_NORM, GELU then MUL, ADD, MUL_S, KV_WRITE,
   QKV_NORM_ROPE, and ATTN_PREFILL_QC for the attention). The buffers of a
   layer are those of the layer before (about 120 MB for 256 tokens, not
   5.8 GB). The profile of the program showed two records that ran on one
   thread, as the Python path did: KV_WRITE (17 per cent of a block) and
   MUL (16 per cent); both now run over the team (and RMS_NORM takes
   gemma_rms_norm_body, MUL_S an omp for), with the same values. With a
   PartKVCache the attention is one PART_PREFILL record for each part (the
   team writes the rows of the heads of the part on its node). The products
   stay in one team (see 1). A dense model only: the experts of the 26B
   take other kernels in a program, so its prompt keeps the Python path
   (with 1 and 3).

   The hidden state of a prompt of 2600 tokens, every row of the cache,
   and the logits of 4 steps after it are the same with the program and
   with the Python path; a PartKVCache gives the rows and the steps of one
   KVCache (2048 tokens, 8 steps), with every page on its node.

The prompt of 2048 tokens of the 12B (tokens/s):

    form                                   before    kernel, attention    program
    one team (48 threads)                  100-118       122-130          167-171
    one node (24 threads)                   90-97        115-120             -
    2 parts (PartKVCache)                   75-88            -              163

llama.cpp on this machine gives 101 to 112 (two nodes) and 63 (one node).

The benchmark after the four changes (pp2048, then tg128 at a depth of
2048, 2 reps, the clocks at 2.0 GHz):

    model   form                       pp2048 (tok/s)    tg128 (tok/s)
    12B     2 parts (PartKVCache)       159.8, 176.2      10.02,  9.40
    12B     one team on node 0          146.8, 150.5       7.89,  7.94
    12B     one team on both nodes      187.1, 163.8       8.12,  7.72
    26B     2 parts                      57.5,  58.1       3.69, 14.07 (rep 0 has the balance)
    26B     one team on node 0           72.7,  74.3      19.51, 19.59

The prompt of the 26B keeps the Python path (its experts: kq_moe, not the
new kernel), and the parts are slower than one node for it, in the prompt
and in the steps. That is the next work on the CPU.

The prompt of the 12B on the GPU (RTX 3090): the products already take
the int8 tensor cores (k_gemm_q8, 84 per cent of the time of the kernels;
NP_GEMMA_GPU_TC=8 changes nothing), and the GPU is busy for 791 of the 792
ms of the pass. A run with a new KVCache for each rep took 1.45 s, not
0.79: _gpu_attach detaches the cache of the rep before (kv.detach copies
its rows from the GPU into that host cache, 607 ms), and no one reads that
cache again. With one KVCache cut to 0 between the reps, as llama-bench
does, the pass gives 2425 to 2478 tok/s against 2718 for llama.cpp. A weak
reference to the attached cache would skip the copy for a cache that the
caller no longer holds.

Done: Model._gpu_cache (and E4B) is a weak reference now. The cache that
was on the GPU before gets its rows only when the caller still holds it (a
Session of the server does); a dropped cache needs no copy. A new KVCache
for each rep then gives 2337 to 2488 tok/s. Model.gpu_sync(cache) (and
E4B.gpu_sync) writes the rows of the GPU into the host cache and keeps the
cache on the GPU, so a server can read or save a cache at any time: the
rows of a held cache after another cache came are those of gpu_sync, and 8
GPU steps with gpu_sync between them give the logits of steps without it.

### The prompt of the 26B as a program, and the int8 activations

np_gemma/prompt.py now takes a model with experts: the router is ROUTER_MT
(the fused router of a step, for each token of the block) and the experts
are KQ_QUANT and KQ_MOE with the GELU (as ops._q4x_moe). The router of the
Python path for a prompt block is a float32 product of NumPy (BLAS) and the
NumPy softmax, which a record cannot give again, so the program has the
bits of the Python path with ops.router_mt as its router: the hidden state
and every row of the cache are the same (2048 tokens). The two routers
select the same experts for every token of every layer of a block (their
weights differ by at most 2e-6), and a call of one does not change the
other (checked).

The 26B, a prompt of 2048 tokens, one team, 2.0 GHz:

    Python path                     32.0 s     64 tok/s
    Python path with router_mt      10.8 s    190 tok/s
    program                         9.3-10.0 s  205-221 tok/s

The router of the Python path cost most of the time: its BLAS product runs
on the threads of OpenBLAS (OPENBLAS_NUM_THREADS is min(8, cores) by
default, np_gemma/__init__.py), which spin next to the threads of OpenMP;
the norms after the router then took 6 to 10 ms in place of 0.2. With
OPENBLAS_NUM_THREADS=1 the Python path takes 2.5 s for 512 tokens, as the
program (2.4 s). The program has no BLAS in the prompt.

The values of the 26B prompt differ from those of the Python path by 18 to
23 per cent of the largest value of the hidden state, and the top token by
12 to 22 per cent, although the routers differ by 2e-6. The cause is the
int8 activations of the prompt (NP_GEMMA_INT4_Q8=1, the Q8_0 form of
llama.cpp): the weights of the router times (1 + 1e-6 noise) move the
hidden state of a block of 256 tokens by these amounts:

    activations    median row    top token the same
    int8           1.15e-1       202/256
    int16          1.6e-3        253/256
    float32        2.2e-4        256/256

A small change moves an int8 value across a rounding step, and the 30
layers of the 26B make it grow (the 12B does not show it: 5.5e-4 in the
check of the paired split). The NLL of the last 255 tokens of a prompt of
1024 (lower is better):

    text          float32   int16    int8     top token as float (int16, int8)
    README        3.167     3.161    3.541    254, 205
    SPLIT_PLAN    4.732     4.714    4.650    252, 183
    parts.py      5.211     5.154    5.347    218, 191

    README, int8 or int16 for the dense products and the experts:
    dense int8,  experts int8     3.541   205/255   6.3 s
    dense int8,  experts int16    3.265   231/255  14.3 s
    dense int16, experts int8     3.310   218/255  10.5 s
    dense int16, experts int16    3.161   254/255  17.7 s

int16 for both gives the NLL of float32; the products of both kinds add to
the noise of int8. The int16 path of the prompt was the Python one (about 3
times the time of int8).

### The int16 prompt of the 26B

The prompt program now takes int16 x (np_gemma/prompt.py, c.q16):

- KQ_QUANT16 (151): the int16 rows of x, a scale of max |x| / 32767 for
  each 32 values (gemma_quant_group32_i16, as the cache).
- KQ_LINEAR16 (152): kq_q4x_gemm16 on the KQ_Q4X copy. A block of 32 codes
  of a group of 16 rows becomes 16 vectors of int16 (code - 8, the pairs of
  vpdpwssd), once for 2 groups of rows; a tile of 2 groups x 4 tokens
  takes them. The columns go in chunks of 128 blocks, so the codes of a
  chunk stay in L2, and the tile goes on from the sums of the chunk before.
- KQ_MOE with act bit 2: the int16 rows of h and of the GELU for each pair
  of token and expert (kq_q4x_rows16, 1 group x up to 8 tokens). A matrix
  that is not KQ_Q4X takes int8 rows, as before.

A product of 256 tokens on 24 threads of node 0 (2.0 GHz), and its error
against float64 (max |d| / max):

    matrix            int16      int8
    15360 x 3840     11.8 ms    5.3 ms     error 2.1e-5 (int8 5.3e-3)
    3840 x 15360     14.9 ms    6.5 ms     (26 ms before the chunks)
    2816 x 8192       5.4 ms
    4224 x 2816       2.7 ms

The NLL of the first 1024 tokens of each text (the 26B, all positions):

    form                         README   SPLIT_PLAN   parts.py   prompt tok/s
    float32 (Python)             4.374    5.117        5.126       38-40
    int16, Python                4.371    5.110        5.099       40-44
    int16, program               4.415    5.127        5.095      141-253
    int8, program                4.442    5.408        5.160      199-319

A prompt of 2048 tokens, 3 runs (one team, 2.0 GHz):

    26B   int16 194, 173, 215 tok/s    int8 202, 200, 241
    26B   int16, 2 parts: 230, 234, 244
    12B   int16 102, 84, 101 tok/s     int8 145, 202, 215

The products of the 26B are a smaller part of its prompt (an expert
takes a few tokens of each block, and the router, the attention and the
sums stay as they were), so int16 costs it about 10 per cent; the 12B
spends most of its prompt in dense products, which take twice the time. ops.prompt_act now gives "16" to a
model with experts when the library has the int16 kernels (VNNI), and "1"
(int8) to the rest; NP_GEMMA_INT4_Q8 sets one form for all. The int16
program and the int16 Python path are not the same bits (the Python path
takes float32 for the dense matrices, and the BLAS router): their hidden
states differ by a median of 4.4e-2 of the largest value, as the routers
choose another expert where two are near.

### Media in the prompt program

A prompt with media (the soft rows of an image or a clip) ran in Python on
the CPU. The program now takes it: prompt_step puts the soft rows in x after
the embedding, and binds the slot "limit", the last key of each query
(Model._media_limit; 0 without media). ATTN_PREFILL_QC and PART_PREFILL
pass it to gemma_attn_prefill_qc_body, whose tasks (AVX-512 and AVX2) take
the limit in place of the position for the causal mask and for the last
key of a tile; the window still counts from the position. The Python path takes the same kernel with the
limit (ops.flash_prefill_qc), in place of the NumPy attention.

Two spans of 280 random rows (bidirectional) in 1100 tokens, one across a
block, at 2.0 GHz:

    12B int8, program against Python             the same bits (one team, 2 parts)
    12B int16 Python, flash with limit against   1e-4 median of max |d| / max
        the NumPy attention (float products)
    12B int16: program 9.3 s, Python 36.9 s
    26B int16: program 4.7 s (2 parts 8.0 s, the same bits), Python 20.9 s

scripts/check_mm_prompt.py on the 12B (--source gguf, against
transformers; mean KL over the top 64, top token, the prompt and the
answer):

    reference   int8 Python        int8 program       int16 Python       int16 program
    image       0.00079 40/40 2.7s  0.00079 40/40 2.1s  0.00005 40/40 11.7s  0.00005 40/40 2.7s
    audio       0.00015 48/48 4.4s  0.00015 48/48 2.7s  0.00001 48/48 14.8s  0.00001 48/48 4.1s
    video       0.00273 42/44 20.4s 0.00273 42/44 13.1s 0.00012 44/44 79.8s  0.00013 44/44 20.0s

int16 takes the KL of the 12B down by 10 to 20 times on media, for the time
of the int8 Python path.

### The int16 prompt on the GPU

The tensor cores of the RTX 3090 take no int16 x. The GPU program of a
prompt with Model.prompt_act "16" (the 26B) gives x the int16 form instead
(gg_load flag 16, g->i8 2): k_quant_x2 makes q = round(x / s) with s =
max |x| / 16256 for each 32 values, and two int8 planes, hi = round(q / 128)
and lo = q - 128 hi. A block of 32 columns takes mma of hi, times 128, then
mma of lo into the same int32 sums (mma16832_x2): 128 sum(hi w) + sum(lo w)
= sum(q w), exact, and below 2^22 for i2f_exact (16256 x 8 x 32). x keeps
about 15 bits. k_gemm_q8 (the dense products) and kt_tile of KT_Q4X
(k_moe_gemm_q4x, the experts) take it with X2: a row of a step holds the
hi values and then the lo values, in 2 buffers of shared memory in place
of 3. The fused int8 x of the record before (qx_fuse) is off in this form.

The 26B on the GPU, the NLL of the first 1024 tokens and a prompt of 4096:

    form                              README   SPLIT_PLAN   parts.py   pp4096
    int16                             4.421    5.167        5.071      2754 tok/s
    int8                              4.358    5.344        5.193      3154
    int16, float32 attention          4.364    5.164        5.070      1437
    int8, float32 attention           4.427    5.188        5.154      1537
    CPU float32                       4.374    5.117        5.126

The hidden state of README against CPU float32 (max |d| / max of a row):

    GPU int16                     median 9.7e-2
    GPU int8                      median 1.1e-1
    GPU int16, float32 attention  median 2.1e-2
    GPU int8, float32 attention   median 1.3e-1
    CPU int16 program             median 4.3e-2
    CPU int8 program              median 1.3e-1

The int16 products are right: with the float32 attention the GPU is nearer
float32 than the CPU program. With the float16 attention of the tensor
cores (NP_GEMMA_GPU_ATTN_TC=1, the default) that attention is most of what
is left, and the float32 kernel takes twice the time; a split of the
query into two float16 values would be the next step. check_gpu_prompt
(11962 tokens, then 32 steps) passes.

### int16 for the 12B?

The 12B, the same texts (NLL of the first 1024 tokens; tok/s of each):

    CPU float32 (Python)       5.506  5.430  6.731     32-35 tok/s
    CPU int16 program          5.508  5.431  6.729    121-140
    CPU int8 program           5.546  5.380  6.697    205-234
    GPU int16                  5.518  5.429  6.731    pp4096 1650
    GPU int8                   5.543  5.407  6.709    pp4096 2340

int16 gives the NLL of float32 on the 12B too, and the KL to transformers
on media falls 10 to 20 times (check_mm_prompt above); int8 moves the NLL
by up to 0.05 either way. It costs the 12B 1.7 times the prompt time on the
CPU and 1.4 times on the GPU, since its prompt is mostly dense products.
The default of the 12B stays int8; NP_GEMMA_INT4_Q8=16 gives int16.

## Phases

- Phase 0: measure. The latency and the rate of a copy over the PCIe link,
  with pinned memory (done, see the measurements). The rate of an int4 GEMV
  on the GPU, against the 448 GB/s of its memory (done). The use of each
  expert of the 26B on several texts, for the hot experts (done). Find a
  machine with two sockets for the NUMA work (open).
- Phase 1: places and parts in the compiler, and a runtime of several parts
  on the CPU. Emulate two nodes on jackal. Test: the same bits as one part,
  for a step of one token (done, see the results of phase 1).
- Phase 2: NUMA for real. Node-local weights, threads that stay on their
  node, and the barrier across the nodes. Measure the decode on a machine with two
  sockets, against one team.
- Phase 3: the GPU backend for the decode step. The kernels of the int4
  GEMV, the Q6_K head, the norms, the rope, the int16 attention, the router,
  and the experts. First run the whole E4B on the GPU, because it fits and
  needs no split. Test it against the CPU and the reference. (The E4B is
  done, see the results of phase 3.)
- Phase 4 (done): the CPU and GPU split of the 26B. First the operation
  split, with the experts on the CPU. Then the hot experts on the GPU. Then
  a comparison with llama.cpp for each split. Its layer split is slower, so
  this runtime does not add one.
- Phase 5 (done): the prompt pass and the MTP group on the GPU. See the
  results of phase 5. The prompt pass is
  limited by the work of the multiply, not by memory, so the GPU gains the
  most there.

## An estimate for the 26B on jackal

This is an estimate, not a measurement:

    part                                      bytes     place   time
    attention, dense part, router, head       1.6 GB    GPU     about 4 ms
    experts                                   0.8 GB    CPU     about 18 ms
    30 round trips of 11 KB                   -         link    about 0.4 ms

A step thus takes about 23 to 25 ms, or about 40 tokens/s, against 17 now.
The attention and the experts do not overlap in one layer, because each
needs the result of the other. The dense feed-forward part can overlap the
experts.

With 1700 hot experts on the GPU (about 4.9 GB), the parts change:

    part                                      bytes     place   time
    attention, dense part, router, head       1.6 GB    GPU     about 4 ms
    hot experts, about 80% of the reads       0.64 GB   GPU     about 1.6 ms
    other experts, about 20% of the reads     0.16 GB   CPU     about 4 to 5 ms
    30 round trips of 11 KB                   -         link    about 0.4 ms

The hot experts on the GPU and the other experts on the CPU run at the same
time. A step then takes about 10 to 12 ms, or about 80 to 100 tokens/s. The
CPU part of a layer is small: about two experts. The fixed cost of each
layer on the CPU (the start of the threads, the barriers) then matters
more. This estimate is less certain than the first one.

## Risks

- The GPU of jackal also drives the display. Its free memory changes with
  the use of Xorg and Chrome. The runtime must check the free memory when it
  loads the model.
- The link is PCIe generation 3 with 8 lanes. A round trip costs 12 to 15
  microseconds, so the operation split is better than the layer split. A
  large copy is slow, so the weights must not move during a step.
- A second set of kernels, for the GPU, doubles the work of each new
  operation. Keep the GPU set small: only the operations of the decode step
  at first.
- The project uses NumPy and C, with no other dependency. CUDA must stay
  optional, and the CPU path must not need it.
- A NUMA test needs a machine with two sockets. jackal can only test the
  correctness.
