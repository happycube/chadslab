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
the CPU, and their output back to the GPU. That is 60 transfers of 11 KB
for each token. The latency of a transfer, not its size, then decides the
cost.

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

## Phases

- Phase 0: measure. The latency and the rate of a copy over the PCIe link,
  with pinned memory. The rate of an int4 GEMV on the GPU, against the 448
  GB/s of its memory. Find a machine with two sockets for the NUMA work.
- Phase 1: places and parts in the compiler, and a runtime of several parts
  on the CPU. Emulate two nodes on jackal. Test: the same bits as one part.
- Phase 2: NUMA for real. Node-local weights, threads that stay on their
  node, and the barrier across the nodes. Measure the decode on a machine with two
  sockets, against one team.
- Phase 3: the GPU backend for the decode step. The kernels of the int4
  GEMV, the Q6_K head, the norms, the rope, the int16 attention, the router,
  and the experts. First run the whole E4B on the GPU, because it fits and
  needs no split. Test it against the CPU and the reference.
- Phase 4: the CPU and GPU split of the 26B. First the operation split, with
  the experts on the CPU. Then the layer split, to compare. Measure the
  tokens/s against the CPU program and against llama.cpp with the same
  split.
- Phase 5: the prompt pass and the MTP group on the GPU. The prompt pass is
  limited by the work of the multiply, not by memory, so the GPU gains the
  most there.

## An estimate for the 26B on jackal

This is an estimate, not a measurement:

    part                                      bytes     place   time
    attention, dense part, router, head       1.6 GB    GPU     about 4 ms
    experts                                   0.8 GB    CPU     about 18 ms
    60 transfers of 11 KB                     -         link    about 1 to 2 ms

A step thus takes about 25 ms, or about 40 tokens/s, against 17 now. The
CPU part and the GPU part do not overlap in one layer, because each needs
the result of the other. Phase 0 measures the latency of a transfer, which
can change this estimate by a large amount.

## Risks

- The GPU of jackal also drives the display. Its free memory changes with
  the use of Xorg and Chrome. The runtime must check the free memory when it
  loads the model.
- The link is PCIe generation 3 with 8 lanes. If a transfer costs more than
  about 30 microseconds, the operation split loses to the layer split. Phase
  0 decides this.
- A second set of kernels, for the GPU, doubles the work of each new
  operation. Keep the GPU set small: only the operations of the decode step
  at first.
- The project uses NumPy and C, with no other dependency. CUDA must stay
  optional, and the CPU path must not need it.
- A NUMA test needs a machine with two sockets. jackal can only test the
  correctness.
