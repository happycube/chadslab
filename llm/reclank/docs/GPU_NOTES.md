# How the GPU path became fast

These notes give the method and the changes that made the GPU path of this
runtime fast, in the order of their gain. SPLIT_PLAN.md has the full data of
each test. The GPU is an RTX 5060 Ti (16 GB) on a PCIe 3.0 x8 link (6.9
GB/s pinned). The CPU is a Xeon W-2295 (18 cores, 67 GB/s).

## Results

    model and task                  first GPU version   now          llama.cpp
    E4B, prompt of 1024 tokens      1527 tok/s          4492 tok/s   5178 tok/s
    E4B, decode                     84 tok/s            94.5 tok/s   112 tok/s
    E4B, decode with MTP            124 tok/s           128 tok/s    -
    26B, decode (1.5 GB hot)        48 tok/s            58-61 tok/s  43 tok/s
    26B, decode with MTP            39 tok/s            65-68 tok/s  -

The values of the 26B depend on the text (HotCache). llama.cpp puts the 26B
on the GPU in parts of layers. Its experts stay on the CPU.

## The method

1. Measure first, then change. Each step started with a profile. Nsight
   Systems gave the kernels and the idle time of the GPU. gg_profile gave
   the time of each record, and cProfile gave the host. Nsight Compute did
   not run (RmProfilingAdminOnly), so the tests below took its place.
2. Compare with a reference. A profile of llama.cpp on the same model and
   the same machine showed which parts were slow, not only which parts were
   large.
3. Remove a part to find its cost. A kernel ran without its copies, its
   math, or its scales (the result was wrong, only the time counted). This
   showed that the int8 products waited for the copies of the weights, not
   for the math.
4. Simulate first, then build. The expert selections of 6 real answers
   went to a file.
   A simulation of 12 cache policies on that file chose LFU with decay
   before any GPU code changed. The live result agreed with the simulation
   (3.37 against 3.32 cold experts in each layer).
5. Keep the values the same. Each speed change had a test that the result
   did not change. Some tests looked for the same bits (the fused kernels,
   the new attention). Some looked for the same tokens (MTP, HotCache). The
   tensor cores had a test of the share of top tokens against float32. A change that changes the values (the reuse of the
   experts in MTP) stays a test, with numbers for its error.
6. Look at the whole timeline, not only the kernels. In the decode of the
   E4B, the kernels took the same time as in llama.cpp. The difference was
   in the gaps between them and on the host.

## The changes, with their gain

The prompt pass of the E4B (1024 tokens, int8 products):

- The rows of the embeddings in one C call, not one NumPy call for each
  token: 343 ms to 7 ms. The GPU waited for the host.
- Attention: one block for the 4 query heads that share a key head. The
  copies use cp.async and two buffers, and the rows in shared memory have a
  pad. 88 ms to 16 ms, the same bits.
- The projection of the layer input on the tensor cores (float16 from the
  bfloat16 rows in registers): 11 ms to 1.8 ms.
- Fused small operations (add_norm, gelu_mul) with no fused multiply-add,
  so the values stay the same: 39 ms to 22 ms.
- The quantization of x one time for each group of matrices (q, k, v;
  gate, up). Each thread takes 4 values. 15.6 ms to 6.8 ms.

The decode of the E4B:

- Fewer kernels: the fused forms in the decode step, and add_norm2 (the
  next norm in the same kernel). About 210 fewer kernels in each step.
- Programmatic dependent launch: a kernel starts before the one before it
  ends. It helps only when the step has few gaps of other kinds.
- The head for one row, the logits in pinned memory, and an argmax in C
  with AVX2 (0.21 ms to 0.04 ms).

The decode of the 26B:

- The router, the attention, and the dense part on the GPU; the experts on
  the CPU. The experts that the GPU holds run there (the hot experts).
- HotCache: the hot experts follow the text. Scores with decay, at most 8
  copies in each step by a worker thread, and the slot table changes only
  when the GPU is idle. 1.5 GB of hot experts then hold about 60% of the
  selections, not 20%. The rate went up by 19% to 41%.
- The host part of HotCache runs while the GPU runs the head.
- MTP verify groups count their routers (GP_COUNT), so HotCache also
  learns during MTP: 41 to 54 tok/s from a new cache.

## What did not help

- A prefetch of the next matrix to L2 before the wait of PDL.
- More buffers or a second block on each SM for the int8 products.
- Pinned host memory for the experts: the link is slow (6.9 GB/s), and the
  copies of HotCache were not the cost.
- Copies in pieces of 32 KB or 128 KB.
- Several graphs for one step: Nsight Systems showed a shorter launch, but
  the trace itself made the launch slow.
- A seed of HotCache from the prompt pass: a small gain for a short prompt,
  a loss for a long prompt.
- A cold expert that goes to the GPU only from its second use: 1% to 7%
  fewer copies, the same rate.

## Lessons for other work

- The host part of a step is often the cost: Python work, a copy in a
  loop, a sync. Measure it on its own.
- A kernel with few bytes of math is a latency problem: fewer, larger
  operations help more than a faster kernel.
- A product with a quantized matrix waits for its data more often than for
  its math. The layout and the size of the copies matter.
- The data of the real task gives a better plan than a general rule. The
  experts of real answers were a better guide than the most used experts
  of other texts.
