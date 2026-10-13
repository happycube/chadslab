# Plan and record: a multi-level (cache-blocked) GEMM for the prompt pass

## Goal

Make the prompt GEMM use the FMA units of the machine. The earlier kernels
waited on the memory. The multi-level GEMM uses the L3, L2, and L1 caches in
levels, as a standard GEMM does. The decode path is not changed.

## The machine

    item                     value
    cores                    6, one socket, one NUMA node
    L1 data cache            32 KB for each core
    L2 cache                 1 MB for each core
    L3 cache                 8.3 MB, shared
    AVX-512                  FMA, BW, VL. No VNNI.
    plain read               41 to 43 GB/s (six threads)
    pure FMA, one core       210 GFLOP/s (about 1.8 FMA for each clock)
    pure FMA, six threads    602 to 676 GFLOP/s (the machine was loaded)
    convert int8 and add     294 to 379 GFLOP/s (six threads)

Measured with scripts/peak.c. The machine gives only about half of six single
cores when other users load it.

## The design

A GEMM computes C[m, n] = sum over k of A[m, k] * B[k, n]. For this model:

    A   the weights, int8, shape (rows, cols)
    B   the x data, float32, shape (cols, tokens)
    C   the output, float32, shape (tokens, rows)

### The micro kernel

Keep 16 rows and 16 tokens. Vectorize over the rows. One vector holds 16 rows.
The accumulator of each token is one vector. Thus the count is 16 vectors.

For each column k:

    load the 16 weights of the column from the packed A panel (one instruction)
    for each token n in 16:
        broadcast x[k, n] from the packed B panel (one load)
        fused multiply and add into the accumulator of token n

The sum over the columns stays in the vector lanes. Each lane is one output
row. Thus the code needs no horizontal sum. The loads use the load ports and
the fused multiply and add uses the float ports. The two sets of ports work at
the same time. The balance gives near the full speed.

### The panels and the blocks

    value    size    work
    ML_KC    256     the K block
    ML_MC    128     the rows of one A panel
    ML_NC    128     the tokens of one B panel
    ML_MR    16      the rows of one micro tile
    ML_NR    16      the tokens of one micro tile

The A panel is 128 * 256 = 32 KB of int8. The B panel is 256 * 128 * 4 = 128 KB
of float. Both fit in the L2 cache.

### The loop order

One thread owns one row block. It then runs the K blocks in order. The K loop
must stay inside the row loop. The second K block adds to the output of the
first K block. An add must not run at the same time as the first store.

    for each row block (MC), in parallel:
        for each K block (KC):
            pack the A panel
            for each token block (NC):
                pack the B panel
                run the 16 by 16 micro tiles

## Result

The GEMM gives this rate for one matrix:

    shape                 K-vectorized   multi-level   gain
    gate 15360x3840       334.9 GFLOP/s  441.4 GFLOP/s  1.32x
    down 3840x15360       333.5          455.7          1.37x
    o    3840x8192        324.5          456.9          1.41x

The whole model, one process, in turn:

    tokens    multi-level   K-vectorized   gain
    128        8.484 s       10.680 s       1.26x
    256       16.466 s       21.784 s       1.32x
    512       34.252 s       45.053 s       1.32x

The code uses the multi-level GEMM at 128 tokens or more. A shorter prompt
keeps the K-vectorized tile. The AVX2 library keeps the K-vectorized tile,
because the micro kernel needs AVX-512.

## Errors that were found

1. The micro kernel did not multiply by the scales of each row. Every output
   was wrong by that value.
2. The micro kernel used the token offset of the B panel for the load but not
   for the store. Every token tile wrote to the same rows of the output.
3. The second and later K blocks wrote over the first K block. The micro
   kernel needed an add.
4. The loop order put the K block on the outside with collapse(2). Then two
   threads could add and store the same output at the same time. The K loop
   moved inside the row loop.

Error 4 was the last and the most important. Errors 1 to 3 gave a wrong value.
Error 4 gave a value that changed from run to run.

## Tests

    test                                  result
    all model shapes, 32 to 256 tokens    correct
    48 greedy token ids against the ref   equal
    AVX2 library build                    correct

## The AVX2 port

The multi-level GEMM was guarded by __AVX512F__ only. The AVX2 library then
fell back to the token-vectorized tile. This work gave AVX2 its own multi-level
GEMM and K-vectorized tile.

* The micro kernel vectorizes over 8 rows, not 16. ML_MR and ML_NR are 8.
* The AVX2 block sizes are ML_KC 64, ML_MC 64, and ML_NC 64. A KC of 64 makes
  the float32 B panel 16 KB, so the B panel stays in the 32 KB L1 cache. A
  larger KC was about 10 percent slower.
* The AVX2 K-vectorized tile uses a tile of 4 rows and 2 tokens. The 8
  accumulators, 4 weight vectors, and 2 x vectors fit in the 16 registers.

The result at 256 tokens was:

    shape                 before    after
    int8 15360x3840        120       289 GFLOP/s
    int8 3840x3840         124       335 GFLOP/s
    int4 15360x3840         69       308 GFLOP/s
    int4 3840x3840          69       337 GFLOP/s

The int4 mode also got a prompt GEMM. That kernel decodes one row block to a
float32 A panel and reuses the panel for every token block. The int4 decode
kernel now does the sign decode in the byte lanes. That change gave 2.6 times
more speed on a wide matrix.

## What is left

The AVX-512 multi-level GEMM reached 428 to 490 GFLOP/s against the measured
FMA value of 602 to 676 GFLOP/s. The AVX2 value is 289 to 337 GFLOP/s. The
panel pack and the broadcast of each x value control the rest of the time. The
next changes are a wider micro tile and a packed B panel.

## What the plan did not change

* The decode path. It uses the 4-row int8 GEMV kernel and the 4-row int4 dot.
* The weight cache.
* The attention and the norms.
