# Programs: the opcodes, how they run, and what uses them

This document describes the program system of np_gemma. A program is a
list of records, and each record has an opcode and up to 24 operands. The
models compile their steps, prompt groups, heads, and media encoders into
programs. Three interpreters run the records:

- the C interpreter on the CPU (np_gemma/csrc/bf16_linear.c, gemma_run);
- the CUDA runner on the GPU (np_gemma/csrc/gpu.cu, gg_run);
- a NumPy interpreter for checks (np_gemma/program.py, Program.run_py).

Section 1 gives the format. Section 2 tells how programs are made. Section
3 tells how each interpreter runs them. Section 4 is the table of all the
opcodes. Section 5 lists the builders, section 6 the weight formats, and
section 7 the switches and the hardware. Section 8 lists the open points.

The line numbers are those of the tree when this was written.

## 1. The format

### 1.1 The buffer

Program.finish (program.py:214) makes one int64 array, Program.buf:

    int64  magic ("GPROG001", 0x4750524F47303031), env count, record count, 0
    int64  env[env count]          the slots: parameters and scalar variables
    record code[record count]      224 bytes (28 int64 words) each

A record (REC in program.py:133, gp_rec in bf16_linear.c:7691 and
gpu.cu:60) has these fields:

    int32 op; int32 flags; uint8 tag[24]; int64 v[24]

- NARG is 24. It must agree in program.py, bf16_linear.c, and gpu.cu.
  finish() checks the record size against gemma_gp_record_size().
- flags is not used.
- Each operand has a tag: T_NONE (0, not used), T_INT (1, a literal), T_F32
  (2, float32 bits), or T_SLOT (3, an index into env).
- An address is a T_INT literal. None is (T_INT, 0), a null address.
- An ndarray operand gives its address. Program.keep holds the array, so
  the address stays good for the life of the program.

### 1.2 Slots and scalar operations

A slot is a named value in env. bind() writes it before a run. An array
gives its address, a float its bits, and an int its value.
Examples are pos, scores, base.N, cos.N, k.N, seq, and nreal.

Opcodes 1 to 6 (S_MOV, S_ADD, S_SUB, S_MUL, S_MAX, S_MIN) compute int64
values into slots: dst = a op b. The compiler emits them for positions and
cache addresses that change at each run, for example base + row * stride.
It folds constants in Python. Only S_ADD, S_SUB, S_MUL, and S_MAX occur;
S_MOV and S_MIN are never emitted.

- On the CPU each thread has a private copy of env. All the threads compute
  each scalar record, so a scalar record needs no barrier.
- On the GPU the host computes the scalar records (gg_scalars) before it
  copies env to the device.

### 1.3 The Program class (program.py:158)

| Method | What it does |
|---|---|
| slot(name, value) | Returns the slot of a name, or makes it |
| temp() | Makes a slot for a scalar result |
| emit(op, *args) | Appends one record; the operand order is that of gp_step in C |
| finish() | Makes buf; env and code are views of it |
| bind(**kw) | Writes slots |
| run(limit) | Runs in C on the CPU; limit runs only the first records |
| profile() | Runs in C with a barrier after each record; the time of each record in ms |
| profile_ops(reps) | The time of each opcode name |
| dump(limit) | The records as text |
| run_py(limit) | Runs in the NumPy interpreter (section 3.4) |

prog.names holds the buffers of the compiler (x, xn, tok, _scratch, ...).
gpu.GPUProgram runs the same buf on the GPU (section 3.2).

## 2. How programs are made

### 2.1 The compiler

Compiler (program.py:657) compiles forms (Lisp lists) into records:

| Form | Result |
|---|---|
| (seq f ...), (layer i f ...) | Compiles each form in order |
| (let name e), (let (a b) e) | Gives a name to a result |
| (set name e) | Compiles the kernel e into the buffer name |
| (slot name) | A slot |
| (w layer key), (m module), (t key) | A weight of the model |
| (+ ...), (- ...), (* ...), (max ...) | Scalar records |
| any other head | A kernel of KERNELS (program.py:1412) |

A kernel function (k_rms_norm, k_int4_multi4, k_attn_qc, k_moe, k_enc_linear,
...) chooses the opcodes. For example, int4_multi4 gives INT4_MULTI4 for one
token and INT4_MULTI4_MT for a group. For a GGUF matrix it gives KQ_QUANT and
KQ_LINEAR instead.

The forms of the Gemma 4 12B and 26B are layer_form and step_form. Those of
the E2B and E4B are e4b_layer_form and e4b_step_form. format_form prints a
form. The subclasses of Compiler change the kernels:

| Class | File | Change |
|---|---|---|
| _EncCompiler | gemma4_encoders.py:103 | No model; packed x16 weights for the CPU; Q8_0 linears |
| PartCompiler | parts.py:97 | One NUMA part: XBAR, MOE_PART, ATTN_QC_H, ATTN_F32_H |
| PoolCompiler | gpu.py:448 | Buffers that the layers share |
| E4BGroupCompiler | gpu.py:490 | One ATTN_F32H for each query of a small group |
| SplitCompiler | gpu.py:857 | The 26B on the GPU, experts on the CPU: the handoff records |

### 2.2 Programs made by hand

The Qwen builders emit records directly (prog.emit), with no forms:
compile_qwen_step (qwen.py) and compile_qwen4_step (qwen4.py). The heads, the
MTP drafter, and the CPU sub-programs of the GPU builders are also made by
hand.

### 2.3 Rewrites

Two passes change the records of GPU programs after compilation:

- gpu._fuse_kq (gpu.py:516) drops KQ_QUANT. It joins 2 to 5 KQ_LINEAR
  records with the same x (16 rows or less) into one KQ_MULTI.
- qwen_gpu._fuse (qwen_gpu.py:160) does the same. It also joins ADD and an
  RMS_NORM in place into ADD_RMS.

KQ_MULTI and ADD_RMS come only from these passes.

## 3. How the programs run

### 3.1 On the CPU (the C interpreter)

The library is np_gemma/csrc/bf16_linear.c. It includes moe.c,
mlx_affine.c, kquants.c, deltanet.c, hyperconn.c, and qsa.c.

- Program.run calls cops.gp_run, which calls gemma_run (bf16_linear.c:9017).
  gemma_run checks the magic and opens one OpenMP parallel region. In it,
  gp_exec gives each thread a private copy of env and runs gp_step on each
  record (bf16_linear.c:8139).
- One region for the whole program replaced about 420 regions for each step
  (PERF_PLAN.md, phase 2).
- A kernel body divides its loop with an orphaned "omp for". The implicit
  barrier at its end separates it from the next record.
- A small operation on one row (RMS_NORM, ADD, MUL_S, COPY, MUL,
  GELU_MUL_ROWS, KV_WRITE, KV_WRITE_HEADS) runs in "omp single".
- Scratch memory comes from "omp single copyprivate".
- The expert kernels (kq_moe_body, ma_moe_body) use schedule(runtime):
  dynamic for groups and static for one token. QSA uses dynamic,1.
- gemma_profile runs the same records with a barrier after each one and
  gives the time of each record.
- gemma_run_parts runs one program for each NUMA part in nested regions
  (SPLIT_PLAN.md). XBAR is the barrier between the parts.

An opcode with no case in gp_step goes to "default: break". It does
nothing and gives no error. The GPU records (84 to 98, 110 to 113, 124 to
132) are such opcodes on the CPU.

The build (cops.py:92) compiles one library with -O3 -fopenmp and one of
three flag sets:

| Library | Flags | Selection |
|---|---|---|
| AVX2 | -mavx2 -mfma -mf16c | NP_GEMMA_ARCH=avx2, or a CPU with no AVX-512 |
| AVX-512 | -mavx512f -mavx512bw -mavx512vl | NP_GEMMA_ARCH=avx512, or AVX-512 with no VNNI |
| VNNI | the AVX-512 flags and -mavx512vnni | NP_GEMMA_ARCH=vnni, or a CPU with VNNI |

The file is np_gemma/_libs/libgemma_<hash>.so. The hash covers the sources,
the platform, the compiler, and the flags. Most kernels select the ISA when
they compile (__AVX512F__, __AVX512VNNI__, __AVX2__). The bfloat16, int8, and
Q6_K kernels also select it at run time (gemma_have_avx512). The code has no
ARM or NEON kernels. The build always gives x86 flags, so on other CPUs the
models use NumPy.

### 3.2 On the GPU (the CUDA runner)

gpu.py builds np_gemma/csrc/gpu.cu with nvcc -O3 -arch=native into
np_gemma/_libs/libgemma_gpu_<hash>.so. GPUProgram(prog, graph, mirror, tc,
i8, atc) (gpu.py:298) loads a program:

1. Mirror (gpu.py:204) makes a device copy of each host array of
   prog.keep, the first time a record uses it. One mirror serves all the
   programs of a model, so the programs share the weights.
2. The record operands that are host addresses become device addresses,
   in a copy of buf. The CPU program stays as it was. SKIP (gpu.py:61)
   lists the operands that must stay host addresses: the int4 float32
   scales, pinned host buffers, and CPU sub-programs.
3. gg_load (gpu.cu:9616) copies the records and env to the device. It makes
   the attention scratch (part, 8.4 MB), the float16 scratch (xh), and the
   int8 scratch (kqx) that the records need.
4. gg_load cuts the records into segments. A boundary record (TO_HOST,
   CPU_JOIN, TO_DEV, FETCH, FETCH_WAIT, FETCH_DONE, CPU_START, CPU_WAIT)
   ends a segment. A segment also ends after 32 records (the first) or 160
   records (the others), so that the GPU starts early.

The flags of gg_load: 1 = CUDA graphs, 2 = no tensor cores, 4 = int8
activations for the int4 products, 8 = the attention on the tensor cores
also with flag 2.

A run (gg_run, gpu.cu:9771):

1. The host copies env, computes the scalar records, and copies env to the
   device (asynchronously).
2. gg_exec records a CUDA graph for each segment at the first run, or when
   gg_prepare (gpu.cu:9755) runs. It records all the segments before it
   launches one, because a lazy module load during a capture can wait for a
   CPU task.
3. It launches the graphs in order. At a boundary it first runs the CPU
   tasks of the records before it (section 3.3), then the boundary action.
4. gg_run does not wait for the GPU. download() waits.

Programmatic dependent launch (PDL): on compute capability 9 or more, each
kernel starts with griddepcontrol.wait, and the graph edges between kernels
become programmatic edges. A kernel then starts while the kernel before it
ends. NP_GEMMA_GPU_PDL=0 turns this off.

A kernel gets its operands in one of two ways:

- A: the kernel gets the device record and env (DP, DI, di). Slot values
  are read at each run, so one graph serves every position and every
  cache. The launch sizes (grid, block) must be literals: hlit() refuses a
  slot.
- B: the host resolves the operands when it launches (hi, hlit, hlitf). In
  a graph these values are fixed when the graph is recorded. The gemm
  kernels (k_gemm*, k_mt_gemv*), all the k_enc_* kernels, MOE_GPU, and
  D2H/H2D use this way. Their operands must be literals, which the
  compilers give.

A record that the GPU cannot run fails with "no GPU kernel for operation
N". A size that is a slot fails with "operation N: a size is not a
literal, or the form is not supported". A shape that the kernel does not
take gives the same error.

### 3.3 CPU work inside a GPU program

Some records hand work to the CPU. The work is a normal CPU program (a
Program.buf). gg_cpu_run runs it; gpu.py sets gg_cpu_run to gemma_run, the
C interpreter. The handoff has three forms:

- Events (NP_GEMMA_GPU_FLAGS=0): TO_HOST copies to pinned memory and
  records an event. CPU_JOIN waits for the event and runs the CPU program on
  the runner thread. TO_DEV copies the result back. The GPU kernels between
  TO_HOST and CPU_JOIN run at the same time as the CPU program.
- Flags in one graph (the default): SIGNAL is a kernel that copies to
  pinned memory and writes a flag. CPU_TASK has no kernel. After the
  launches, the runner waits for the flag, runs the CPU program, and writes
  its own flag. AWAIT is a kernel that waits for that flag and copies the
  result to the device; a count of 0 gives zeros with no wait. The flags
  take the value of the slot seq, which grows at each run.
- A helper thread: CPU_START waits for an event and gives a CPU program to
  a helper thread. CPU_WAIT waits for it. The runner goes on with GPU work
  and FETCH copies in the meantime.

The mixed groups of the Qwen prompts (qwen_gpu.py:597) use the third form
twice. First a one-record CPU program, MOE_PLAN (moe.c:97), chooses which
experts the GPU copies and which the CPU computes. Then FETCH copies the
chosen experts on a worker thread while a CPU program (KQ_QUANT, KQ_MOE)
computes the others. KQ_GROUP_MOE runs the hot, copied, and shared experts
on the GPU.

The timeouts: AWAIT stops the GPU (__trap) after 60 s, and the context is
then lost. A CPU_TASK gives an error after 30 s.

### 3.4 In NumPy (run_py)

Program.run_py (program.py:298) runs the records one at a time in Python
(_py_step, program.py:324). It calls the C function of each kernel, or
NumPy. scripts/check_program.py uses it to find the first record where the
two differ. It covers 47 opcodes: the scalar records, the
Gemma 4 and E4B records, ADD_NORM, COUNT, and the 15 ENC_* records. It does
not run the Qwen, MLX, K-quant, QSA, or GPU handoff records. By default the
Gemma programs have KQ records too (the KQ_Q4X copies). Thus run_py is a
reference only with NP_GEMMA_Q4X=0.

## 4. The opcodes

The columns CPU, GPU, and Py tell which interpreter has the opcode. "sub"
means that the opcode runs in a CPU sub-program of a GPU builder. The
builders (B1 to B18) are in section 5.

### 4.1 Scalar records (1 to 6)

| # | Name | Meaning | CPU | GPU | Py | Used by |
|---|---|---|---|---|---|---|
| 1 | S_MOV | dst = a | yes | host | yes | none |
| 2 | S_ADD | dst = a + b | yes | host | yes | B1 B2 B5 B7 B8 B10-B14 |
| 3 | S_SUB | dst = a - b | yes | host | yes | B1 B2 B7 B8 |
| 4 | S_MUL | dst = a * b | yes | host | yes | B1 B2 B7 B8 B10-B14 |
| 5 | S_MAX | dst = max(a, b) | yes | host | yes | B1 B2 B7 B8 |
| 6 | S_MIN | dst = min(a, b) | yes | host | yes | none |

### 4.2 Rows and values (16 to 21)

| # | Name | Meaning | CPU | GPU | Py | Used by |
|---|---|---|---|---|---|---|
| 16 | RMS_NORM | RMS norm of each row (w may be null) | yes | yes | yes | B1-B5 B7-B14 |
| 17 | ADD | out = a + b | yes | yes | yes | B1-B5 B7-B12 B14 |
| 18 | MUL_S | out = x times a float | yes | yes | yes | B1-B5 B7-B9 |
| 19 | COPY | copy bytes | yes | yes | yes | B1 B2 B7 B8 B9 B14 |
| 20 | GELU | tanh GELU | yes | yes | yes | B3 B4 B5 B9 |
| 21 | MUL | out = a b, a row stride for b | yes | yes | yes | B3 B4 B5 B9 |

### 4.3 Matrices (32 to 39)

The int4 format is the Gemma 4 QAT format: blocks of 32 values with a
float32 scale array. A null scale pointer means a KQ_Q4X copy (section 6).

| # | Name | Meaning | CPU | GPU | Py | Used by |
|---|---|---|---|---|---|---|
| 32 | INT4_LINEAR | int4 matrix, one row of x | yes | yes | yes | B1 B2 B3 B4 B7 B9 |
| 33 | INT4_MULTI4 | up to 4 int4 matrices on one x | yes | yes | yes | B1 B2 B3 B4 B7 |
| 34 | RMS_NORM_MULTI4 | the norm, then INT4_MULTI4 | yes | yes | yes | B1 B2 B7 |
| 35 | GELU_MUL_INT4 | gelu(g) u, then an int4 matrix | yes | yes | yes | B1 B2 B7 |
| 36 | INT4_LINEAR_MT | int4 matrix, t rows of x | yes | yes | yes | B1 B3 B5 B8 |
| 37 | INT4_MULTI4_MT | up to 4 int4 matrices, t rows | yes | yes | yes | B1 B3 B5 B8 |
| 38 | GELU_MUL_ROWS | gelu(g) u, row by row | yes | yes | yes | B1 B4 B5 B8 |
| 39 | BF16_LINEAR | bfloat16 matrix | yes | yes | yes | B3 B4 B5 |

On the GPU, INT4_LINEAR_MT and INT4_MULTI4_MT use k_mt_gemv for 16 rows or
less. Larger groups use the tensor cores (k_gemm_tc2) or int8 (k_gemm_q8).
The CPU uses one kernel for all sizes. A group decodes each weight once for
4 tokens (AVX-512) or 2 (AVX2).

### 4.4 Attention (48 to 57)

| # | Name | Meaning | CPU | GPU | Py | Used by |
|---|---|---|---|---|---|---|
| 48 | QKV_NORM_ROPE | norms of q, k, v, then RoPE | yes | yes | yes | B1 B2 B7 B8 |
| 49 | KV_WRITE | write K and V rows; also the int16 copy | yes | yes | yes | B1 B2 B7 B8 B10-B14 |
| 50 | ATTN_QC | one query over the int16 cache | yes | yes | yes | B1 B2 B7 B8 B9 B10-B12 |
| 51 | ATTN_F32 | one query over the float cache (window) | yes | yes | yes | B1 B2 B7 |
| 52 | ATTN_QC_MT | a group of queries, int16 cache; lim for images | yes | yes | yes | B1 B8 B10-B12 |
| 53 | ATTN_F32_MT | a group of queries, float cache | yes | no | yes | B1 (NP_GEMMA_ATTN=0) |
| 54 | QKV_NORM | norms of q, k, v | yes | yes | yes | B3 B4 B5 |
| 55 | ROPE | RoPE of q and k | yes | yes | yes | B3 B4 B5 B9 |
| 56 | KV_WRITE_HEADS | write the E4B cache (heads, positions, hd) | yes | yes | yes | B3 B4 B5 |
| 57 | ATTN_F32H | queries over the E4B cache, with a window | yes | yes | yes | B3 B4 B5 B9 B10-B12 |

On the GPU, ATTN_QC_MT with more than 16 queries runs flash attention on the
tensor cores (k_flash_qc_tc). Operand 16 (lim) gives the last key of each
query, so the tokens of an image can see each other (the 12B and the 26B).

### 4.5 Experts of Gemma 4 (64 to 68)

| # | Name | Meaning | CPU | GPU | Py | Used by |
|---|---|---|---|---|---|---|
| 64 | ROUTER | the router of one token: top-k weights | yes | yes | yes | B1 B2 B7 |
| 65 | MOE | the int4 experts of one token | yes | no | yes | B1, B7 (sub) |
| 66 | ROUTER_MT | the router of t tokens | yes | yes | yes | B1 B8 |
| 67 | MOE_MT | the int4 experts of a group | yes | no | yes | B1, B8 (sub) |
| 68 | MOE_N | MOE with the count in memory (the GPU writes it) | yes | no | no | B7 (sub) |

MOE, MOE_MT, and MOE_N occur only with NP_GEMMA_Q4X=0. By default the
experts are KQ_Q4X copies and use KQ_MOE.

### 4.6 Parts (80 to 83; NP_GEMMA_PARTS of 2 or more)

| # | Name | Meaning | CPU | GPU | Py | Used by |
|---|---|---|---|---|---|---|
| 80 | XBAR | the barrier between the NUMA parts | yes | no | no | B2 |
| 81 | MOE_PART | the rows of the experts of one part | yes | no | no | B2 |
| 82 | ATTN_QC_H | ATTN_QC for some of the heads | yes | no | no | B2 |
| 83 | ATTN_F32_H | ATTN_F32 for some of the heads | yes | no | no | B2 |

### 4.7 The GPU and the CPU together (84 to 98)

| # | Name | Meaning | CPU | GPU | Py | Used by |
|---|---|---|---|---|---|---|
| 84 | TO_HOST | up to 3 copies to pinned memory, then an event (boundary) | no | yes | no | B7 B8 B12 B14 |
| 85 | CPU_JOIN | wait for the event, run a CPU program (boundary) | no | yes | no | B7 B8 B12 B14 |
| 86 | TO_DEV | a copy from the host (boundary) | no | yes | no | B7 B8 B12 B14 |
| 87 | HOT_SPLIT | the selected experts that are not on the GPU | no | yes | no | B7 B12 B14 |
| 88 | HOT_MOE | the hot experts of the 26B | no | yes | no | B7 B8 |
| 89 | MOE_GPU | all the experts of a large group of the 26B | no | yes | no | B8 |
| 90 | FETCH | a worker thread copies expert weights (boundary) | no | yes | no | B8 B12 B14 |
| 91 | FETCH_WAIT | the stream waits for a FETCH (boundary) | no | yes | no | B8 B12 B14 |
| 92 | FETCH_DONE | a copy buffer is free again (boundary) | no | yes | no | B8 B12 B14 |
| 93 | HOT_SPLIT_MT | HOT_SPLIT for each pair of a group | no | yes | no | B8 B12 B14 |
| 94 | F32_LINEAR | float32 matrix, one row | no | yes | no | B9 |
| 95 | DRAFT_HEAD | the centroid head of the drafter | no | yes | no | B9 |
| 96 | ARGMAX | the index of the largest value | no | yes | no | B6 B9 |
| 97 | ADD_NORM | out = (x + norm(o) w) scale, and a second norm | no | yes | yes | B4 B5 B7 |
| 98 | COUNT | counts of the selected experts (HotCache) | no | yes | yes | B8 |

### 4.8 MLX, the Gated DeltaNet, and Qwen (100 to 109)

| # | Name | Meaning | CPU | GPU | Py | Used by |
|---|---|---|---|---|---|---|
| 100 | MA_QUANT | int8 rows of x for the MLX products | yes | no | no | B10 |
| 101 | MA_LINEAR | MLX affine matrix (4 or 8 bits) | yes | no | no | B10 |
| 102 | MA_MOE | MLX affine experts | yes | no | no | B10 |
| 103 | ROUTER_TOPK | softmax, top-k, and scale (Qwen) | yes | yes | no | B10-B14 |
| 104 | GDN | the Gated DeltaNet layer (conv, recurrence, gated norm) | yes | yes | no | B10-B14 |
| 105 | ATTN_PREP | q and k norms, partial RoPE, gate, cache write | yes | yes | no | B10-B14 |
| 106 | SIGMUL | out = x sigmoid(g) | yes | yes | no | B10-B14 |
| 107 | KQ_QUANT | int8 rows of x for the K-quant products | yes | no-op | no | B1 B3 B7 B8 B11-B14 B16-B18 |
| 108 | KQ_LINEAR | GGUF matrix (section 6) | yes | yes | no | B3-B6 B11-B18 |
| 109 | KQ_MOE | GGUF or KQ_Q4X experts, and the shared expert | yes | no | no | B1, B7 B8 B12 B14 (sub), B11 B13 |

The GPU products quantize x themselves, so KQ_QUANT does nothing there, and
the rewrites drop it.

### 4.9 GPU records of Qwen (110 to 113)

| # | Name | Meaning | CPU | GPU | Py | Used by |
|---|---|---|---|---|---|---|
| 110 | KQ_HOT_MOE | the hot experts and the shared expert | no | yes | no | B12 B14 |
| 111 | KQ_MULTI | up to 5 KQ_LINEAR on one x (from the rewrites) | no | yes | no | B4 B5 B12 B14 |
| 112 | ADD_RMS | x += o; h = norm(x) w (from the rewrites) | no | yes | no | B12 |
| 113 | KQ_GROUP_MOE | the experts of a large group on the GPU | no | yes | no | B12 B14 |

### 4.10 Qwen3.8 (114 to 123)

| # | Name | Meaning | CPU | GPU | Py | Used by |
|---|---|---|---|---|---|---|
| 114 | HC_NORM | the grouped norm of the residual streams | yes | yes | no | B13 B14 |
| 115 | HC_ACT | silu(x scale) | yes | yes | no | B13 B14 |
| 116 | HC_MIX | the mean of the streams with gates | yes | yes | no | B13 B14 |
| 117 | HC_ADD | add a block output into each stream | yes | yes | no | B13 B14 |
| 118 | PLE_GATE | the gate of the n-gram rows | yes | yes | no | B13 B14 |
| 119 | PLE_CONV | the dilated conv of the n-gram layer | yes | yes | no | B13 B14 |
| 120 | QSA_SELECT | the key blocks of each query (the indexer) | yes | yes | no | B13 B14 |
| 121 | ATTN_QSA | attention over the selected keys | yes | yes | no | B13 B14 |
| 122 | HC_CAT | the input of the MTP layer | yes | yes | no | B13 B14 (MTP) |
| 123 | MOE_PLAN | the plan of a mixed group (moe.c) | yes | no | no | B12 B14 (sub) |

QSA_SELECT has 24 operands. Operand 22 (qpos) gives the M-RoPE positions of
the rows after an image, and operand 23 the sections. Its scratch has nbmax
keys of 8 bytes for each query; nbmax follows the rows of the run.

### 4.11 CPU work inside a graph (124 to 130)

| # | Name | Meaning | CPU | GPU | Py | Used by |
|---|---|---|---|---|---|---|
| 124 | CPU_START | give a CPU program to the helper thread (boundary) | no | yes | no | B8 B12 B14 |
| 125 | CPU_WAIT | wait for the helper thread (boundary) | no | yes | no | B8 B12 B14 |
| 126 | SIGNAL | a kernel: copies to pinned memory, then a flag | no | yes | no | B7 B12 B14 |
| 127 | AWAIT | a kernel: waits for a flag, then copies to the device | no | yes | no | B7 B12 B14 |
| 128 | D2H | a copy node, device to host | no | yes | no | none |
| 129 | H2D | a copy node, host to device | no | yes | no | none |
| 130 | CPU_TASK | the runner runs a CPU program between two flags | no | yes | no | B7 B12 B14 |

### 4.12 Two GPU records of Gemma 4 (131, 132)

| # | Name | Meaning | CPU | GPU | Py | Used by |
|---|---|---|---|---|---|---|
| 131 | FFN_OUT | the end of a 26B layer in one kernel | no | yes | no | B7 |
| 132 | SOFTCAP | x = tanh(x / cap) cap, the logits | no | yes | no | B6 |

### 4.13 The media encoders (133 to 147)

All three interpreters run these records. On the CPU the rows are divided
over the threads; on the GPU the kernels are k_enc_*.

| # | Name | Meaning | Used by |
|---|---|---|---|
| 133 | ENC_LINEAR | clamps, x W^T + b; W bfloat16 or float32; packed x16 on the CPU | B16 B17 B18 |
| 134 | ENC_RMS | RMS norm of each row | B16 B17 |
| 135 | ENC_GELU_MUL | gelu_tanh(g) u | B16 |
| 136 | ENC_ADD | x += s y | B16 B17 B18 |
| 137 | ENC_ROPE2D | the axial 2D RoPE | B16 B18 |
| 138 | ENC_ATTN | attention of all the rows, scale 1 | B16 B18 |
| 139 | ENC_SILU | x sigmoid(x) | B17 |
| 140 | ENC_MUL_VEC | each row times a vector | B17 B18 |
| 141 | ENC_GLU | a sigmoid(b) of the halves of each row | B17 |
| 142 | ENC_DWCONV | the causal depthwise conv | B17 |
| 143 | ENC_LOCAL_ATTN | the local attention of gemma4a | B17 |
| 144 | ENC_CLAMP | clamp x | B16 B17 (Q8_0) |
| 145 | ENC_BIAS_CLAMP | y = clamp(y + b) | B16 B17 B18 (Q8_0) |
| 146 | ENC_LNORM | LayerNorm with bias | B18 |
| 147 | ENC_GELU | gelu, tanh or erf | B18 |

The limits: ENC_ATTN on the GPU takes a head of at most 128 values.
ENC_LOCAL_ATTN takes a span of at most 32 on the GPU and 64 on the CPU. The
tensor-core ENC_LINEAR needs bfloat16 W and k % 8 == 0.

## 5. The builders

| Code | Model | Builder | Where it runs | Role |
|---|---|---|---|---|
| B1 | Gemma 4 12B, 26B | program.compile_step (program.py:1602) | CPU | decode, MTP verify |
| B2 | Gemma 4 12B, 26B | parts.compile_parts (parts.py:489) | CPU, NUMA parts | decode |
| B3 | E2B, E4B | program.compile_e4b_step (program.py:1804) | CPU | decode, MTP verify |
| B4 | E2B, E4B | E4BGPU (gpu.py:571) | GPU | decode |
| B5 | E2B, E4B | gpu.compile_e4b_group (gpu.py:549) | GPU | prompt, MTP verify |
| B6 | E2B, E4B | E4BGPU._kq_head (gpu.py:757) | GPU | head of a K-quant model |
| B7 | 26B, 12B | gpu.compile_split_step (gpu.py:1460) | GPU and CPU | decode |
| B8 | 26B, 12B | gpu.compile_split_group (gpu.py:1419) | GPU and CPU | prompt, MTP verify |
| B9 | drafters of the E4B and the 26B | gpu.compile_drafter (gpu.py:2347) | GPU | MTP drafts |
| B10 | Qwen3.6 (MLX files) | qwen.compile_qwen_step with QwenCPU | CPU | decode, prompt, verify |
| B11 | Qwen3.6 (GGUF) | qwen.compile_qwen_step with QwenGGUFCPU | CPU | decode, prompt, verify |
| B12 | Qwen3.6 (GGUF) | QwenGPU._compile, _compile_mix (qwen_gpu.py) | GPU and CPU | decode, verify, prompt |
| B13 | Qwen3.8 | qwen4.compile_qwen4_step with Qwen4CPU | CPU | decode, prompt, verify, MTP |
| B14 | Qwen3.8 | Qwen4GPU._compile, _compile_mix, _mtp_group | GPU and CPU | decode, verify, prompt, MTP |
| B15 | Qwen3.8 | Qwen4GPU._head (qwen4_gpu.py:421) | GPU | head (KQ_LINEAR) |
| B16 | gemma4v (E2B, E4B, 26B) | Gemma4Vision.program | CPU or GPU | image encoder |
| B17 | gemma4a (E2B, E4B) | Gemma4Audio.program | CPU or GPU | audio encoder |
| B18 | Qwen3.6, Qwen3.8 | QwenVision.program (vision_qwen.py) | CPU or GPU | image and video encoder |

These files build no programs: model.py, e4b.py, assistant.py (the CPU
drafter), unified.py (the 12B embedder, NumPy), mlx_affine.py, and
st_qwen4.py. The Q6_K heads of the models are not programs: gg_q6k_head
runs them.

The scripts build no records by hand. scripts/export_sexp.py writes the
Qwen3.6 programs as text (sexp/). scripts/check_program.py compares C with
run_py.

## 6. The weight formats

| Format | Type | Layout | Opcodes |
|---|---|---|---|
| Gemma int4 | - | blocks of 32: 16 bytes of codes (offset 8), float32 scales apart | INT4_*, MOE, MOE_MT, MOE_N, MOE_PART |
| bfloat16 | - | rows | BF16_LINEAR, ENC_LINEAR |
| float32 | - | rows | ENC_LINEAR, F32_LINEAR |
| MLX affine | - | groups of 64; 4 or 8 bits; bfloat16 scale and bias | MA_* |
| KQ_F32 | 0 | rows | KQ_LINEAR |
| Q5_1, Q8_0 | 7, 8 | blocks of 32 (GGUF) | KQ_LINEAR, KQ_MOE |
| Q4_K, Q5_K, Q6_K | 12, 13, 14 | blocks of 256 (GGUF) | KQ_LINEAR, KQ_MOE |
| IQ4_NL | 20 | blocks of 32, a table of 16 values | the n-gram table only |
| BF16 | 30 | rows | KQ_LINEAR, KQ_MOE |
| NV4 | 51 | NVFP4 by row: E2M1 codes, E4M3 scales | KQ_LINEAR, KQ_MOE |
| NVX | 53 | NVFP4 in groups of 16 rows | KQ_LINEAR, KQ_MOE |
| KQ_Q4X | 54 | Q4_0 in groups of 16 rows (made from the Gemma int4) | KQ_LINEAR, KQ_MOE, INT4_* (null scales) |
| KQ_Q8X16 | 60 | Q8_0 in groups of 16 rows | KQ_LINEAR (x86 only) |
| KQ_BF16X16 | 61 | bfloat16 in groups of 16 rows | KQ_LINEAR, ENC_LINEAR (x16) |
| KQ_F32X16 | 62 | float32 in groups of 16 rows | KQ_LINEAR, ENC_LINEAR (x16) |

The kernels on the CPU:

- With VNNI, every K-quant type has an int8 kernel (vpdpbusd), in tiles of
  4 rows and 4 tokens for groups.
- With AVX2 or AVX-512 and no VNNI, only 54, 60, 61, and 62 have vector
  kernels. The other K-quant rows are dequantized in plain C.
- The GPU has its own Q8_0 row format (Q8_R, 100) for the Qwen products.
- The KV cache is float32, or int16 with a float32 scale for each 32 values.

## 7. Switches and hardware

### 7.1 Switches that change the records

| Variable | Default | Effect |
|---|---|---|
| NP_GEMMA_PROGRAM | 1 | 0: the Gemma and E4B CPU steps run in Python, with no programs |
| NP_GEMMA_ATTN | 1 | 0: the float attention records (ATTN_F32*) in place of ATTN_QC* |
| NP_GEMMA_PARTS | 1 | 2 or more: B2 (NUMA parts) for one token |
| NP_GEMMA_Q4X | 1 | 0: MOE, MOE_MT, MOE_N in place of KQ_MOE with KQ_Q4X copies |
| NP_GEMMA_GPU | 0 | 1: the Gemma models use the GPU builders |
| NP_GEMMA_GPU_KV | int16 | float: ATTN_F32 in B7 |
| NP_GEMMA_GPU_FLAGS | 1 | 0: TO_HOST, CPU_JOIN, TO_DEV in place of SIGNAL, CPU_TASK, AWAIT |
| NP_GEMMA_GPU_FUSED | 1 | 0: no ADD_NORM or FFN_OUT in B7 |
| NP_GEMMA_GPU_MIX | 1 (26B), 2048 (Qwen) | the mixed groups (CPU_START, MOE_PLAN); 0 turns them off |
| NP_GEMMA_GPU_MIX_MIN | 256 | the shortest part of a Qwen prompt for a mixed group |
| NP_GEMMA_GPU_FETCH_MIN | 700 | the fetch groups of Qwen (FETCH, KQ_GROUP_MOE) |
| NP_GEMMA_GPU_HOT_DYN | 1 | COUNT records after the routers of B8 |
| NP_GEMMA_QWEN_KV | int16 | f32: ATTN_F32H in place of KV_WRITE and ATTN_QC* in B10 to B12 |
| NP_GEMMA_ENC_PY | 0 | 1: the encoders run in NumPy, with no programs |
| NP_GEMMA_ENC_X16 | 1 | 0: no packed x16 weights in ENC_LINEAR on the CPU |
| --mmproj-q8 (serve.py) | auto | Q8_0 encoder linears: ENC_CLAMP, KQ_QUANT, KQ_LINEAR, ENC_BIAS_CLAMP |

Other switches change how the records run, not which records:

- NP_GEMMA_ARCH: the CPU library.
- NP_GEMMA_GPU_TC, _TC_MOE, _I8_MOE, _ATTN_TC: the flags of gg_load.
- NP_GEMMA_GPU_PDL, _ENC_TC, _KQTC, _I8X, _GLU, _FLASH: GPU kernel choices.

### 7.2 The hardware

The CPU:

- x86-64 with AVX2, FMA, and F16C at least. AVX-512 (F, BW, VL) and VNNI
  give the fast paths.
- The CPU library has no ARM build.
- The target of the AVX2 work is an i5-8500 (6 cores). The development CPU
  has VNNI.

The GPU:

- The library builds with -arch=native for the GPU of the machine. The key
  of the file does not include the architecture.
- The kernels need compute capability 8.0 or more (mma.sync, cp.async,
  __nanosleep). PDL needs 9.0 or more.
- The code uses the CUDA 13 forms of cudaGraphGetEdges and of the graph
  dependency calls. This machine has CUDA 13.1 and an RTX 5060 Ti
  (compute capability 12.0).
- Some kernels need more than 48 KB of shared memory (up to about 99 KB).
- The fast kernels take these shapes:
  - flash attention: a head of 256 or 512;
  - tensor-core products: cols % 64 (k_gemm_tc2), cols % 128 (k_gemm_q8,
    k_kq_tc);
  - the Qwen GDN: key and value heads of 128.

## 8. Open points

- The C interpreter ignores an unknown opcode with no error. A GPU record in
  a CPU program does nothing. A check in finish() or gp_step can catch it.
- S_MOV, S_MIN, D2H, and H2D are never emitted.
- In a graph, the operands that the host resolves (way B) are fixed when
  the graph is recorded. A slot there keeps its value of that time. The
  compilers give literals, but no check stops a slot.
- An AWAIT that waits 60 s stops the GPU context; the process must start
  again.
- The runner is not reentrant: one stream (gg_stream) and static state for
  all the programs.
- gg_load does not free its buffers when it fails.
- The float16 and int8 scratch buffers (xh, kqx) and the named buffers of
  bind() belong to each program. The QSA scratch of Qwen3.8 is the only
  shared one (QwenGPU._alloc). Shared buffers for all the sizes of a model
  can give room for more hot experts.
