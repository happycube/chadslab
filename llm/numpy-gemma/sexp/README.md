# The programs of Qwen3.6 as S-expressions

This directory has the programs that run Qwen3.6-35B-A3B in this runtime
(the GGUF file, UD-Q4_K_M). They are written as S-expressions for reading.
scripts/export_sexp.py writes them:

    python scripts/export_sexp.py --out sexp

The runtime does not read these files. It runs the programs as arrays of
records (np_gemma/program.py): the CPU interpreter is gemma_run in
csrc/bf16_linear.c, and the GPU interpreter is csrc/gpu.cu. Each record has
an operation and up to 24 operands. An operand is a literal, a float, or a
slot of the environment. A literal can be the address of an array. The
exporter writes the name of the array in place of the address.

## The files

    file                    records  the program
    cpu_step.sexp           791      one token on the CPU (QwenGGUFProgram)
    gpu_step.sexp           641      one token on the GPU; the cold experts
                                     go to the CPU (QwenGPU)
    cpu_cold_experts.sexp   2        the CPU program that the GPU step runs
                                     for the cold experts of layer 0
    gpu_verify4.sexp        about 780  an MTP verify group of 4 tokens on the
                                     GPU
    gpu_prompt1024.sexp     701      a group of 1024 rows of a prompt on the
                                     GPU; the experts are copied to the GPU

The files come from one run with 0.3 GB of hot experts (a small set, so
the names are the same for each run). The numbers of the buffers (%12)
change from one run to the next.

## The notation

    (program NAME (env $slot ...) RECORD ...)

- A record is (op operand ...). The name of the op is the name of the
  code in program.py, in lower case with "-" (KQ_LINEAR is kq-linear).
- A comment ";; ---- layer N ----" starts the records of layer N. The
  exporter finds the layer from the first weight of a record. Thus the two
  fetch records at the start of the prompt program show as layers 0 and 1.

The operands:

    $pos, $cos, $sin        a slot of the environment; the bind writes it
                            before each run
    $K.3, $kq.3, $S.0       a slot with the address of a cache array: keys
                            (int16 kq, ks, vq, vs) or the state of a
                            linear layer (S, conv)
    $t17                    a slot of a scalar operation (s-add, s-mul)
    w:blk.0.attn_qkv        a weight in the memory map of the GGUF file
    gpu:blk.2.attn_qkv      the copy of a weight that the GPU holds (Q8_0 as
                            type 100, in rows for 16-byte loads)
    gpu:blk.2.hot_gate      the hot experts of layer 2 (the slots)
    gpu:blk.2.slots         the slot of each expert of layer 2, or -1
    gpu:blk.0.table_up      the device address of each expert for a large
                            group (a hot slot, or the copy buffer)
    %12:f32[1x2048]         buffer 12 of the program: its type and shape
    %4:f32[4x4096]+16384    an address inside buffer 4 (a byte offset)
    x, xn                   the input and the output of the program
    #<cpu-program 3>        a CPU program (run by cpu-join)
    8, 2048, 0.0625f        a literal: an int, or a float32
    0                       a null address, or a zero

The type of a matrix is its ggml number: 0 F32, 8 Q8_0, 12 Q4_K, 13 Q5_K,
14 Q6_K. 100 is the GPU form of Q8_0.

## The operations

The operands follow the order of gp_step in C. t is the count of tokens
(rows) of the record.

The products:

    (kq-quant x t cols xq xs xm)
        Quantize t rows of x to int8: xq, a scale xs for each 32 values, and
        xm (xs times the sum of each 16 values). The CPU products read xq.
        The GPU programs have no kq-quant: their products read x.
    (kq-linear xq xs xm x w type rows cols t out)
        out (t x rows) = x W^T. The CPU reads xq; the GPU reads x.
    (kq-multi x cols t n w1 type1 rows1 out1 ... wn typen rowsn outn)
        The GPU only: up to 5 kq-linear records on the same x in one kernel.

The small operations:

    (rms-norm x w out t cols eps)
    (add a b out n)                 out = a + b
    (add-rms x o w h t cols eps)    the GPU only: x += o, then h = rms-norm(x)
    (sigmul x g out n)              out = x sigmoid(g)

The layers:

    (gdn qkv $conv conv_w kernel z a b A_log dt_bias norm_w $S out scratch
         t k_heads v_heads k_dim v_dim eps log tiled $nreal)
        The Gated DeltaNet: the convolution, then the recurrence of each
        value head, then the gated norm. With a log (a verify group) the
        state does not change. tiled 1 is the order of the value heads of
        the GGUF file. $nreal is the count of the real rows of a group.
    (attn-prep qg k v q_norm k_norm $cos $sin K V hs $pos t nq nk hd rot eps
               scale qout gate kout)
        The query and its gate from qg, the norms, and RoPE on the first rot
        values. With the int16 cache (K and V are 0), the key goes to kout.
    (s-mul $t a b) (s-add $t a b)
        Scalar operations on the host: they make the address of the row
        $pos in the cache ($kq.3 + $pos * 1024 bytes).
    (kv-write k v 0 0 kq ks vq vs n)
        Quantize the rows of the key and the value to int16 (a scale for
        each 32 values) and store them in the cache.
    (attn-qc q $kq $ks $vq $vs $scores out nq nk hd n)
        The attention of one query over the first n rows of the int16
        cache. A step and each query of a small group use it.
    (attn-qc-mt q $kq $ks $vq $vs $scores out nq nk hd t $pos 0 0 lo n)
        The attention of a large group (t queries from $pos).

The experts:

    (router-topk logits t experts k val idx)
        The softmax of the router, the best k, and their weights over their
        sum.
    (kq-moe xq xs xm ids val t k experts mats shared_logit hidden inner
            scratch out [count])
        The experts on the CPU: the pairs (token, expert) sorted by expert,
        gate and up, silu, down, and the weighted sum. An id of -1 is no
        expert. count (in memory) gives k for one token (the cold experts).
    (hot-split idx val slots cold cold_val k 1)
        The GPU step: the selected experts that the GPU does not hold go to
        cold, with their count. With 1, cold also gets the selection, for
        HotCache.
    (hot-split-mt idx val slots cold cold_val pairs $nreal k)
        The same for a group: each pair gets its expert if it is cold, else
        -1.
    (kq-hot-moe h val idx slots gate up down act act2 de out k inner hidden
                gtype dtype t sgate sup sdown stype slog)
        The hot experts and the shared expert on the GPU.
    (kq-group-moe h val idx t k experts hidden inner tgate tup tdown gtype
                  dtype sgate sup sdown stype slog work act act2 de out $nreal)
        All the experts of a large group on the GPU, with the tables of the
        device addresses of the experts.

The moves between the GPU and the CPU (the boundaries of the CUDA graphs):

    (to-host dev1 host1 bytes1 dev2 host2 bytes2 dev3 host3 bytes3 event)
        Copy up to 3 arrays to pinned host memory, and record an event.
    (cpu-join #<cpu-program N> event)
        Wait for the event, then run the CPU program.
    (to-dev host dev bytes)
        Copy the output of the CPU program to the GPU.
    (fetch ranges count layer buffer) (fetch-wait layer) (fetch-done buffer)
        The copies of the experts of a layer to one of two GPU buffers, by a
        worker thread. fetch-wait makes the stream wait for the copies of a
        layer; fetch-done frees the buffer for the copies of layer + 2.

## A layer of each program

A linear layer of the CPU step (cpu_step.sexp, layer 0):

    (rms-norm x w:blk.0.attn_norm %0 1 2048 1e-06f)
    (kq-quant %0 1 2048 %1 %2 %3)
    (kq-linear ... w:blk.0.attn_qkv 8 8192 2048 1 %4)      ; q, k, v
    (kq-linear ... w:blk.0.attn_gate 8 4096 2048 1 %5)     ; z
    (kq-linear ... w:blk.0.ssm_beta 0 32 2048 1 %6)        ; b (F32)
    (kq-linear ... w:blk.0.ssm_alpha 0 32 2048 1 %7)       ; a (F32)
    (gdn %4 $conv.0 w:blk.0.ssm_conv1d 4 %5 %7 %6 ... $S.0 %8 ...)
    (kq-quant %8 1 4096 ...)
    (kq-linear ... w:blk.0.ssm_out 8 2048 4096 1 %10)
    (add x %10 x 2048)
    (rms-norm x w:blk.0.post_attention_norm %0 1 2048 1e-06f)
    (kq-linear ... w:blk.0.ffn_gate_inp 0 256 2048 1 %11)  ; the router
    (kq-linear ... w:blk.0.ffn_gate_inp_shexp 0 1 2048 1 %12)
    (router-topk %11 1 256 8 %13 %14)
    (kq-moe ... %14 %13 1 8 256 %15 %12 2048 512 %16 %17)
    (add x %17 x 2048)

The GPU step (gpu_step.sexp) has the same layers, with these changes:

- The products on the same x are one kq-multi, and each add and the next
  norm are one add-rms. That removes about 150 records.
- The experts split into two parts:
  - the record hot-split finds the cold experts, and to-host copies the
    input and the cold experts to the host;
  - kq-hot-moe runs on the GPU while cpu-join runs the CPU program of the
    cold experts;
  - to-dev and add then sum the two parts.

The CPU program of the cold experts (cpu_cold_experts.sexp) has two
records: kq-quant of the pinned input, and kq-moe of the cold list. Its
last operand, %19+32, is the count of cold experts, which the GPU wrote.

A full-attention layer (layer 3) with the int16 cache:

    (attn-prep %4 %5 %6 w:blk.3.attn_q_norm w:blk.3.attn_k_norm $cos $sin
               0 0 0 $pos 1 16 2 256 64 1e-06f 0.0625f %35 %8 %36)
    (s-mul $t16 $pos 1024) (s-add $t17 $kq.3 $t16) ...   ; the row of $pos
    (kv-write %36 %6 0 0 $t17 $t19 $t21 $t23 512)
    (s-add $t24 $pos 1)
    (attn-qc %35 $kq.3 $ks.3 $vq.3 $vs.3 $scores %7 16 2 256 $t24)
    (sigmul %7 %8 %7 4096)                                ; the output gate

The MTP verify group (gpu_verify4.sexp) runs 4 tokens. Its attention is
one attn-qc for each token (n = $pos + 1, ..., $pos + 4), the kernel of the
step. Its gdn records have a log (the state does not change until commit).
Thus a verify group gives the same bits as 4 steps.

The prompt group (gpu_prompt1024.sexp) starts the copies of the experts of
layers 0 and 1 before layer 0. In each layer: fetch-wait, kq-group-moe,
fetch-done, and fetch of layer + 2. The products of 1024 rows use a tiled
kernel, and the attention is attn-qc-mt.
