"""The operation codes of the programs: the one table for Python, the CPU
runner (backends/cpu/csrc/common/interp.c, gp_step), and the GPU runner (backends/cuda/csrc,
gg_launch).

Each row is (name, code, runners, text). runners says where a record of the
code runs: "cpu", "gpu", or both. The C name of a code is GP_<name>.

    python -m reclank.ir.opcodes          # writes reclank/backends/include/opcodes.h
    python -m reclank.ir.opcodes --check  # exit 1 when the header differs

The header has the codes, the record (gp_rec), and gp_op_known(). A runner
gives an error for a code that the table does not have. A known code that a
runner does not run (a GPU record on the CPU) does nothing there, as before
the table; tests/unit/test_opcodes.py checks that each code has a case in
the runners that the table names. PROGRAMS.md describes the records.
"""
from __future__ import annotations

import os
import sys

# The size of a record and the first word of a program.
NARG = 24
MAGIC = 0x4750524F47303031          # "GPROG001"

# The tag of an operand: an integer (also an address), the bits of a float32,
# or the index of a slot of the environment.
T_NONE, T_INT, T_F32, T_SLOT = 0, 1, 2, 3

B, C, G = ("cpu", "gpu"), ("cpu",), ("gpu",)

TABLE = [
    # 1 to 15: scalar operations on the slots of the environment
    ("S_MOV", 1, B, "a slot gets a value"),
    ("S_ADD", 2, B, "the sum of two values"),
    ("S_SUB", 3, B, "the difference of two values"),
    ("S_MUL", 4, B, "the product of two values"),
    ("S_MAX", 5, B, "the larger of two values"),
    ("S_MIN", 6, B, "the smaller of two values"),
    # element operations
    ("RMS_NORM", 16, B, "the RMS norm of rows"),
    ("ADD", 17, B, "a + b"),
    ("MUL_S", 18, B, "x times a scalar"),
    ("COPY", 19, B, "a copy of rows"),
    ("GELU", 20, B, "the tanh GELU"),
    ("MUL", 21, B, "a * b"),
    # int4 matrices (the Gemma 4 QAT files) and bfloat16
    ("INT4_LINEAR", 32, B, "an int4 matrix on one token"),
    ("INT4_MULTI4", 33, B, "up to 4 int4 matrices on the same x"),
    ("RMS_NORM_MULTI4", 34, B, "the norm of x, then up to 4 int4 matrices"),
    ("GELU_MUL_INT4", 35, B, "gelu(gate) * up, then an int4 matrix"),
    ("INT4_LINEAR_MT", 36, B, "an int4 matrix on a small group"),
    ("INT4_MULTI4_MT", 37, B, "up to 4 int4 matrices on a small group"),
    ("GELU_MUL_ROWS", 38, B, "gelu(gate) * up of rows"),
    ("BF16_LINEAR", 39, B, "a bfloat16 matrix"),
    # attention and the cache
    ("QKV_NORM_ROPE", 48, B, "the norms of q, k, v and the rope"),
    ("KV_WRITE", 49, B, "the int16 rows of k and v in the cache"),
    ("ATTN_QC", 50, B, "the attention of one query over the int16 cache"),
    ("ATTN_F32", 51, B, "the attention over the float cache"),
    ("ATTN_QC_MT", 52, B, "ATTN_QC for a small group"),
    ("ATTN_F32_MT", 53, C, "ATTN_F32 for a small group"),
    ("QKV_NORM", 54, B, "the norms of q, k, v"),
    ("ROPE", 55, B, "the rope of q and k"),
    ("KV_WRITE_HEADS", 56, B, "the float rows of k and v (the E4B)"),
    ("ATTN_F32H", 57, B, "the float attention of the E4B"),
    # the int8 cache (KVCache kv="int8"): KV_WRITE, ATTN_QC, ATTN_QC_MT with int8 values
    ("KV_WRITE8", 58, B, "KV_WRITE with int8 values"),
    ("ATTN_Q8", 59, B, "ATTN_QC over int8 values"),
    ("ATTN_Q8_MT", 60, B, "ATTN_QC_MT over int8 values"),
    # int16 keys and int8 values (KVCache kv="k16v8")
    ("KV_WRITEV8", 61, B, "int16 keys, int8 values"),
    ("ATTN_V8", 62, B, "the attention over int16 keys and int8 values"),
    ("ATTN_V8_MT", 63, B, "ATTN_V8 for a small group"),
    # the experts of the 26B
    ("ROUTER", 64, B, "the router of one token"),
    ("MOE", 65, C, "the experts of one token"),
    ("ROUTER_MT", 66, B, "the router of a small group"),
    ("MOE_MT", 67, C, "the experts of a small group"),
    ("MOE_N", 68, C, "the experts of a large group"),
    # the TQ6 cache (reclank/formats/tq6.py)
    ("KV_WRITETQ", 69, B, "KV_WRITE8 with the TQ6 rotation"),
    ("ATTN_TQ", 70, B, "ATTN_Q8 over the TQ6 cache"),
    ("ATTN_TQ_MT", 71, B, "ATTN_Q8_MT over the TQ6 cache"),
    ("TQ_ROT", 72, B, "the rotation of the queries (and its inverse for the output)"),
    # a program in parts (reclank/models/gemma4/parts.py)
    ("XBAR", 80, C, "the barrier between the parts"),
    ("MOE_PART", 81, C, "the experts of one part"),
    ("ATTN_QC_H", 82, C, "ATTN_QC for the heads of one part"),
    ("ATTN_F32_H", 83, C, "ATTN_F32 for the heads of one part"),
    # work between the GPU and the CPU (reclank/models/gemma4/gpu.py)
    ("TO_HOST", 84, G, "rows to pinned host memory"),
    ("CPU_JOIN", 85, G, "wait for the CPU program"),
    ("TO_DEV", 86, G, "rows from pinned host memory"),
    ("HOT_SPLIT", 87, G, "the experts of a token: hot on the GPU, cold on the CPU"),
    ("HOT_MOE", 88, G, "the hot experts of one token"),
    ("MOE_GPU", 89, G, "experts on the GPU"),
    ("FETCH", 90, G, "start the copy of experts to the GPU"),
    ("FETCH_WAIT", 91, G, "wait for a copy of experts"),
    ("FETCH_DONE", 92, G, "the end of the copies of a step"),
    ("HOT_SPLIT_MT", 93, G, "HOT_SPLIT for a small group"),
    ("F32_LINEAR", 94, G, "a float32 matrix"),
    ("DRAFT_HEAD", 95, G, "the head of the MTP drafter"),
    ("ARGMAX", 96, G, "the best token of rows of logits"),
    ("ADD_NORM", 97, G, "x + o, then the norm"),
    ("COUNT", 98, G, "count the uses of experts"),
    # 100 to 102 were the MLX affine records (removed)
    # the Gated DeltaNet and Qwen3.5 (QWEN_PLAN.md)
    ("ROUTER_TOPK", 103, B, "the top k experts of the Qwen router"),
    ("GDN", 104, B, "the Gated DeltaNet"),
    ("ATTN_PREP", 105, B, "the norms and the rope of the Qwen attention"),
    ("SIGMUL", 106, B, "sigmoid(gate) * x"),
    ("KQ_QUANT", 107, B, "int8 rows of x for the GGUF products"),
    ("KQ_LINEAR", 108, B, "a GGUF matrix (K quants, Q8_0, NVFP4, BF12)"),
    ("KQ_MOE", 109, C, "GGUF experts"),
    # the GPU only (reclank/models/qwen/gpu35.py)
    ("KQ_HOT_MOE", 110, G, "the hot GGUF experts"),
    ("KQ_MULTI", 111, G, "up to 5 GGUF matrices on the same x"),
    ("ADD_RMS", 112, G, "x + o, then the RMS norm"),
    ("KQ_GROUP_MOE", 113, G, "GGUF experts of a large group"),
    # qwen4exp: the gated residual and the n-gram layer (backends/cpu/csrc/common/hyperconn.c)
    ("HC_NORM", 114, B, "the norm of the streams"),
    ("HC_ACT", 115, B, "the gates of the streams"),
    ("HC_MIX", 116, B, "the mix of the streams"),
    ("HC_ADD", 117, B, "the output back into the streams"),
    ("PLE_GATE", 118, B, "the gate of the n-gram layer"),
    ("PLE_CONV", 119, B, "the convolution of the n-gram layer"),
    # qwen4exp: QSA (backends/cpu/csrc/common/qsa.c)
    ("QSA_SELECT", 120, B, "the blocks of keys of each query"),
    ("ATTN_QSA", 121, B, "the attention over the selected blocks"),
    ("HC_CAT", 122, B, "the streams side by side"),
    ("MOE_PLAN", 123, C, "the plan of the experts of a group (CPU part)"),
    ("CPU_START", 124, G, "start a CPU program"),
    ("CPU_WAIT", 125, G, "wait for a CPU program"),
    # the handoff to the CPU inside a graph: flags in pinned memory (backends/cuda/csrc)
    ("SIGNAL", 126, G, "set a flag for the CPU"),
    ("AWAIT", 127, G, "wait for a flag of the CPU"),
    ("D2H", 128, G, "a copy to the host"),
    ("H2D", 129, G, "a copy to the device"),
    ("CPU_TASK", 130, G, "a CPU program inside the graph"),
    ("FFN_OUT", 131, G, "the end of a layer of the 26B in one kernel"),
    ("SOFTCAP", 132, G, "x = tanh(x / cap) cap, the soft cap of the logits"),
    # the media encoders (reclank/encoders/runtime.py)
    ("ENC_LINEAR", 133, B, "a matrix of an encoder"),
    ("ENC_RMS", 134, B, "the RMS norm"),
    ("ENC_GELU_MUL", 135, B, "gelu(gate) * up"),
    ("ENC_ADD", 136, B, "x + s y"),
    ("ENC_ROPE2D", 137, B, "the 2D rope of the vision encoders"),
    ("ENC_ATTN", 138, B, "the attention of an encoder"),
    ("ENC_SILU", 139, B, "the SiLU"),
    ("ENC_MUL_VEC", 140, B, "x times a vector"),
    ("ENC_GLU", 141, B, "the GLU"),
    ("ENC_DWCONV", 142, B, "the depthwise convolution of the audio encoder"),
    ("ENC_LOCAL_ATTN", 143, B, "the local attention of the audio encoder"),
    ("ENC_CLAMP", 144, B, "clamp the values"),
    ("ENC_BIAS_CLAMP", 145, B, "a bias, then clamp"),
    ("ENC_LNORM", 146, B, "the layer norm"),
    ("ENC_GELU", 147, B, "the GELU (tanh or erf)"),
    ("Q6K_LINEAR", 148, C, "the rows of a Q6_K matrix (the head of the parts)"),
    ("PART_PREFILL", 149, C, "the attention of a prompt block for the KV heads of a part"),
    ("ATTN_PREFILL_QC", 150, C, "the attention of a prompt block over the int16 cache"),
    ("KQ_QUANT16", 151, C, "the int16 x of a prompt block"),
    ("KQ_LINEAR16", 152, C, "its KQ_Q4X product"),
    # the GPU+parts mode of QwenGPU: the cold experts in a part for each node
    ("KQ_MOE_PART", 153, C, "the cold experts, each part its rows"),
    ("RUN_PARTS", 154, C, "the program runs in the pinned team of the parts"),
    ("HC_PRE", 155, G, "the input of a block from the normed streams (down, act, up, mix, inject)"),
    ("MOE_COMBINE", 156, G, "the routed experts of the GPU and the rows of the CPU, in the order of the router"),
    # DeepSeek V4.1 Flash (DEEPSEEK_V41_PLAN.md, kernels/dsv41)
    ("DS_MXX_LINEAR", 157, G, "x times a KQ_MXX matrix on int16 x (the operations of the CPU)"),
    ("DS_FP8_LINEAR", 158, G, "x times an FP8 matrix with E8M0 scales of 32 x 32 blocks"),
    ("DS_HC_MIX", 159, G, "the raw mixes of the streams of a sublayer"),
    ("DS_HC_PRE", 160, G, "the input of a sublayer: the pre mix of the streams, then the norm"),
    ("DS_HC_POST", 161, G, "the output of a sublayer back into the streams (post, comb)"),
    ("DS_ROPE", 162, G, "the rope of the last values of each head (adjacent pairs)"),
    ("DS_FP4", 163, G, "the FP4 rounding of the reference (E4M3 scales of 16 or E8M0 of 32)"),
    ("DS_WIN_WRITE", 164, G, "the rope and the E4M3 rounding of the window rows of a group"),
    ("DS_COMPRESS", 165, G, "the compressor: the pooled, normed latent of a group of tokens"),
    ("DS_CMP_WRITE", 166, G, "the compressed row and the index key of a group into their caches"),
    ("DS_IDX_SCORE", 167, G, "the scores of the indexer over the compressed entries"),
    ("DS_CAND", 168, G, "the max of each block of scores (the candidates)"),
    ("DS_TOPK", 169, G, "the k best scores, in the order of their indices"),
    ("DS_ATTN", 170, G, "the attention of a query over the window and the selected entries, with sinks"),
    ("DS_ROUTER", 171, G, "the router: sqrt-softplus, the bias for the choice, the scaled weights"),
    ("DS_SWIGLU", 172, G, "silu(min(g, limit)) clamp(u, -limit, limit)"),
    ("DS_ENGRAM", 173, G, "the gates of engram into the streams"),
    ("DS_HOT_MOE", 174, G, "the routed experts the GPU holds (KQ_MXX), the de row of each pair"),
    ("DS_COMBINE", 175, G, "the de rows of the GPU and of the CPU, in the order of the router"),
    ("DS_RING_PUT", 176, G, "the window rows of a group into the ring, after the attention"),
    ("DS_RING_RESTORE", 177, G, "the ring rows of the refused drafts back (DSpark)"),
    ("DS_STREAM_MEAN", 178, G, "the mean of the streams (the input of DSpark)"),
    ("DS_COMPRESS_FIX", 179, G, "the compressor's open group as after the accepted tokens (DSpark)"),
    ("DS_GATHER", 180, G, "the row of a token chosen on the GPU (the Markov embedding)"),
    ("DS_MARKOV", 181, G, "a draft position: the Markov bias into its logits, then their argmax"),
    ("DS_CONF", 182, G, "the confidence of each draft position"),
    ("DS_EMBED", 183, G, "the BF12 embedding rows of the tokens (pinned host memory) into the streams"),
    ("DS_ENGRAM_ROWS", 184, G, "the values of gathered engram rows (E4M3 codes, E8M0 scales)"),
    ("DS_ROWS_AT", 185, G, "the rows of a group at their positions into a buffer of rows from a slot (bounded replay)"),
    ("DS_TOPK_MERGE", 186, G, "the k best so far and the scores of a chunk: the k best (DS_TOPK in chunks)"),
    # the image input of DeepSeek V4.1 (models/deepseek41/vision.py)
    ("ENC_SILU_MUL", 187, B, "silu(gate) * up"),
    ("DS_EMBED_ROWS", 188, G, "the rows of an image span in place of the embedding, into the streams"),
    # LFM2-MoE (docs/LFM25_24B_PLAN.md)
    ("ROUTER_SIG", 189, B, "the router of LFM2: sigmoid scores, the bias for the choice, the k weights normed"),
    ("SHORT_CONV", 190, B, "the double-gated short convolution of LFM2 (B x, the causal conv, times C)"),
    # the transfers of the mixed groups of Qwen (gpu35.QwenGPU._moe_mix) at half the bytes
    ("Q16", 191, B, "float32 values as int16, a float32 scale for each 128 (the int16, then the scales)"),
    ("DQ16", 192, B, "the float32 values of Q16"),
    # GLM-5.3-Flash (docs/GLM53_PLAN.md)
    ("F16_GLINEAR", 193, G, "x times a float16 matrix in groups (block diagonal: GLM's kv_b of each head)"),
    ("GLM_POOL", 194, G, "the index keys normed, the pools of 4 (softmax of gate + ape) into the pooled-key cache"),
    ("GLM_EXPAND", 195, G, "the tokens of the chosen pools and the open tail of each query"),
    ("GLM_HC_MIX", 196, G, "the mixes of the streams over many blocks, then the post and comb coefficients once a token"),
    ("GLM_HC_POST", 197, G, "post y + comb^T res from the coefficients of GLM_HC_MIX"),
    ("GLM_ROPE", 198, G, "the neox RoPE of rows at positions pos + j (the DSpark drafter)"),
    ("GLM_RING_PUT", 199, G, "the real rows (the last W) into the slots (pos + j) % W of a ring"),
    ("GLM_DSP_ATTN", 200, G, "the drafter's attention: its block over the context ring and its own rows"),
    ("GLM_ATTN", 201, G, "the MLA of a prompt group on the tensor cores (64 heads over their selected latents)"),
    # Qwen3.8-Flash-Next-FP4-Local (docs/FP4LOCAL_PLAN.md)
    ("BF16_ROUND", 202, B, "x (n values) rounded to bfloat16 in place (the activations of vLLM's contract)"),
    ("B12X_GLU", 203, G, "the act of the contract, bf16(bf16(g sigma(g)) u) of g, u in bfloat16"),
    ("MOE_B12X_OUT", 204, G, "the MoE output of the contract: bf16(bf16(routed + cpu) + the gated shared expert)"),
    ("MX8_ACT", 205, B, "x (n values) as dynamic MXFP8 in place: e4m3(x / 2^e) 2^e, e = ceil(log2(amax / 448)) of each 32"),
    # Nemotron 3.5 Bolt (docs/NEMOTRON_BOLT_PLAN.md)
    ("NM_MAMBA", 206, B, "the Mamba-2 mixer: the causal conv, the recurrence of the state, the gated group norm"),
    ("RELU2", 207, B, "relu(x)^2"),
    ("NM_MOE", 208, C, "the routed experts of the latent: up, relu^2, down, the weighted sum"),
    ("NM_ATTN", 209, G, "the attention of each query over the float rows of the cache up to its position (no RoPE)"),
    ("NM_SPLIT", 210, G, "the pairs of the experts the GPU does not hold (the CPU's), and their count"),
    ("NM_HOT_MOE", 211, G, "the routed experts of the latent the GPU holds: up, relu^2, down, the sum of each token"),
    ("NM_QX", 212, G, "float rows as int8 with a scale of each 32 (the x of the dp4a products)"),
    ("NM_IQ2_LIN", 213, G, "an IQ2_XXS matrix on the int8 rows of NM_QX (dp4a)"),
    ("NM_KV", 214, G, "the keys and values of a run into the bfloat16 rows of the cache"),
    ("NM_ATTN2", 215, G, "the attention in chunks of positions (a KV head and its query heads a block), then the merge"),
    ("NM_GROUP_MOE", 216, G, "the routed experts of a prompt group on the GPU (cold ones read from mapped host memory)"),
    ("NM_GEMM8", 217, G, "an IQ2_XXS, NVFP4 or Q8_0 matrix on the int8 rows of a prompt group (tiles of 64 x 64, dp4a)"),
    ("NM_QX2", 218, C, "float rows as int16 in two int8 planes and a scale of each 32 (kq_quant_part16)"),
    ("NM_LIN2", 219, C, "an IQ2_XXS, NVFP4 or Q8_0 matrix on the two planes of NM_QX2"),
    ("NM_ADDNORM", 220, B, "the residual sum and the norm of the next block in one pass: h += a (+ b), out = rmsnorm(h) w"),
    ("NM_DPREP", 221, B, "the drafter's q and k norms and NEOX rope; k and v into the ring of the context (LiLiCorr)"),
    ("NM_DATTN", 222, B, "the drafter's attention: a block over the window of the ring and itself, a sink a head"),
    ("NM_DCONV", 223, B, "the drafter's grouped dynamic conv over a block (base kernel + the token's coefficients)"),
    ("NM_LTOPK", 224, B, "the LiLiCorr candidates: the k best of each slot's logits and their features"),
    ("NM_LATT", 225, B, "the LiLiCorr attention over the candidates with the slot biases"),
    ("NM_LROWS", 226, B, "rows plus a bias and rows (repeated), then an activation (none, silu)"),
    ("NM_LCAT", 227, B, "[x, a, x a] of rows and a vector"),
    ("NM_LSCORE", 228, B, "the LiLiCorr lattice scores of the L2-normed out and in vectors"),
    ("REMOTE_POST", 229, C, "the cold experts of a step that an expert device holds to it, the others to a list"),
    ("REMOTE_JOIN", 230, C, "wait for the expert device's job and add its sum"),
    ("NM_REMOTE_POST", 231, C, "the cold pairs of a Nemotron step that an expert device holds to it, the others to a list"),
    ("DS_HC_STEP", 232, G, "a sublayer's post, then the next one's mixes and pre (DeepSeek V4.1)"),
    ("NM_HOT_NV4", 233, G, "NM_HOT_MOE for KQ_NV4 experts (Nemotron Super): float x, a warp a row"),
    ("NM_PLAN", 234, G, "a prompt group's experts: the cold ones of few pairs the CPU's, the rest the GPU's (by their costs)"),
    ("DS_STAGE_PICK", 235, G, "a prompt group's heaviest cold experts of a layer to the GPU's stage slots (DeepSeek V4.1)"),
    ("DS_STAGE_PULL", 236, G, "the experts of DS_STAGE_PICK from the registered host memory into the stage, beside the stream"),
    ("DS_STAGE_JOIN", 237, G, "the stream waits for a DS_STAGE_PULL"),
]

OPS = {name: code for name, code, _, _ in TABLE}
OP_NAMES = {code: name for name, code in OPS.items()}
RUNNERS = {name: runners for name, _, runners, _ in TABLE}
globals().update(OPS)
__all__ = ["NARG", "MAGIC", "T_NONE", "T_INT", "T_F32", "T_SLOT", "OP_NAMES"] + list(OPS)

HEADER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backends",
                      "include", "opcodes.h")


def header():
    """The text of backends/include/opcodes.h."""
    w = max(len(n) for n in OPS)
    rows = []
    for name, code, runners, text in TABLE:
        rows.append("    GP_%-*s = %3d,   /* %-7s %s */" % (w, name, code, "+".join(runners), text))
    cases = []
    line = "    "
    for name in OPS:
        item = "case GP_%s: " % name
        if len(line) + len(item) > 92:
            cases.append(line.rstrip())
            line = "    "
        line += item
    cases.append(line.rstrip())
    return """/* The operation codes of the programs. Generated by reclank/ir/opcodes.py
 * (python -m reclank.ir.opcodes): do not edit. */
#ifndef RECLANK_OPCODES_H
#define RECLANK_OPCODES_H

#include <stdint.h>

#define GP_NARG %d
#define GP_MAGIC 0x%XLL   /* "GPROG001" */

typedef struct {
    int32_t op;
    int32_t flags;
    uint8_t tag[GP_NARG];
    int64_t v[GP_NARG];
} gp_rec;

enum { GP_T_NONE = %d, GP_T_INT = %d, GP_T_F32 = %d, GP_T_SLOT = %d };

enum {
%s
};

/* 1 for a code of the table, else 0: a runner fails on an unknown code. */
static inline int gp_op_known(int op)
{
    switch (op) {
%s
        return 1;
    default:
        return 0;
    }
}

#endif
""" % (NARG, MAGIC, T_NONE, T_INT, T_F32, T_SLOT, "\n".join(rows), "\n".join(cases))


def main(argv):
    text = header()
    old = open(HEADER).read() if os.path.exists(HEADER) else None
    if "--check" in argv:
        if old != text:
            print("%s differs from reclank/ir/opcodes.py: run python -m reclank.ir.opcodes" % HEADER)
            return 1
        return 0
    if old != text:
        with open(HEADER, "w") as f:
            f.write(text)
        print("wrote", HEADER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
