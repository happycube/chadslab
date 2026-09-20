# numpy-gemma — NumPy-only Gemma 4 12B runtime

A from-scratch, NumPy-only inference runtime for
`google/gemma-4-12B-it-qat-q4_0-unquantized`, built as the Phase 1 baseline of
[../Gemma LLM Runtime Learning Plan.md](../Gemma%20LLM%20Runtime%20Learning%20Plan.md).
The runtime imports **only NumPy** — no PyTorch, no transformers, no `safetensors`
package (the loader is ~90 lines of `struct` + `mmap` + bit-shift bf16 decoding; the
tokenizer is pure Python over `tokenizer.json`).

## Status

| Area | State |
|---|---|
| Safetensors mmap reader, bf16 -> f32 | done |
| RMSNorm, GeGLU(tanh), linear, softmax, softcap | done |
| GQA + per-head QK-norm, default + proportional RoPE | done |
| Interleaved sliding/global attention (per-layer head_dim) | done |
| 48-layer forward, tied lm_head, logit softcapping | done |
| KV cache + autoregressive decode | done |
| BPE tokenizer, byte fallback, chat template | done |

## Verified against Hugging Face

| Scope | Result |
|---|---|
| layer 0, position 0 (single token) | 16/16 tensors, cosine >= 0.999996 |
| layer 0, position 3 (RoPE active) | 16/16 tensors, cosine >= 0.99999 |
| through global layer 5 (K=V, p-RoPE) | 85/85 tensors, cosine >= 0.999 |
| full 48 layers + logits | 671/671 tensors, cosine >= 0.999 |
| `last_hidden_state` (post final norm) | cosine 0.999678 |
| greedy first generated token | HF 50429 = NumPy 50429 |
| KV cache vs batch prefill | relative L2 4.8e-07 |
| tokenizer encode / decode vs HF | 25/25 and 25/25 exact |
| chat template vs HF (5 message shapes) | 5/5 exact |

`scripts/check_trace.py` diffs every intermediate against a captured HF bf16 trace
(`../gemma4-12b-qat-pytorch/scripts/capture_trace.py`). Differences are bf16-rounding
scale: HF computes in bf16 (with fp32 norms/softmax), this runtime computes entirely
in fp32, so error accumulates with depth; cosine stays >= 0.999 throughout.

## Layout

```
numpy-gemma/
├── np_gemma/
│   ├── st.py          # safetensors mmap reader (bf16 -> f32)
│   ├── config.py      # config + per-layer plan (head_dim, kv heads, k_eq_v)
│   ├── ops.py         # rms_norm, linear, gelu_tanh, softmax, softcap
│   ├── rope.py        # default + proportional RoPE
│   ├── model.py       # KVCache, attention, 48-layer forward, generate
│   └── tokenizer.py   # BPE + byte fallback + chat template
└── scripts/
    ├── check_trace.py     # diff every intermediate against an HF trace
    ├── check_cache.py     # incremental decode vs batch prefill
    ├── check_tokenizer.py # differential test vs HF AutoTokenizer
    ├── generate.py        # autoregressive decode (ids in/out)
    └── chat.py            # text -> tokens -> forward -> text
```

## Run

```bash
cd numpy-gemma
PY=../gemma4-12b-qat-pytorch/.venv/bin/python
SNAP=$(dirname "$(find ../gemma4-12b-qat-pytorch/.cache/huggingface -name model.safetensors | head -1)")

# full 48-layer diff against the captured reference trace
PYTHONPATH=. $PY scripts/check_trace.py \
    --config "$SNAP/config.json" --weights "$SNAP/model.safetensors" \
    --trace ../gemma4-12b-qat-pytorch/traces/ref-france-pos3

# tokenizer differential test vs HF (fast)
PYTHONPATH=. $PY scripts/check_tokenizer.py --snapshot "$SNAP"

# KV-cache consistency (fast, use --layers 1)
PYTHONPATH=. $PY scripts/check_cache.py --config "$SNAP/config.json" \
    --weights "$SNAP/model.safetensors" \
    --trace ../gemma4-12b-qat-pytorch/traces/ref-france-pos3 --layers 1

# end-to-end chat (slow: one prefill, then ~5-6 min/token)
PYTHONPATH=. $PY scripts/chat.py --snapshot "$SNAP" \
    --prompt "The capital of France is" --max-new-tokens 1
```

Add `--layers N` to `check_trace.py` for a fast partial check (layer 0 loads only
~0.5 GB of the 24 GB checkpoint). Generation is I/O bound: every token re-reads all
weights from the mmap unless `--cache-weights` keeps ~50 GB of float32 weights resident.

## Architecture notes (these are the Gemma 4 traps)

* **Per-layer geometry.** 48 layers; 40 sliding layers use `head_dim=256`, 8 KV heads
  (q 4096, k/v 2048, o 4096->3840); the 8 global layers (indices 5,11,...,47) use
  `head_dim=512`, **1** KV head, q 8192, o 8192->3840.
* **K=V on global layers** (`attention_k_eq_v`): there is no `v_proj`. V is the raw
  key projection, RMSNorm-normalized (no scale, no RoPE), while K gets k_norm + RoPE.
* **RMSNorm multiplies its weight directly** — no `1 + weight` offset in this checkpoint,
  and the weights are large (mean 6.6, max 193), so adding 1 would be wrong.
* **Attention scaling is 1.0**, not `1/sqrt(head_dim)`; per-head QK-norm replaces it.
* **RoPE**: sliding layers use the default scheme, `theta=1e4`. Global layers use
  **proportional RoPE** with `partial_rotary_factor=0.25`, `theta=1e6`: only 64 of the
  256 angle pairs rotate, and the inverse frequencies are zero-padded to the full
  head_dim so `rotate_half` (NeoX half-split) pairs dim i with dim i+head_dim/2.
* **Embedding scale** is `sqrt(hidden_size)` = sqrt(3840).
* **Per-layer `layer_scalar`** is a trained scalar (0.0544 at layer 0, 0.365 at layer 5,
  0.0496 at layer 47) that multiplies the block output. It is not 1.
* **Tied embeddings** and **final logit softcapping** of 30 (`tanh(logits/30)*30`).
* Attention mask is causal; sliding layers additionally mask keys older than 1024.
* **Tokenizer**: normalizer maps space -> U+2581, BPE with `byte_fallback`, decoder is
  Replace(U+2581 -> space) + ByteFallback + Fuse. Chat template opens a system turn with
  `<|think|>` when thinking, and closes an empty `<|channel>thought\n<channel|>` when not.

## Next steps

1. Trim the per-layer KV cache to the sliding window and use a ring buffer.
2. Optional bf16 rounding after each op to track HF more tightly.
3. Port the verified graph to C (Phase 3).
