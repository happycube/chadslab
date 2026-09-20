
I had Der^Hepseek 4.1 Flash (in deepseek harness) make me a simple CPU runtime for gemma 4 12B QAT
as a preliminary for the project.  It gets .28 tps in 4-bit mode.

## Sample output

[gen ] 256 tokens in 890.3s (0.29 tok/s)

===== response =====
**Grouped-Query Attention (GQA)** is an architecture designed to balance the efficiency of Multi-Head Attention (MHA) and the speed of Multi-Query Attention (MQA). It is currently a standard component in many modern Large Language Models (like Llama 2 and 3).

Here is a brief breakdown of how it works and why it matters:

### 1. The Context: The Spectrum
To understand GQA, you have to look at the two extremes it sits between:

*   **Multi-Head Attention (MHA):** Every "Query" head has its own corresponding "Key" and "Value" head.
    *   *Pros:* Highest quality/representation.
    *   *Cons:* Very memory-intensive and slow during inference because the KV cache (the memory of previous tokens) is huge.
*   **Multi-Query Attention (MQA):** All "Query" heads share a *single* "Key" and "Value" head.
    *   *Pros:* Extremely fast and memory-efficient (tiny KV cache).
    *   *Cons:* Significant loss in model quality because the shared heads become a bottleneck for information.

### 2. How GQA

From Deepseek 4.1 flash:

# Gemma 4 12B QAT — PyTorch reference environment

A self-contained [uv](https://docs.astral.sh/uv/) virtualenv for running Google's
**Gemma 4 12B QAT** checkpoint (`google/gemma-4-12B-it-qat-q4_0-unquantized`) in
PyTorch, quantized to 4-bit at load time with bitsandbytes NF4 (CPU) or torchao
int4 (CUDA).

This is the known-good reference companion to the from-scratch runtime in
[../Gemma LLM Runtime Learning Plan.md](../Gemma%20LLM%20Runtime%20Learning%20Plan.md):
use it to generate golden tokens/logits and to sanity-check your own kernels.

## What gets run

| | |
|---|---|
| **Checkpoint** | [google/gemma-4-12B-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized) |
| **Size on disk** | **23.95 GB** (11,959,730,224 params, stored bf16) |
| **Quantization** | Q4_0 **quantization-aware training**; weights sit on the 4-bit symmetric lattice, group size 32 |
| **Architecture** | gemma4_unified, 48 layers, hidden 3840, 16 heads / 8 KV heads, head_dim 256, sliding-window 1024 interleaved with global attention, 256K context, text + image + audio |

The "unquantized" QAT repo deliberately ships **half-precision** weights extracted
from the QAT pipeline: they are already trained to be robust to 4-bit rounding, so
re-quantizing them in PyTorch is near-lossless. The sibling repo
[google/gemma-4-12B-it-qat-w4a16-ct](https://huggingface.co/google/gemma-4-12B-it-qat-w4a16-ct)
is the same weights **pre-serialized** in compressed-tensors w4a16 (~10.3 GB, no
in-process quantization step, vLLM-friendly).

## Measured on this machine (CPU-only, no NVIDIA driver)

| mode | result | model footprint | load time |
|---|---|---|---|
| ``` (no quant) | expected ~24 GB | — | — |
| `--quant int8` | loads + generates; torchao Int8Tensor | 23.9 GB reported* | 242 s |
| `--quant nf4` | **loads + generates; real 4-bit (Params4bit, uint8)** | **7.50 GB** | 306 s |
| `--quant int4` | **fails on CPU**: `ImportError: Requires mslk >= 1.0.0` | — | — |

\* `get_memory_footprint()` reports 4 bytes/elem for Int8Tensor, so its number is not
the real int8 size (int8 stores ~12 GB). Use RSS / `du` for truth. Peak RSS during any
load is ~29 GB because the bf16 shards are staged before quantization; it does not
mean the resident model is 29 GB.

Smoke command used:

```bash
python scripts/chat.py --quant nf4 --inspect --max-new-tokens 16 \
    --prompt "Say hello in exactly three words."
# -> [insp] model.language_model.layers.0.self_attn.q_proj: Params4bit dtype=torch.uint8
# -> Hello, how are you?
```

CPU decode is ~0.1 tok/s for this 12B model; that is expected without a GPU.

## Layout

```
gemma4-12b-qat-pytorch/
├── setup.sh                 # create .venv, install torch + requirements
├── requirements.txt         # Python deps (torch itself is installed by setup.sh)
├── requirements-optional.txt
├── scripts/
│   ├── _bootstrap.py        # pins HF caches inside ./.cache (sandbox-safe)
│   ├── check_env.py         # verify versions + load config/processor (no weights)
│   ├── download_model.py    # snapshot_download the checkpoint (~24 GB)
│   ├── inspect_weights.py   # safetensors dtype/size histogram
│   └── chat.py              # load + generate, with quantization options
├── .venv/                   # the virtualenv            (gitignored)
└── .cache/                  # uv / pip / Hugging Face caches (gitignored)
```

## Quick start

```bash
cd gemma4-12b-qat-pytorch

./setup.sh                     # CPU build by default; TORCH_VARIANT=cu130 for NVIDIA

source .venv/bin/activate
python scripts/check_env.py    # fast: no model weights downloaded
python scripts/download_model.py   # ~24 GB into ./.cache/huggingface
python scripts/chat.py --quant nf4 --prompt "Explain grouped-query attention briefly."
```

Everything stays inside this directory. The DSH file sandbox makes `~`/$HOME
read-only, so _bootstrap.py forces HF_HOME=./.cache/huggingface, and setup.sh
forces the uv/pip caches into ./.cache too.

### Portability (sshfs)

This project is sometimes reached over sshfs at a different absolute path than
where it really lives (the real location is
`jackal:~/proj/chadslab/llm/gemma4-12b-qat-pytorch`). setup.sh therefore builds a
**relocatable** venv that carries no host-specific path:

* `.venv/bin/python` is a *relative* symlink to the project-local interpreter in
  `.cache/uv-python/`
* `.venv/bin/activate` derives `VIRTUAL_ENV` from its own location (`--relocatable`)
* console scripts use `#!/usr/bin/env python3`
* `pyvenv.cfg` has a relative `home`

Always `source .venv/bin/activate` before running (so `env python3` resolves to the
venv), or call `.venv/bin/python` directly. Re-running `./setup.sh` is safe and
idempotent; run it on the machine you actually use if you want `pyvenv.cfg` to
reflect that machine's path.

### Quantization modes (--quant)

| --quant | Backend | Bytes/param | CPU | CUDA |
|---|---|---|---|---|
| int4 | torchao Int4WeightOnlyConfig(group_size=32) | 0.5 | needs mslk (no CPU wheel) | yes |
| nf4 | bitsandbytes NF4 (load_in_4bit) | 0.5 | **yes, tested** | yes |
| int8 | torchao Int8WeightOnlyConfig | 1.0 | **yes, tested** | yes |
| bf16 / none | no quantization | 2.0 | yes | yes |
| auto (default) | int4 on CUDA, int8 on CPU | | | |

```bash
# CPU: real 4-bit
python scripts/chat.py --quant nf4 --prompt "..."
# CPU: 8-bit
python scripts/chat.py --quant int8 --prompt "..."
# CUDA: int4 matching the QAT lattice (install mslk first, see setup.sh)
python scripts/chat.py --quant int4 --group-size 32 --prompt "..."
# reference output, unquantized
python scripts/chat.py --quant bf16 --prompt "..."
# load the pre-quantized w4a16 checkpoint instead
python scripts/chat.py --quant none --model google/gemma-4-12B-it-qat-w4a16-ct
```

Generation defaults follow the model card: temperature=1.0, top_p=0.95,
top_k=64; add --thinking to enable the reasoning trace.

### About int4 on CPU

torchao 0.18 routes int4 weight-only through Meta's **MSLK** kernels, and mslk only
ships per-CUDA wheels under `https://download.pytorch.org/whl/` (there is no CPU
build). setup.sh installs mslk automatically when TORCH_VARIANT is a CUDA channel.
On a CPU box, use **--quant nf4** for 4-bit or **--quant int8**.

## Hardware notes

* **No NVIDIA driver in this workspace.** nvidia-smi fails, so setup.sh selected the
  **CPU** build (torch 2.14.0+cpu). Memory is fine (125 GB RAM) but throughput is
  ~0.1 tok/s for this 12B model.
* For speed, rebuild on a CUDA host: TORCH_VARIANT=cu130 ./setup.sh, then
  --quant int4. For serving, use vLLM with the -w4a16-ct repo.
* Disk: the checkpoint is 24 GB and the venv ~3 GB. Keep ~30 GB free.

## Related checkpoints

| Repo | Format | Use |
|---|---|---|
| google/gemma-4-12B-it-qat-q4_0-unquantized | bf16, Q4_0 lattice | **default here** |
| google/gemma-4-12B-it-qat-w4a16-ct | compressed-tensors w4a16 | pre-quantized, vLLM |
| google/gemma-4-12B-it-qat-q4_0-unquantized-assistant | bf16 | MTP speculative-decoding drafter |
| google/gemma-4-12B-it | bf16 | non-QAT baseline for accuracy comparisons |

## Troubleshooting

* **`ModuleNotFoundError: torchvision`** — the Gemma 4 processor imports
  `torchvision.transforms.v2`. Run ./setup.sh (it installs torch/torchvision/
  torchaudio from the same channel) rather than installing only requirements.txt.
* **`ImportError: Requires mslk >= 1.0.0`** — torchao int4 was requested on a CPU-only
  or mslk-less install; use --quant nf4 or --quant int8.
* **`Read-only file system` writing to ~/.cache** — expected under the DSH sandbox;
  keep using the scripts, which redirect caches into ./.cache.
* **Gated-repo 401/403** — Gemma 4 weights may require accepting the license and
  `hf auth login`. Set HF_TOKEN if needed.
