
# reclank : by clankers, for running clankers

This is still a research project in running LLM's in vibe-coded code, 
definitely not a product yet!  (There are plenty of other projects 
kinda like this one, but this one is mine.)

The main things about it:

- BF12 format for crunching bf16 weights by 23% with <.01% near-zero
  weights zeroed out.  Great for saving VRAM in MoE models when you
  want unquantized dense weights!
- very few outside dependencies.
- Should work from low (gemma 4 e4b, 26b, qwen 3.6 35a3b) up to Flash
  (Qwen 3.8 F-N NVFP4, Deepseek 4.1-Flash with virtually-original weights)
- Has full NUMA support for dual Xeons/EPYC/etc.
- Can use two GPUs in one process: a second card holds more experts.
- Speculative decoding: MTP drafters, DSpark for DeepSeek, LiLiCorr for
  Nemotron.

What it doesn't have yet:

- No AMD*/Intel support.  Pascal (GTX 1070) and Maxwell run, but get
  less testing than Ampere and newer.

(* - I want to bypass ROCm and run on Polaris, maybe even Hawaii, by 
going to /dev/amdgpu and /dev/amdkfd, but that's another project!)

Maybe I'll deslop it enough to be useful to other people.  Maybe ;)

## What this is

reclank runs large language models on a CPU, on a GPU, or on both. The
runtime uses NumPy, C, and CUDA. It does not use PyTorch. It does not use
transformers. NumPy is the only outside dependency of the runtime. The
check scripts and the bench scripts use torch and transformers only to
build a reference.

A C kernel reads the packed weights in place. The file layout is the
runtime layout. The model builds no float32 copy of a weight. One decode
step runs as one program: a list of records that the CPU runner and the
GPU runner both execute. See `docs/PROGRAMS.md`.

This file follows ASD-STE100, Simplified Technical English. A sentence in a
description has 25 words or less. A sentence in an instruction has 20 words or
less.

`scripts/check_ste100.py` checks a mechanical subset of these rules on
any Markdown file. It uses only the Python standard library. Use `--ing-nouns`
to add technical terms for a subject area. It does not check the full standard.

Run it on this file with `python3 scripts/check_ste100.py README.md --ing-nouns scripts/ste100_ing_nouns.txt`.

The long benchmark log that followed the old marker is in the git history.
The files in `docs/` hold the measured detail.

## Quick start

These steps start a server with a small model, then send it one request.
"Start here" and "The server" below give the details.

1. Get the code and NumPy. You need Python 3.10 or newer and a C
   compiler.

```sh
git clone https://github.com/happycube/chadslab.git
cd chadslab/llm/reclank
python3 -m venv .venv
. .venv/bin/activate
pip install numpy
export PYTHONPATH=.
```

2. Get a GGUF model file. Good first models are Gemma 4 26B-A4B
   (the Unsloth UD-Q4_K_XL file) and LFM2.5-24B-A2B (the LiquidAI
   QAD Q4_0 file). Keep model files out of this folder.

3. Start the server:

```sh
python -m server --model ~/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf --port 8080
```

   Gemma runs on the CPU by default. Add `--backend cuda` to use an
   NVIDIA GPU. Qwen and LFM use the GPU by default. Add `--backend cpu`
   to keep them on the CPU.

   The first start builds the kernels. The C build takes about a minute.
   The CUDA build takes some minutes.

4. Send a chat request:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

   Any OpenAI compatible client works. Set its base URL to
   `http://127.0.0.1:8080/v1` and its key to `change-me`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="change-me")
reply = client.chat.completions.create(
    model="reclank", messages=[{"role": "user", "content": "Hello!"}])
print(reply.choices[0].message.content)
```

   Set your own key with `--api-key`. Stop the server with Ctrl-C.

A large model such as DeepSeek V4.1 Flash needs more steps. See "The
server" and "Weightkeeper".

## Models

| Model | Kind | Package |
|---|---|---|
| Gemma 4 26B-A4B | Mixture of experts, quantization-aware training | `reclank/models/gemma4/` |
| Gemma 4 12B | Dense, quantization-aware training | `reclank/models/gemma4/` |
| Gemma 4 E4B and E2B | Small dense models with Per-Layer Embeddings | `reclank/models/gemma4/e4b.py` |
| Qwen3.6-35B-A3B | MoE, 256 experts, DeltaNet plus full attention | `reclank/models/qwen/` |
| Qwen3.8-Flash-Next | MoE, 125B, 512 experts, NVFP4, MTP drafter | `reclank/models/qwen/` |
| DeepSeek V4.1 Flash | MoE, 510 GB checkpoint, MXFP4 experts, engram tables, DSpark drafter | `reclank/models/deepseek41/` |
| LFM2.5-24B-A2B | MoE, 64 experts, short convolutions plus attention | `reclank/models/qwen/` |
| Nemotron 3.5 Bolt | MoE, Mamba plus attention, LiLiCorr drafter | `reclank/models/nemotron/` |
| GLM-5.3-Flash | MoE, EXL3 experts, DSpark drafter (check scripts, no server yet) | `reclank/models/glm53/` |

The encoders in `reclank/encoders/` add images, audio, and video clips to
these models. The server takes an encoder file with `--mmproj`.

## Weight formats

The runtime reads the standard GGML types in place:
`F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K,`
`Q5_K, Q6_K, Q8_K, IQ4_NL, IQ4_XS`.

The runtime also gives its own types inside a GGUF file:

| Type | Bits | What it is |
|---|---|---|
| BF12 | 12.25 | bf16 weights, near-zero values zeroed, no real loss |
| RQ8_0 | 8.5 | Q8_0 blocks over TQ6-rotated rows |
| RQ6_K | 6.5 | K-quant blocks over rotated rows |
| RQ5_K, RQ4_K | 5.5, 4.5 | The same idea at other sizes |
| NV4, NVX | 4 | NVFP4 rows from an NVIDIA ModelOpt checkpoint |
| E4M3_ROWS | 8 | FP8 rows |

Other weight files work too: compressed-tensors SafeTensors (the E4B
mobile checkpoint), NVFP4 ModelOpt checkpoints, EXL3 trellis files, and
the Unsloth UD-Q4_K_XL GGUF files. The runtime also reads the original
Qwen3.8 bf16 checkpoint and the DeepSeek V4.1 safetensors shards in
place. `scripts/convert_nvfp4_gguf.py` and
`scripts/convert_q8_gguf.py` make reclank forms from a checkpoint.

## Files

    reclank/
    ├── README.md          This file.
    ├── docs/              Plans and notes. See "Documentation".
    ├── reclank/           The package.
    │   ├── settings.py    The RECLANK_* settings. The old NP_GEMMA_* names still work.
    │   ├── runtime/       threads.py: the thread counts. numa.py: the NUMA placement.
    │   ├── formats/       gguf.py, safetensors.py, nvfp4_st.py, tq6.py, exl3.py, dsv41_st.py.
    │   ├── ir/            opcodes.py: the operation codes. program.py: the records, the compiler.
    │   ├── core/          ops.py, sampling.py, speculative.py (MTP), think_guard.py, session.py.
    │   ├── encoders/      runtime.py and media.py: the media encoders.
    │   ├── models/        gemma4/, qwen/, deepseek41/: one package per family.
    │   └── backends/      cpu/: the C kernels through ctypes. cuda/: gpu.cu, the GPU memory.
    ├── server/            The OpenAI compatible HTTP server. Run it with python -m server.
    ├── np_gemma/          The old name of the package. It gives the modules of reclank.
    ├── tests/             unit/ (pytest), golden/ (bit-exact dumps), quality/ (the suite).
    ├── sexp/              Readable listings of the compiled programs. See sexp/README.md.
    └── scripts/           serve.py, session.py, chat.py, generate.py, and about 180
                           check, bench, profile, and convert scripts.

Model weights never live in this repository. The `.gitignore` file keeps
`models/`, `models2/`, and `*.gguf` out of it.

## Start here

You need Python 3.10 or newer, NumPy, and a C compiler. For the GPU path
you need an NVIDIA GPU, the CUDA toolkit, and nvcc.

Run every command from the root of this folder with the package on the
path:

```sh
export PYTHONPATH=.
```

The C kernels build on first use. The CUDA kernels build on first use too.
Both caches live under `~/.cache/reclank/libs/`, out of the source tree.
Check what the build will choose:

```sh
python -m reclank.backends.cpu.build --explain avx512_vnni
python -m reclank.backends.cuda.build --explain
```

The default thread count is one OpenMP thread per physical core. One BLAS
thread is much faster for the small attention matrices:

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=6
```

The first int8 load builds a weight cache in `~/.cache/reclank/weights`.
That load takes minutes. A later load takes seconds.

## The server

One server runs every family. It uses only the Python standard library.
It finds the family from the GGUF architecture or from the `config.json`
of a checkpoint folder.

```sh
python -m server --model MODEL.gguf --ctx 98304 --port 8081
```

The three wrapper scripts set the options of one family for you:

```sh
# Gemma 4 26B from the Unsloth file, CPU, greedy
python scripts/serve.py \
    --gguf models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
    --model-id gemma-4-26B_q4_0-it --dtype int4 --port 8080 --temperature 0.0

# Qwen3.8-Flash-Next on the GPU, experts partly hot
python scripts/serve_qwen4.py --ctx 98304 --port 8081

# DeepSeek V4.1 Flash from its checkpoint folder, with the weightkeeper
python scripts/serve_dsv41.py -m CKPT_DIR
```

`scripts/serve.py` maps the old flags to the new ones: `--gguf` to
`--model`, `--gpu dense` to `--backend cuda --gpu-dense`, and `--gpu hot`
to `--backend cuda`. `scripts/serve_dsv41.py` forces
`OMP_NUM_THREADS=1`, because that path needs it.

The paths are:

    GET  /v1/models               List the model.
    GET  /v1/models/{id}          One model.
    POST /v1/chat/completions     A chat turn. Set stream true for the events.
    POST /v1/completions          A raw text prompt.
    GET  /health                  The status.

The server also takes the bare paths with no `/v1`, because some clients
use them. A stream uses the server-sent-event format and ends with the
`data: [DONE]` line. Set `stream_options.include_usage` to get the token
counts in the last event.

The model shares one key and value cache. The server answers one request
at a time. The other requests wait.

The server splits the thought channel. It puts the thought in
`reasoning_content` and the answer in `content`. Open the channel with
`--thinking`, or per request with the `thinking` field, with
`reasoning_effort`, or with `chat_template_kwargs.enable_thinking`. The
choices are auto, on, off, low, medium, high, and xhigh.

The server gives tool calls. Send the OpenAI `tools` field. The model
answers with `tool_calls` and the finish reason `tool_calls`. Send the
result back as a message with the role `tool`.

A client points at `http://127.0.0.1:8080/v1`. The Bearer key defaults to
`change-me`. Set it with `--api-key` or with `RECLANK_API_KEY`. An empty
key turns the check off.

Key options of `python -m server`:

| Option | Default | What it does |
|---|---|---|
| `--model`, `-m` | required | A GGUF file or a checkpoint folder. |
| `--family` | auto | One of gemma4, qwen, deepseek41, nemotron. |
| `--backend` | auto | cpu or cuda. Auto picks CPU for Gemma, GPU for Qwen. |
| `--gpu-dense` | off | Put the dense weights and the head on the GPU, the experts on the CPU. |
| `--hot-gb` | model default | Put the most used experts on the GPU, up to this size. |
| `--resident` | off | Put all the weights on the GPU. |
| `--experts FILE` | off | Read the routed experts from a second file in another form. |
| `--ctx` | model default | The size of the key and value cache in tokens. |
| `--kv-attn` | int16 | Gemma: cache read form. One of int16, int8, k16v8, rq8, k16vr8, float. |
| `--drafter DIR`, `--mtp N` | off | Multi-token prediction with a drafter, N drafts a step. |
| `--thinking` | off | auto, on, off, or an effort level. |
| `--mmproj FILE` | off | The encoder file for images, audio, and video. |
| `--temperature`, `--top-k`, `--top-p` | model defaults | Sampling. |
| `--debug DIR` | off | Write one record per turn into DIR. |

Use `python -m server --help` for the full list.

`scripts/serve_forever.sh` starts a server again after an error exit:

```sh
scripts/serve_forever.sh python scripts/serve_qwen4.py --port 8082
```

## Command-line scripts

These scripts run the NumPy path of the Gemma models from a Hugging Face
snapshot. They need no server.

```sh
# One load, many prompts. Type :reset or :q inside.
python scripts/session.py --snapshot "$SNAP" --dtype int8

# One prompt, then stop.
python scripts/chat.py --snapshot "$SNAP" --dtype int8 --prompt "Hello."

# Token ids in, token ids out. For a reference comparison.
python scripts/generate.py --config C --weights W --input-ids "1,2,3"
```

The `--dtype` choices are f32, bf16, int8, and int4. The f32 mode needs
about 70 GB of memory for the 12B. The int8 mode needs about 12 GB and
gives the same tokens in the tests.

`scripts/gguf_generate.py` runs a GGUF file from the command line. It
takes the same `--gpu` modes as the server.

## Settings

`reclank/settings.py` reads the `RECLANK_*` environment variables. The old
`NP_GEMMA_*` names still work. A `RECLANK_*` name wins. The code holds
about 250 settings, mostly for kernel control. `grep -r RECLANK_ reclank/`
lists them. The important ones:

| Variable | Default | What it does |
|---|---|---|
| `RECLANK_GPU` | off | The switch of the GPU path. 1 turns CUDA on. |
| `RECLANK_GPU_DEVICE` | 0 | The CUDA device, as nvidia-smi lists it. |
| `RECLANK_EXPERT_GPUS` | off | A second GPU that holds more experts. See `docs/TWO_GPU.md`. |
| `RECLANK_GPU_ARCH` | from nvidia-smi | The CUDA architecture, for example sm_120a. |
| `RECLANK_CPU` | auto | The C tier: scalar, avx2, avx2_vnni, avx512, avx512_vnni. |
| `RECLANK_KERNEL` | auto | The NumPy path runner: c, numba, or numpy. |
| `RECLANK_NUMA` | on | Turn the NUMA placement off with 0. |
| `RECLANK_PROGRAM` | 1 | Run one step as one C program. |
| `RECLANK_CACHE` | ~/.cache/reclank/weights | The weight cache folder. |
| `RECLANK_FLASH` | 1 | The flash attention kernel of a prompt pass. |
| `RECLANK_MTP` | model default | Turn multi-token prediction on or off. |
| `RECLANK_API_KEY` | change-me | The Bearer key of the server. |
| `RECLANK_QUIET` | 0 | Do not print the load report. |
| `RECLANK_LIBS_DIR` | ~/.cache/reclank/libs | The folder of the built C and CUDA libraries. |

## Hardware

The GPU path needs NVIDIA. It has no AMD or Intel support. Ampere and
newer get the most testing. Pascal (a GTX 1070 or a P102) runs too, and
Maxwell runs in a mode with no dp4a (`docs/MAXWELL_PLAN.md`).

The build picks the CUDA toolkit that can build the arch of the card.
CUDA 13 stops at sm_75, and Pascal (sm_61) needs CUDA 12.
`RECLANK_GPU_CC=61` makes a newer GPU act as a Pascal card. Small cards
run the large MoE models with a split: some experts stay on the GPU, the
CPU computes the rest.

Two GPUs can run one model in one process. The first card holds the dense
weights, the cache, and the hot experts. The second card holds more
experts. Set `RECLANK_GPU_DEVICE` to the first card and
`RECLANK_EXPERT_GPUS` to the second. Gemma 4 26B, Nemotron 3.5 Bolt, and
Qwen3.6 use this. `docs/TWO_GPU.md` gives the commands and the results.

The CPU path needs a C compiler. It uses AVX-512 and VNNI when the CPU
has them, and AVX2 otherwise. The scalar tier runs everywhere.

NUMA is first-class. `reclank/runtime/numa.py` places memory with mbind
from libc, with no libnuma. A step can run in parts, one per node. The
weightkeeper holds one shared segment per node with huge pages.

## Weightkeeper

The weightkeeper loads the experts of a huge model once and serves many
server starts from them. It reads 289 GB of expert rows in about 100 s,
then a server starts in seconds.

```sh
python scripts/weightkeeper.py CKPT            # load and serve
sudo python scripts/weightkeeper.py CKPT --lock  # also lock 1 GB pages
python scripts/weightkeeper.py --status
python scripts/weightkeeper.py --stop
```

## Tests

The unit tests need no model files:

```sh
python -m pytest tests/unit -q
python -m pytest tests/unit -q -m "not gpu"   # skip the gpu marked checks
```

The gpu marker needs a free NVIDIA GPU and nvcc. The GPU checks skip
themselves when the GPU holds more than 1000 MiB.

`tests/golden/` holds the bit-exact dumps. Run
`tests/golden/run_golden.sh OUT` before and after a change, then compare
with `python tests/golden/golden.py compare OUT_BEFORE OUT_AFTER`.

`tests/quality/` holds the answer-quality suite. The `scripts/` folder
holds about 90 `check_*.py` scripts that compare one part against a
reference, and about 30 `bench_*.py` scripts.

## Documentation

| File | Subject |
|---|---|
| `docs/LEARNING.md` | A guided walk of the dense 12B path. |
| `docs/PROGRAMS.md` | The program and opcode system. |
| `docs/GPU_NOTES.md` | How the GPU path became fast, in gain order. |
| `docs/HANDOFF_QWEN38.md` | The state of the Qwen3.8-Flash-Next work. |
| `docs/DEEPSEEK_V41_PLAN.md` | DeepSeek V4.1 Flash from its 510 GB checkpoint. |
| `docs/DEEPSEEK_V4_PLAN.md` | DeepSeek V4 Flash Vision, the next step. |
| `docs/LFM25_24B_PLAN.md` | LFM2.5-24B-A2B, from its BF12 and Q4_0 GGUF files. |
| `docs/NEMOTRON_BOLT_PLAN.md`, `docs/NEMOTRON_LESSONS.md` | Nemotron 3.5 Bolt, and what that port taught. |
| `docs/GLM53_PLAN.md` | GLM-5.3-Flash in EXL3, with its DSpark drafter. |
| `docs/TWO_GPU.md` | Two GPUs in one process: the expert device. |
| `docs/MAXWELL_PLAN.md` | The Maxwell mode. |
| `docs/MINICPM_V47_PLAN.md` | A plan for MiniCPM-V 4.7 35B-A3B. |
| `docs/QWEN38_PLAN.md`, `docs/QWEN_PLAN.md`, `docs/QWEN_DENSE_QUANT.md` | The Qwen plans. |
| `docs/RQ6_MIX_PLAN.md`, `docs/RQ8_EXPERTS_PLAN.md` | Rotated expert forms on the GPU. |
| `docs/INT4_Q8_PLAN.md` | int4 weights with int8 activations. |
| `docs/EXL3_PLAN.md` | The ExLlamaV3 trellis format. |
| `docs/MTP_PLAN.md`, `docs/MTP_DRAFTER_QUANT.md` | Multi-token prediction. |
| `docs/MULTIMODAL_PLAN.md` | Image, audio, and video input. |
| `docs/MOE_PLAN.md` | The Gemma 4 26B-A4B support. |
| `docs/CPU_PLAN.md`, `docs/PERF_PLAN.md`, `docs/PREFILL_GEMM_PLAN.md` | Speed work. |
| `docs/SPLIT_PLAN.md` | One step across programs, NUMA nodes, and the GPU. |
| `docs/REFACTOR_PLAN.md` | The plan of this package layout. |
| `docs/TEST_PLAN.md` | The test framework. |
| `docs/WEIGHT_STRUCTURE.md` | Four negative results on weight structure. |
| `docs/how-llms-work.html` | The eight phases of inference, for a newcomer. |

## Program listings

`sexp/` holds s-expression listings of the compiled programs of one
model, for human reading. The runtime executes record arrays, never these
files. `python scripts/export_sexp.py --out sexp` makes them again.
`sexp/README.md` explains the notation.

## Limits

* The server answers one request at a time. The model shares one key and
  value cache.
* The runtime has no AMD or Intel GPU support. Pascal and Maxwell get
  less testing than NVIDIA Ampere and newer.
* The C path needs a C compiler. Without one, the model falls back to the
  Numba path or the NumPy path.
* The key and value cache uses buffers of a fixed size. A sliding layer
  keeps one window.
* The int8 mode quantizes each row with one scale. A smaller group gives
  a smaller error and a slower multiply.
* This is a research project. Expect rough edges.
