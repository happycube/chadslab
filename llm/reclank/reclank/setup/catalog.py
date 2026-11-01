"""The models of the setup report: one Model a checkpoint, with what it needs.

Each entry gives the Hub repo and files, the download size, the license, the
memory of each mode (Need: the GPU's and the host's GiB, or the host's
alone), the server that loads it and its flags. A Need gives its source (a
plan, a commit, the file) and est=True when the number is an estimate (the
file's size plus the cache and the buffers) and not a measure.

The download sizes are the Hub's (python -m reclank.setup --check-hub
compares them, and the licenses, with the Hub). To add a model, add a Model
to MODELS; nothing else lists them.

status:
    ready      the server runs it
    manual     the server runs it after a step that is not a download
               (a conversion); the guide names the step
    not-yet    reclank has no path for it yet; why says what is missing
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Need:
    ram: float                # host GiB
    vram: float = 0.0         # GPU GiB (0: the CPU mode)
    src: str = ""
    est: bool = False
    note: str = ""


@dataclass
class Model:
    id: str
    name: str
    kind: str                 # chat, embed, image, speech, music, video, asr, upscale, stereo, interpolate
    repo: str
    dir: str                  # the folder under the models folder
    gb: float                 # the download (1e9 bytes)
    license: str
    files: list | None = None  # names or patterns in the repo (None: all of it)
    revision: str | None = None  # a branch of the repo (EXL3 sizes)
    status: str = "ready"
    why: str = ""
    gpu: Need | None = None
    cpu: Need | None = None
    min_cc: float = 6.1       # Pascal; RECLANK_GPU_CC emulates the lower ones
    server: str = "llm"       # llm (python -m server), embed, generate
    args: str = ""            # the server's flags; {d} is the model's folder
    gpu_args: str = ""
    cpu_args: str = ""
    model_id: str = ""        # the "model" of a request to a server of several
    commercial: bool = True   # False: the license is research or non-commercial
    first: int = 0            # the order of a first model (1 first; 0 never)
    notes: list = field(default_factory=list)


# a diffusers folder of Stable Diffusion: the safetensors of each part, not
# the checkpoints of the other formats beside them
SD_FILES = ["model_index.json", "*.md", "feature_extractor/*",
            "scheduler/*", "tokenizer/*", "text_encoder/config.json", "text_encoder/model.safetensors",
            "unet/config.json", "unet/diffusion_pytorch_model.safetensors", "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors"]

KINDS = ("chat", "embed", "image", "speech", "music", "video", "asr", "upscale", "stereo", "interpolate")

MODELS = [
    # ---- chat (python -m server, port 8080)
    Model("gemma4-26b", "Gemma 4 26B-A4B (QAT Q4_K_XL)", "chat",
          "unsloth/gemma-4-26B-A4B-it-qat-GGUF", "gemma-4-26B-unsloth-UD-Q4_K_XL", 15.44, "Apache-2.0 (Gemma 4)",
          files=["gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf", "mmproj-BF16.gguf"],
          gpu=Need(16, 7, "docs/MAXWELL_PLAN.md 3.5 (an 8 GB card: dense, cache and hot experts; "
                   "the cold 9 GB of experts on the CPU); models/gemma4/engine.py SMALL_CARD", est=True),
          cpu=Need(17, 0, "the file (14.2 GB) and the cache", est=True),
          args="--model {d}/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf --mmproj {d}/mmproj-BF16.gguf",
          gpu_args="--backend cuda", cpu_args="--backend cpu", first=1,
          notes=["README Quick start: a good first model"]),
    Model("gemma4-12b", "Gemma 4 12B (QAT Q4_K_XL, MTP drafter)", "chat",
          "unsloth/gemma-4-12B-it-qat-GGUF", "gemma-4-12B-unsloth-UD-Q4_K_XL", 7.15, "Apache-2.0 (Gemma 4)",
          files=["gemma-4-12B-it-qat-UD-Q4_K_XL.gguf", "mmproj-BF16.gguf", "mtp-gemma-4-12B-it.gguf"],
          gpu=Need(9, 9, "the file (6.7 GB), the cache and the buffers", est=True),
          cpu=Need(9, 0, "the file (6.7 GB) and the cache", est=True),
          args="--model {d}/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf --mmproj {d}/mmproj-BF16.gguf",
          gpu_args="--backend cuda", cpu_args="--backend cpu", first=3,
          notes=["MTP: add --drafter {d}/mtp-gemma-4-12B-it.gguf"]),
    Model("gemma4-e4b", "Gemma 4 E4B (QAT Q4_0, MTP drafter)", "chat",
          "unsloth/gemma-4-E4B-it-qat-GGUF", "gemma-4-E4B-unsloth-UD-Q4_K_XL", 4.28, "Apache-2.0 (Gemma 4)",
          files=["gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf", "mtp-gemma-4-E4B-it.gguf"],
          gpu=Need(6, 6, "docs/SPLIT_PLAN.md (the 5 GB file fits an 8 GB card alone)", est=True),
          cpu=Need(6, 0, "models/gemma4/e4b.py (int4 mode: 2.2 GB packed; the per-layer table read from the "
                   "file's map)", est=True),
          args="--model {d}/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf",
          gpu_args="--backend cuda --drafter {d}/mtp-gemma-4-E4B-it.gguf", cpu_args="--backend cpu", first=4,
          notes=["on the GPU the drafter (MTP) is faster (docs/SPLIT_PLAN.md, phase 5)"]),
    Model("gemma4-e2b", "Gemma 4 E2B (QAT Q4_0, MTP drafter)", "chat",
          "unsloth/gemma-4-E2B-it-qat-GGUF", "gemma-4-E2B-unsloth-UD-Q4_K_XL", 2.68, "Apache-2.0 (Gemma 4)",
          files=["gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf", "mtp-gemma-4-E2B-it.gguf"],
          gpu=Need(4, 4, "the E4B's path (models/gemma4/e4b.py) with a 2.6 GB file", est=True),
          cpu=Need(4, 0, "the E4B's path with a 2.6 GB file", est=True),
          args="--model {d}/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf",
          gpu_args="--backend cuda --drafter {d}/mtp-gemma-4-E2B-it.gguf", cpu_args="--backend cpu", first=5,
          notes=["no local check of the E2B file is recorded in docs/"]),
    Model("qwen36-35b", "Qwen3.6-35B-A3B (UD-Q4_K_M)", "chat",
          "unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-GGUF", 23.04, "Apache-2.0",
          files=["Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", "mmproj-BF16.gguf"],
          gpu=Need(26, 7, "docs/QWEN_PLAN.md (dense 2.0 GB, experts 19.6 GB: the hot ones on the GPU, "
                   "the rest on the CPU)", est=True),
          cpu=Need(26, 0, "the file (22.1 GB) and the cache", est=True),
          args="--model {d}/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --mmproj {d}/mmproj-BF16.gguf",
          gpu_args="--backend cuda", cpu_args="--backend cpu"),
    Model("lfm25-24b", "LFM2.5-24B-A2B (QAD Q4_0)", "chat",
          "LiquidAI/LFM2.5-24B-A2B-GGUF", "LFM2.5-24B-A2B-GGUF", 13.58, "LFM Open License v1.0",
          files=["LFM2.5-24B-A2B-QAD-Q4_0.gguf"],
          gpu=Need(16, 7, "the file (13.6 GB): on a 24 GB card all 64 experts are on the GPU (186 tok/s); a "
                   "smaller card keeps the cold ones on the CPU", est=True),
          cpu=Need(16, 0, "the file and the cache; 63 ms a token on 24 threads of the Xeon", est=True),
          args="--model {d}/LFM2.5-24B-A2B-QAD-Q4_0.gguf --ctx 131072",
          gpu_args="--backend cuda", cpu_args="--backend cpu", first=2,
          notes=["README Quick start: a good first model",
                 "the license limits commercial use above a revenue threshold: read it"]),
    Model("nemotron-bolt-74b", "Nemotron 3.5 Bolt 74B-A9B (24 GB form)", "chat",
          "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB", "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB", 24.12,
          "OpenMDW-1.1",
          gpu=Need(26, 6, "docs/NEMOTRON_BOLT_PLAN.md 3 (5.4 GB at 32K with every expert on the CPU; a 24 GB "
                   "card holds all 17 GB of experts; the RAM an estimate)", est=True,
                   note="--ctx 32768 on a small card"),
          cpu=Need(26, 0, "docs/NEMOTRON_BOLT_PLAN.md (the 24 GB of weights; AVX2 on 8 cores 274 ms a token)",
                   est=True),
          args="--model {d}", gpu_args="--backend cuda", cpu_args="--backend cpu",
          notes=["the LiLiCorr drafter (nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr, 1 GB) speeds "
                 "up the GPU: --drafter DIR"]),
    Model("nemotron-super", "Nemotron 3.5 Super VL 120B-A12B (NVFP4)", "chat",
          "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4", "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4",
          75.06, "OpenMDW-1.1",
          gpu=Need(72, 22, "docs/NEMOTRON_SUPER_PLAN.md 5 (7.2 GB dense, 12-13 GB hot experts on 24 GB; "
                   "the host a copy of the 66 GB of experts a NUMA node)", est=True),
          args="--model {d} --ctx 131072 --mtp 3", gpu_args="--backend cuda",
          notes=["a weightkeeper (scripts/weightkeeper.py) keeps the experts between restarts"]),
    Model("qwen38-27b", "Qwen3.8-27B (EXL3 5.0 bpw)", "chat",
          "turboderp/Qwen3.8-27B-exl3", "Qwen3.8-27B-exl3-SC_5.00bpw_H6_V6", 19.58, "Apache-2.0",
          revision="SC_5.00bpw_H6_V6",
          gpu=Need(24, 22, "every matrix on the GPU (19.6 GB) and the cache", est=True),
          args="--model {d}", gpu_args="--backend cuda"),
    Model("qwen38-fn-exl3", "Qwen3.8-Flash-Next 125B (EXL3 5.05 bpw)", "chat",
          "turboderp/Qwen3.8-Flash-Next-exl3", "Qwen3.8-Flash-Next-exl3-5.05bpw_h6_ng6", 123.25,
          "Qwen Community 1.0", revision="5.05bpw_h6_ng6",
          gpu=Need(130, 22, "the experts in RAM (a copy a NUMA node when they fit), the dense part and the hot "
                   "experts on a 24 GB card", est=True),
          args="--model {d} --ctx 163840", gpu_args="--backend cuda",
          notes=["images need --mmproj with the original Qwen/Qwen3.8-Flash-Next folder"]),
    Model("qwen38-fn-fp4l", "Qwen3.8-Flash-Next-FP4-Local", "chat",
          "nvidia/Qwen3.8-Flash-Next-FP4-Local", "Qwen3.8-Flash-Next-FP4-Local", 102.26,
          "NVIDIA Open Model License",
          gpu=Need(110, 22, "docs/FP4LOCAL_PLAN.md 4 (a node's 68 GB of experts, the 30.7 GB n-gram table, "
                   "9 GB; 176 GB with a copy on each of two nodes)", est=True),
          args="--model {d} --ctx 262144 --mmproj {d} --mmproj-gpu lend", gpu_args="--backend cuda",
          notes=["start the weightkeeper first: python scripts/weightkeeper.py {d}"]),
    Model("deepseek41", "DeepSeek V4.1 Flash", "chat",
          "deepseek-ai/DeepSeek-V4.1-Flash-1017", "DeepSeek-V4.1-Flash-1017", 510.31, "MIT",
          gpu=Need(300, 22, "docs/DEEPSEEK_V41_PLAN.md 4.1 (experts 288.8 GB in RAM; the 203 GB engram "
                   "tables read in place from the disk; dense 7.5 + head 2.6 + hot experts on 24 GB)", est=True),
          args="--model {d} --mmproj {d}", gpu_args="--backend cuda",
          notes=["start the weightkeeper first: python scripts/weightkeeper.py {d}",
                 "keep the checkpoint on an NVMe disk: the engram rows are read from it each token"]),
    Model("glm53-flash", "GLM-5.3-Flash (EXL3 4.05 bpw)", "chat",
          "turboderp/GLM-5.3-Flash-exl3", "GLM-5.3-Flash-exl3-4.05bpw", 165.15, "MIT", revision="4.05bpw",
          status="not-yet",
          why="check scripts only: no server engine (README Models)"),
    Model("minicpm-v47", "MiniCPM-V 4.7 35B-A3B", "chat",
          "openbmb/MiniCPM-V-4.7-35B-A3B", "MiniCPM-V-4.7-35B-A3B", 70.56, "Apache-2.0", status="not-yet",
          why="a plan (docs/MINICPM_V47_PLAN.md): no engine yet"),
    Model("nemotron-bolt-e4b", "Nemotron 3.5 Bolt E4B (BF16)", "chat",
          "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16", "NVIDIA-Nemotron-3.5-Bolt-E4B-BF16", 16.03, "OpenMDW-1.1",
          files=["*.json", "*.safetensors", "*.jinja", "*.md"],
          status="not-yet",
          why="a Gemma 4 E4B fine-tune in a bfloat16 safetensors folder: the gemma4 family loads GGUF files "
              "only, and its chat template is NVIDIA's (thinking budget, conversational mode)",
          gpu=Need(18, 16, "the 16 GB of bfloat16 weights (7.9 GB of layers, 5.6 GB per-layer table, 1.3 GB "
                   "embedding, 0.9 GB towers)", est=True),
          notes=["NVIDIA's 4 GB form, nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB (one Q4_K_M GGUF, 6.2 GB, "
                 "general.architecture gemma4), may load through the Gemma E4B path: not yet checked"]),

    # ---- embeddings (python -m server.embed, port 8082)
    Model("embeddinggemma2", "EmbeddingGemma 2 (text, image, audio)", "embed",
          "google/embeddinggemma-2", "embeddinggemma-2", 1.53, "Apache-2.0",
          gpu=Need(4, 3, "the 1.5 GB of weights in BF12 and the towers", est=True),
          cpu=Need(4, 0, "the 1.5 GB of weights", est=True),
          server="embed", args="--model {d}", gpu_args="--gpu"),

    # ---- image, speech, music, video (python -m server.generate, port 8084)
    Model("flux2-klein", "FLUX.2 [klein] 4B", "image",
          "black-forest-labs/FLUX.2-klein-4B", "flux2-klein-4b", 15.98, "Apache-2.0",
          files=["*.json", "*.md", "scheduler/*", "text_encoder/*", "tokenizer/*", "transformer/*", "vae/*"],
          gpu=Need(20, 16, "the card's peak measured at 1024 x 1024: 16.0 GB (the text encoder and the DiT on it "
                   "at the prompt; docs/GENERATION_PLAN.md, FLUX.2 reference images); the host's 20 GB an estimate",
                   est=True),
          cpu=Need(40, 0, "the DiT and the encoder in float32 (--cpu)", est=True),
          server="generate", args="--flux2 {d}", cpu_args="--cpu", model_id="flux2-klein",
          notes=["editing: POST /v1/images/edits (multipart: 1 to 4 reference images, image or image[]); the "
                 "GPU's peak 16.9 GB with one reference at 1024 x 1024, 17.9 GB with two"]),
    Model("sd-1.5", "Stable Diffusion 1.5", "image",
          "sd-legacy/stable-diffusion-v1-5", "sd-1.5", 4.27, "CreativeML OpenRAIL-M", files=SD_FILES,
          gpu=Need(6, 5, "commit e7e3831 (a GPU run's host peak 5.8 GB, 2.5 GB with --f20; the card's peak is "
                   "not recorded)", est=True),
          cpu=Need(11, 0, "commit e7e3831 (the CPU's peak 10.2 GB, 3.6 GB with --f20)"),
          server="generate", args="--sd sd-1.5={d}", cpu_args="--cpu", model_id="sd-1.5",
          notes=["OpenRAIL: use restrictions (no regional ones)",
                 "--f20 keeps the UNet in F20: the CPU's peak 3.6 GB, a GPU run's host 2.5 GB (the image within "
                 "about 0.1 level of 255)"]),
    Model("sd-2.0-base", "Stable Diffusion 2.0 (512)", "image",
          "sd2-community/stable-diffusion-2-base", "sd-2.0-base", 5.16, "CreativeML OpenRAIL++-M", files=SD_FILES,
          gpu=Need(6, 5, "as SD 1.5 (commit e7e3831)", est=True),
          cpu=Need(11, 0, "as SD 1.5 (commit e7e3831)", est=True),
          server="generate", args="--sd sd-2.0-base={d}", cpu_args="--cpu", model_id="sd-2.0-base"),
    Model("sd-2.0-768", "Stable Diffusion 2.0 (768, v-prediction)", "image",
          "sd2-community/stable-diffusion-2", "sd-2.0-768", 5.16, "CreativeML OpenRAIL++-M", files=SD_FILES,
          gpu=Need(6, 6, "as SD 1.5 with a larger UNet input (commit e7e3831)", est=True),
          cpu=Need(12, 0, "as SD 1.5 (commit e7e3831)", est=True),
          server="generate", args="--sd sd-2.0-768={d}", cpu_args="--cpu", model_id="sd-2.0-768"),
    Model("qwen-image-2.1", "Qwen-Image-2.1 (RGBA)", "image",
          "Qwen/Qwen-Image-2.1", "qwen-image-2.1", 33.13, "Qwen Research License", commercial=False,
          gpu=Need(32, 15, "commit 43b877c (the GPU 14.6 GB at 1024 x 1024, 16.4 GB at 2048; the text encoder "
                   "on the CPU, 12 GB: its matrices in BF12)"),
          cpu=Need(48, 0, "the 14 GB DiT in float32 (--cpu) and the 12 GB text encoder (BF12)", est=True),
          server="generate", args="--qwen-image {d}", cpu_args="--cpu", model_id="qwen-image-2.1",
          notes=["editing: POST /v1/images/edits (multipart: image or image[], up to 10 reference images, and "
                 "prompt); the GPU's peak 14.4 GiB with one 1024 x 1024 reference, 14.7 with two, 17.7 with "
                 "six (the DiT's cache of their keys and values in F20 in host memory: 0.63 MB a token)",
                 "an edit of 40 steps at 1024 x 1024 takes about 4 minutes on a 3090 (the text encoder on the CPU "
                 "reads the references' tokens: 31 s a reference on 12 threads)"]),
    Model("z-image-turbo", "Z-Image Turbo 6B", "image",
          "Tongyi-MAI/Z-Image-Turbo", "Z-Image-Turbo", 32.85, "Apache-2.0",
          files=["model_index.json", "*.md", "scheduler/*", "tokenizer/*", "text_encoder/*", "transformer/*",
                 "vae/*"],
          gpu=Need(32, 16, "the GPU 15.6 GB at 1024 x 1024, 16.5 GB at 1536 (nvidia-smi); the host's RSS 30.4 GB "
                   "(the DiT's 15.3 GB of F20 kept for its return, the text encoder on the CPU, 8 GB)"),
          cpu=Need(72, 0, "a 1024 x 1024 image in float32 (--cpu): 69.4 GB of RSS (the checkpoint's 24.1 GB "
                   "mapped, its packed copy, the text encoder)"),
          server="generate", args="--z-image {d}", cpu_args="--cpu", model_id="z-image-turbo",
          notes=["at most 1536 x 1536 of area on a 24 GB GPU: the VAE's buffers of 2048 x 2048 need more",
                 "a 1024 x 1024 image of 9 steps takes about 67 s on a 3090 (5.7 s a step)"]),
    Model("higgs-tts-3", "Higgs TTS 3 4B", "speech",
          "bosonai/higgs-tts-3-4b", "higgs-tts-3-4b", 9.32, "Boson Higgs research license", commercial=False,
          gpu=Need(12, 10, "docs/GENERATION_PLAN.md 1.1 (about 9 GB in bfloat16)", est=True),
          cpu=Need(20, 0, "the 9 GB of bfloat16 weights in float32 (--cpu)", est=True),
          server="generate", args="--higgs {d}", cpu_args="--cpu", model_id="higgs-tts-3",
          notes=["cloning a voice needs its speaker's consent"]),
    Model("ace-step-1.5", "ACE-Step 1.5 turbo + 5 Hz planner LM", "music",
          "ACE-Step/Ace-Step1.5", "ace-step-1.5", 10.10, "MIT",
          gpu=Need(14, 12, "the 10 GB of weights (DiT, planner LM 1.7B, Qwen3-Embedding 0.6B, VAE)", est=True),
          cpu=Need(24, 0, "the weights in float32 (--cpu)", est=True),
          server="generate", args="--acestep {d}", cpu_args="--cpu", model_id="ace-step-1.5"),
    Model("wan2.2-ti2v-5b", "Wan 2.2 TI2V-5B (text to video)", "video",
          "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "Wan2.2-TI2V-5B-Diffusers", 34.20, "Apache-2.0",
          gpu=Need(24, 14, "a 832 x 480 clip of 17 frames: 13.7 GiB at the peak (nvidia-smi)"),
          cpu=Need(48, 0, "docs/GENERATION_PLAN.md 8 (the DiT 19.7 GB in float32, umT5 10.3 GB, the VAE's "
                   "17.6 GB of RSS for 17 frames)", est=True),
          server="generate", args="--wan {d}", cpu_args="--cpu", model_id="wan2.2-ti2v-5b",
          notes=["720p (1280 x 704, 121 frames) needs 23 GB of the GPU: 16.9 GB in the steps, 22.6 GB in the VAE "
                 "(docs/GENERATION_PLAN.md 8)",
                 "a clip of 17 frames takes about 4 min on a 3090; 720p about 80 min"]),
    Model("wan2.2-t2v-a14b", "Wan 2.2 T2V-A14B (text to video, two 14B experts)", "video",
          "Wan-AI/Wan2.2-T2V-A14B-Diffusers", "Wan2.2-T2V-A14B-Diffusers", 126.0, "Apache-2.0",
          gpu=Need(84, 19, "measured: the server, 1280 x 720 x 5: 18.0 GiB of the card at the peak (the VAE), "
                   "832 x 480 x 5: 12.4 GiB; 80 GiB of host at the peak (VmHWM: both experts' F20, 71.4 GB, "
                   "in the process or mapped from a weightkeeper)"),
          server="generate", args="--wan {d}", model_id="wan2.2-t2v-a14b",
          notes=["GPU only: the experts' blocks are streamed to the card from host memory (F20, 12 GB/s)",
                 "a weightkeeper holds both experts (python scripts/weightkeeper.py {d} --name wan-a14b: "
                 "73 GB, ~15 min from an HDD once); the server then starts in seconds, else it loads them "
                 "itself (~8-20 min)",
                 "a 40-step clip: 832 x 480 x 17 about 31 min, x 81 about 3.2 h; 1280 x 720 x 33 about 3 h on a 3090"]),
    Model("whisper-large-v3", "Whisper large-v3 (speech to text)", "asr",
          "openai/whisper-large-v3", "whisper-large-v3", 3.09, "Apache-2.0",
          files=["*.json", "*.txt", "model.safetensors"],
          gpu=Need(9, 5, "measured: the server after 5 min of speech, 4980 MiB of the card (nvidia-smi), "
                   "8.5 GiB of host at the peak (VmHWM)"),
          cpu=Need(16, 0, "measured: the --cpu server after 5 min of speech, 15.6 GiB at the peak (VmHWM)"),
          server="generate", args="--whisper {d}", cpu_args="--cpu", model_id="whisper-large-v3",
          notes=["POST /v1/audio/transcriptions and /v1/audio/translations (OpenAI's, multipart); "
                 "audio of any length (transformers' long-form)",
                 "5 min of Slovak talk: 13 s on a 3090, about 4.5 min on 12 CPU threads"]),

    # ---- 2D to stereo 3D (python -m server.generate, docs/STEREO_PLAN.md)
    Model("da3mono", "Depth Anything 3 Mono Large (photo to stereo)", "stereo",
          "depth-anything/DA3MONO-LARGE", "DA3MONO-LARGE", 1.34, "Apache-2.0",
          files=["*.json", "*.md", "model.safetensors"],
          gpu=Need(3, 3, "measured: the server, 2.16 GB of the card (RESIDENCY's load) and the warp's 0.1-0.4 GB; "
                   "host: the 1.34 GB of weights and 0.56 GiB of a 504 program's buffers", est=True),
          cpu=Need(3, 0, "the 1.34 GB of weights and 0.56 GiB of a 504 program's buffers", est=True),
          server="generate", args="--stereo-da3 {d}", cpu_args="--cpu", model_id="da3mono",
          notes=["POST /v1/images/stereo: a photo to side-by-side (_LR), half SBS, over-under or anaglyph",
                 "depth 0.12 s at 504 on a 3090; a photo to a stereo JPEG in about 1-3 s"]),
    Model("moge-2", "MoGe-2 ViT-L normal (photo to stereo, metric depth)", "stereo",
          "Ruicheng/moge-2-vitl-normal", "moge-2-vitl-normal", 1.32, "MIT",
          status="manual", why="model.pt is converted once with torch: scripts/pt_to_safetensors.py "
                               "DIR/model.pt (model.safetensors and model.json beside it)",
          gpu=Need(4, 4, "measured: the server, 2.95 GB of the card (RESIDENCY's load) and the warp's 0.1-0.4 GB; "
                   "host: the 1.32 GB of weights and 1.34 GiB of a 3600-token program's buffers", est=True),
          cpu=Need(4, 0, "the 1.32 GB of weights and 1.34 GiB of a 3600-token program's buffers", est=True),
          server="generate", args="--stereo-moge {d}", cpu_args="--cpu", model_id="moge-2",
          notes=["POST /v1/images/stereo (the default model); \"mode\": \"metric\" sets the parallax from the "
                 "depth in metres and the focal (a baseline in mm)",
                 "depth 0.94 s on a 3090 (about 70 s on 10 CPU threads)"]),
    Model("vda-small", "Video Depth Anything Small (video to stereo)", "stereo",
          "depth-anything/Video-Depth-Anything-Small", "Video-Depth-Anything-Small", 0.12, "Apache-2.0",
          status="manual", why="the .pth is converted once with torch: scripts/pt_to_safetensors.py "
                               "DIR/video_depth_anything_vits.pth (its .safetensors beside it)",
          gpu=Need(4, 4, "measured: the server, 3.41 GB of the card for a 32-frame window of 518 x 686 (RESIDENCY's "
                   "load), the preprocessing's 0.06 GB, the warp's 0.13 GB; host: 2.82 GiB of the window's "
                   "buffers", est=True),
          cpu=Need(4, 0, "2.82 GiB of a 32-frame window's buffers", est=True),
          server="generate", args="--stereo-vda {d}", cpu_args="--cpu", model_id="vda-small",
          notes=["POST /v1/videos/stereo (multipart): a video file to a stereo MP4 (HEVC by NVENC, the st3d box, "
                 "the audio kept), a job of /v1/videos",
                 "1440 x 1080 at about 4-5 frames a second on a 3090",
                 "VDA's Base and Large are CC-BY-NC-4.0: not supported"]),
    # ---- video frame interpolation (python -m server.generate, docs/VFI_PLAN.md)
    Model("rife-4.25", "Practical-RIFE 4.25 (video frame interpolation: 2x, 4x, any rate)", "interpolate",
          "hzwer/Practical-RIFE", "Practical-RIFE", 0.025, "MIT",
          status="manual", why="the weights are a zip on Google Drive (the README's v4.25 link), not a Hugging "
                               "Face download: unzip it into DIR, then convert its pickle once with torch "
                               "(weights_only): python scripts/pt_to_safetensors.py DIR/4.25/flownet.pkl",
          gpu=Need(6, 4, "measured: RESIDENCY's loads of the Head and IFNet programs (two Head programs and one IFNet a "
                   "size) 0.7 GB at 832 x 480, 1.7 GB at 1440 x 1080, 3.9 GB at 3328 x 1920 (scale 0.5); the card's peak "
                   "2.8 GiB for a server doing 1440 x 1080; host: the programs' buffers, 1.9 GB at 1080p, 4.4 GB at 4K"),
          cpu=Need(4, 0, "the programs' buffers: 0.9 GB at 720p, 1.9 GB at 1080p", est=True),
          server="generate", args="--rife {d}", cpu_args="--cpu", model_id="rife-4.25",
          notes=["POST /v1/videos/interpolate (multipart: a video file, or video_id of a completed job): x2 (default), "
                 "x4, or a target rate (fps); scene cuts not interpolated across; the audio copied",
                 "on a 3090: an output frame 40 ms at 720p, 81 ms at 1080p, 191 ms at 4K (scale 0.5) and each input frame's "
                 "Head 3-22 ms; a 1440 x 1080 clip at 29.97 doubled at about 15 frames a second end to end (NVENC)"]),
    # ---- the models that are not a plain download
    Model("seedvr2-3b", "SeedVR2-3B (one-step restoration and upscaling of images and video)", "upscale",
          "ByteDance-Seed/SeedVR2-3B", "SeedVR2-3B", 14.6, "Apache-2.0",
          files=["seedvr2_ema_3b.pth", "ema_vae.pth", "pos_emb.pt", "neg_emb.pt", "README.md"],
          status="manual", why="the .pth checkpoints are converted once to safetensors with torch "
                               "(weights_only): python scripts/pt_to_safetensors.py DIR/seedvr2_ema_3b.pth DIR/ema_vae.pth DIR/pos_emb.pt (DIR: the download)",
          gpu=Need(40, 22, "measured (nvidia-smi, the card's peak): an image 1280 x 800 9.3 GiB, 2352 x 1760 "
                   "12.1 GiB; a video 832 x 480 x 17 14.8 GiB (the server), 1280 x 704 x 21 21.7 GiB; the host "
                   "keeps the DiT's F20 (8.5 GB) and the video VAE's causal caches (23 GB at 720p)", est=True),
          cpu=Need(60, 0, "the DiT in float32 (13.6 GB, mapped, and its packed copy) and the video VAE's caches",
                   est=True),
          server="generate", args="--seedvr2 {d}", cpu_args="--cpu", model_id="seedvr2-3b",
          notes=["POST /v1/images/upscale (x2 by default, at most 2048 x 2048 of output) and POST "
                 "/v1/videos/upscale jobs (at most 1280 x 720 and 121 frames on a 24 GB GPU)",
                 "on a 3090: a 1280 x 800 image about 14 s, 2352 x 1760 about 40 s, 832 x 480 x 17 about 58 s, "
                 "1280 x 704 x 21 about 2 min"]),
    Model("seedvr2-7b", "SeedVR2-7B (one-step restoration and upscaling, the larger)", "upscale",
          "ByteDance-Seed/SeedVR2-7B", "SeedVR2-7B", 33.97, "Apache-2.0",
          files=["seedvr2_ema_7b.pth", "ema_vae.pth", "README.md"],
          status="manual", why="the .pth checkpoints are converted once to safetensors with torch (weights_only), "
                               "with the 3B's text embedding (the 7B's repository has none): python "
                               "scripts/pt_to_safetensors.py DIR/seedvr2_ema_7b.pth DIR/ema_vae.pth "
                               "3B_DIR/pos_emb.pt --out DIR",
          gpu=Need(24, 22, "measured: the DiT's blocks streamed (F20, 20.6 GB of host); the card's peak 4.1 GiB "
                   "at 1184 x 880, 12.1 GiB at 2352 x 1760 (the VAE), 21.7 GiB at 1280 x 704 x 21 (the VAE's "
                   "caches); 20.6-23.1 GiB of host at the peak (VmHWM)"),
          cpu=Need(72, 0, "the DiT in float32 (33 GB, mapped, and its packed copy)", est=True),
          server="generate", args="--seedvr2-7b {d}", cpu_args="--cpu", model_id="seedvr2-7b",
          notes=["the same routes as seedvr2-3b; one SeedVR2 model loaded at a time (a request of the other "
                 "loads it: the 7B about 4 min from an HDD)",
                 "on a 3090: a 1184 x 880 image about 10 s, 2352 x 1760 about 42 s, 1280 x 704 x 21 about 2 min"]),
    Model("music-embed", "MuQ-MuLan + CLAP (music embeddings)", "embed",
          "OpenMuQ/MuQ-MuLan-large", "music", 2.65, "CC-BY-NC-4.0 (MuQ-MuLan)", commercial=False,
          status="manual", why="the weights are converted with torch: scripts/convert_music.py "
                               "(docs/MUSIC_EMBED_PLAN.md), from MuQ-MuLan and the original LAION CLAP checkpoint",
          gpu=Need(8, 4, "the 3.4 GB of converted weights", est=True),
          cpu=Need(8, 0, "the 3.4 GB of converted weights", est=True),
          server="embed", args="--music {d}", cpu_args="--music-device cpu", model_id="music"),
]


def by_id(mid):
    for m in MODELS:
        if m.id == mid:
            return m
    return None
