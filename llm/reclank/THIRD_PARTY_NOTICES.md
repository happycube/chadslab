# Third-party notices

Parts of reclank are ported or translated from the open-source projects below: model code, file
formats and kernels written from their source. Each project's copyright and license notice is kept
here, and each derived file says what it was ported from at its top. The licenses of model weights
are not code notices: they are the models' own (reclank/setup/catalog.py lists them).

## diffusers

- Source: https://github.com/huggingface/diffusers
- License: Apache License 2.0 (no NOTICE file)
- Copyright:

  - Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved. (src/diffusers/models/transformers/transformer_flux2.py)
  - Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved. (pipelines/flux2/pipeline_flux2_klein.py)
  - Copyright 2025 The Black Forest Labs Team and The HuggingFace Team. All rights reserved. (pipelines/flux2/image_processor.py)
  - Copyright 2026 The HuggingFace Team. All rights reserved. (models/autoencoders/autoencoder_kl_flux2.py, autoencoder_kl.py, vae.py, autoencoder_oobleck.py; models/embeddings.py; models/unets/unet_2d_condition.py; pipelines/stable_diffusion/pipeline_stable_diffusion.py)
  - Copyright 2026 Qwen-Image Team, The HuggingFace Team. All rights reserved. (models/transformers/transformer_qwenimage21.py, pipelines/qwenimage21/pipeline_qwenimage21.py)
  - Copyright 2026 The Qwen Team and The HuggingFace Team. All rights reserved. (models/autoencoders/autoencoder_kl_qwenimage21.py)
  - Copyright 2025 Zhejiang University Team and The HuggingFace Team. All rights reserved. (schedulers/scheduling_pndm.py)
  - Copyright 2025 Stanford University Team and The HuggingFace Team. All rights reserved. (schedulers/scheduling_ddim.py)
  - Copyright 2025 TSAIL Team and The HuggingFace Team. All rights reserved. (schedulers/scheduling_unipc_multistep.py)
  - Copyright 2025 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved. (schedulers/scheduling_flow_match_euler_discrete.py)
  - Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved. (models/transformers/transformer_wan.py, models/autoencoders/autoencoder_kl_wan.py, pipelines/wan/pipeline_wan.py)
  - Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved. (models/transformers/transformer_z_image.py, pipelines/z_image/pipeline_z_image.py)

- reclank files derived from it:

  - `reclank/models/flux2/dit.py`: transformer_flux2.py, embeddings.py
  - `reclank/models/flux2/pipeline.py`: pipeline_flux2_klein.py, flux2/image_processor.py, scheduling_flow_match_euler_discrete.py
  - `reclank/models/flux2/vae.py`: autoencoder_kl_flux2.py, autoencoder_kl.py, vae.py
  - `reclank/models/qwen_image/dit.py`: transformer_qwenimage21.py
  - `reclank/models/qwen_image/pipeline.py`: pipeline_qwenimage21.py, scheduling_flow_match_euler_discrete.py
  - `reclank/models/qwen_image/vae.py`: autoencoder_kl_qwenimage21.py
  - `reclank/models/sd/pipeline.py`: pipeline_stable_diffusion.py
  - `reclank/models/sd/scheduler.py`: scheduling_pndm.py, scheduling_ddim.py
  - `reclank/models/sd/unet.py`: unet_2d_condition.py
  - `reclank/models/wan/dit.py`: transformer_wan.py
  - `reclank/models/wan/pipeline.py`: pipeline_wan.py
  - `reclank/models/wan/scheduler.py`: scheduling_unipc_multistep.py
  - `reclank/models/wan/vae.py`: autoencoder_kl_wan.py
  - `reclank/models/zimage/dit.py`: transformer_z_image.py
  - `reclank/models/zimage/pipeline.py`: pipeline_z_image.py, scheduling_flow_match_euler_discrete.py
  - `reclank/models/t5/encoder.py`: pipeline_wan.py (_get_t5_prompt_embeds)
  - `reclank/models/t5/tokenizer.py`: pipeline_wan.py (prompt_clean)
  - `reclank/models/acestep/vae.py`: autoencoder_oobleck.py
  - `reclank/models/audio/dac.py`: autoencoder_oobleck.py
  - `reclank/models/seedvr2/dit.py`: embeddings.py (get_timestep_embedding)
  - `scripts/flux2_reference.py`: the loop of pipeline_flux2_klein.py
  - `scripts/qwen_image_reference.py`: the loop of pipeline_qwenimage21.py
  - `scripts/wan_reference.py`: the loop of pipeline_wan.py
  - `scripts/zimage_reference.py`: the loop of pipeline_z_image.py

## transformers

- Source: https://github.com/huggingface/transformers
- License: Apache License 2.0 (no NOTICE file)
- Copyright:

  - Copyright 2018- The Hugging Face team. All rights reserved. (LICENSE)
  - Copyright 2021 The OpenAI Team Authors and The HuggingFace Team. All rights reserved. (src/transformers/models/clip/modeling_clip.py)
  - Copyright 2021 The Open AI Team Authors and The HuggingFace Inc. team. (models/clip/tokenization_clip.py)
  - Copyright 2023 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team. (models/umt5/modeling_umt5.py)
  - Copyright 2018 T5 Authors and HuggingFace Inc. team. (models/t5/tokenization_t5.py)
  - Copyright 2021 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved. (models/hubert/modeling_hubert.py)
  - Copyright 2024 Descript and The HuggingFace Inc. team. All rights reserved. (models/dac/modeling_dac.py)
  - Copyright 2025 Boson AI and The HuggingFace Team. All rights reserved. (models/higgs_audio_v2_tokenizer/modeling_higgs_audio_v2_tokenizer.py)
  - Copyright 2022 The OpenAI Authors and The HuggingFace Inc. team. All rights reserved. (models/whisper/modeling_whisper.py)
  - Copyright 2022 The HuggingFace Inc. team. (models/whisper/feature_extraction_whisper.py)
  - Copyright 2024 The HuggingFace Inc. team. (models/whisper/generation_whisper.py)
  - Copyright 2022 The HuggingFace Inc. team. All rights reserved. (models/whisper/tokenization_whisper.py)
  - Copyright 2024 The HuggingFace Inc. team and Google DeepMind. (generation/logits_process.py)
  - Copyright 2023 The HuggingFace Inc. team and the librosa & torchaudio authors. (audio_utils.py)
  - Copyright 2023 The LAION-AI Team and The HuggingFace Team. All rights reserved. (models/clap/modeling_clap.py)
  - Copyright 2023 The HuggingFace Inc. team. All rights reserved. (models/clap/feature_extraction_clap.py)
  - Copyright 2022 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved. (models/wav2vec2_conformer, as MuQ carries it; transformers' modeling file has no header, its configuration_wav2vec2_conformer.py the same line)
  - Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team. (models/roberta/modeling_roberta.py)
  - Copyright 2019 Facebook AI Research and the HuggingFace Inc. team. (models/xlm_roberta/modeling_xlm_roberta.py)
  - Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved. (models/qwen3/modeling_qwen3.py)
  - Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved. (models/qwen3_5_moe/modeling_qwen3_5_moe.py, models/qwen3_5/modeling_qwen3_5.py, models/qwen3_vl/modeling_qwen3_vl.py)
  - Copyright 2025 The HuggingFace Team. All rights reserved. (models/lfm2_moe/modeling_lfm2_moe.py)
  - Copyright 2026 The Qwen Team and The HuggingFace Inc. team. All rights reserved. (models/qwen4_exp/modeling_qwen4_exp.py)
  - Copyright 2026 the HuggingFace Team. All rights reserved. (models/gemma4/modeling_gemma4.py, processing_gemma4.py, video_processing_gemma4.py; models/gemma4_unified/*; models/gemma4_assistant/modeling_gemma4_assistant.py; models/embedding_gemma2/modeling_embedding_gemma2.py; models/glm5_next/modeling_glm5_next.py; models/diffusion_gemma/modeling_diffusion_gemma.py, generation_diffusion_gemma.py)
  - Copyright 2026 Google LLC (models/gemma4/feature_extraction_gemma4.py)

- reclank files derived from it:

  - `reclank/models/sd/clip.py`: modeling_clip.py
  - `reclank/models/sd/tokenizer.py`: tokenization_clip.py
  - `reclank/models/t5/encoder.py`: modeling_umt5.py
  - `reclank/models/t5/tokenizer.py`: tokenization_t5.py
  - `reclank/models/audio/hubert.py`: modeling_hubert.py
  - `reclank/models/audio/dac.py`: modeling_dac.py
  - `reclank/models/higgs/codec.py`: modeling_higgs_audio_v2_tokenizer.py
  - `reclank/models/whisper/encoder.py`: modeling_whisper.py, feature_extraction_whisper.py, audio_utils.py
  - `reclank/models/whisper/decoder.py`: modeling_whisper.py, generation_whisper.py, tokenization_whisper.py, logits_process.py
  - `reclank/models/music/clap.py`: modeling_clap.py, feature_extraction_clap.py
  - `reclank/models/music/frontend.py`: audio_utils.py, feature_extraction_clap.py
  - `reclank/models/music/muq.py`: modeling_wav2vec2_conformer.py
  - `reclank/models/music/text.py`: modeling_roberta.py, modeling_xlm_roberta.py, modeling_clap.py
  - `reclank/models/qwen3/dense.py`: modeling_qwen3.py
  - `reclank/models/qwen3/decode.py`: modeling_qwen3.py
  - `reclank/models/qwen/qwen35.py`: modeling_qwen3_5_moe.py, modeling_qwen3_5.py, modeling_lfm2_moe.py
  - `reclank/models/qwen/qwen38.py`: modeling_qwen4_exp.py
  - `reclank/models/qwen/vision.py`: modeling_qwen3_vl.py, modeling_qwen3_5_moe.py, modeling_qwen4_exp.py
  - `reclank/models/gemma4/model.py`: modeling_gemma4.py, modeling_gemma4_unified.py
  - `reclank/models/gemma4/e4b.py`: modeling_gemma4.py
  - `reclank/models/gemma4/unified.py`: gemma4_unified (modeling, processing, video_processing)
  - `reclank/models/gemma4/encoders.py`: modeling_gemma4.py, feature_extraction_gemma4.py
  - `reclank/models/gemma4/assistant.py`: modeling_gemma4_assistant.py
  - `reclank/models/dgemma/model.py`, `runtime.py`: modeling_diffusion_gemma.py
  - `reclank/models/dgemma/sampler.py`: generation_diffusion_gemma.py
  - `reclank/encoders/runtime.py`: modeling_gemma4.py, feature_extraction_gemma4.py
  - `reclank/encoders/media.py`: processing_gemma4.py, video_processing_gemma4.py, modeling_qwen3_vl.py
  - `reclank/models/embedgemma2/model.py`: modeling_embedding_gemma2.py
  - `reclank/models/glm53/model.py`: modeling_glm5_next.py

## llama.cpp and ggml

- Source: https://github.com/ggml-org/llama.cpp (ggml: https://github.com/ggml-org/ggml)
- License: MIT License
- Copyright:

  - Copyright (c) 2023-2026 The ggml authors (LICENSE; reclank/backends/include/iq2_tables.h quotes the earlier 2023-2024 line)

- reclank files derived from it:

  - `reclank/backends/include/iq2_tables.h`: the IQ2_XXS tables of ggml/src/ggml-common.h
  - `reclank/formats/gguf.py`: the block formats (ggml-common.h, ggml-quants.c)
  - `reclank/formats/iq2.py`: dequantize_row_iq2_xxs
  - `reclank/formats/qwen38_orig.py`: conversion/qwen.py, _reorder_v_heads
  - `reclank/backends/cpu/csrc/kernels/kq/convert.c`: quantize_row_q8_0_ref, make_qx_quants, make_qkx2_quants, the K-quant quantizers
  - `reclank/backends/cpu/csrc/kernels/kq/dots.c`: the block formats
  - `reclank/backends/cpu/csrc/kernels/kq/formats.c`: the block formats
  - `reclank/backends/cpu/csrc/kernels/kq/iq2.c`: IQ2_XXS
  - `reclank/backends/cuda/csrc/kernels/quant/kquants.cu`: the block formats
  - `reclank/backends/cuda/csrc/kernels/head/head.cu`: dequantize_row_q6_K
  - `reclank/backends/cuda/csrc/kernels/nemotron/iq2.cuh`: IQ2_XXS
  - `reclank/models/qwen/qwen38.py`: src/models/qwen4exp.cpp (the MTP layer)
  - `reclank/models/nemotron/lilicorr.py`: src/models/dflash.cpp, common/speculative.cpp

## ExLlamaV3

- Source: https://github.com/turboderp-org/exllamav3
- License: MIT License
- Copyright:

  - Copyright (c) 2025 Turboderp

- reclank files derived from it:

  - `reclank/formats/exl3.py`
  - `reclank/formats/exl3_qwen4.py`
  - `reclank/formats/exl3_glm53.py`
  - `reclank/backends/cpu/csrc/kernels/kq/exl3.c`
  - `reclank/backends/cuda/csrc/kernels/quant/exl3.cu`

## DeepSeek V4.1 Flash reference code

- Source: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017 (inference/)
- License: MIT License
- Copyright:

  - Copyright (c) 2023 DeepSeek

- reclank files derived from it:

  - `reclank/models/deepseek41/model.py`: inference/model.py
  - `reclank/models/deepseek41/engram.py`: inference/engram.py
  - `reclank/models/deepseek41/vision.py`: inference/image_processor.py, inference/vision.py
  - `reclank/models/deepseek41/dspark.py`: inference/model.py (DSparkBlock)
  - `reclank/backends/cuda/csrc/kernels/dsv41/step.cu`: inference/kernel.py (the quantizers)
  - `scripts/dsv41_ref/kernel.py`: inference/kernel.py

## NVIDIA Nemotron 3.5 checkpoint code

- Source: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB, https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4
- License: Apache License 2.0 (the code files' own headers; the weights' license is the model's)
- Copyright:

  - Copyright 2024 The HuggingFace Inc. team. All rights reserved. / Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved. (modeling_nemotron_h.py)
  - Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved. (modeling_radio.py, image_processing_nemotron_h_omni.py, modeling_nemotron_h_omni.py)

- reclank files derived from it:

  - `reclank/models/nemotron/model.py`: modeling_nemotron_h.py
  - `reclank/models/nemotron/radio.py`: image_processing_nemotron_h_omni.py, modeling_radio.py, modeling_nemotron_h_omni.py

## vLLM

- Source: https://github.com/vllm-project/vllm (and the fork https://github.com/local-inference-lab/vllm)
- License: Apache License 2.0 (no NOTICE file)
- Copyright:

  - Copyright contributors to the vLLM project (SPDX-FileCopyrightText)

- reclank files derived from it:

  - `reclank/models/nemotron/model.py`: vllm/model_executor/models/nemotron_h_mtp.py
  - `reclank/formats/fp4local_st.py`: the fork's vllm/models/qwen4_exp/nvidia
  - `scripts/qfn_fp4l_reference.py`: the fork's vllm/models/qwen4_exp/nvidia

## FlashInfer (b12x)

- Source: https://github.com/local-inference-lab/flashinfer (upstream https://github.com/flashinfer-ai/flashinfer)
- License: Apache License 2.0
- Copyright:

  - Copyright (c) 2025 by the b12x authors. (flashinfer/experimental/b12x)
  - Copyright 2025-2026 NVIDIA; Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/) (NOTICE)

- reclank files derived from it:

  - `reclank/formats/fp4local_st.py`: scripts/convert_csf_blocks.py, flashinfer/experimental/b12x
  - `reclank/backends/cpu/csrc/kernels/kq/convert.c`: scripts/convert_csf_blocks.py (the CSF decode)
  - `reclank/backends/cpu/csrc/kernels/kq/nvx.c`: b12x quantize_block_fp4
  - `reclank/backends/cuda/csrc/kernels/moe/b12x.cu`: b12x's A4 prefill
  - `scripts/qfn_fp4l_reference.py`: scripts/convert_csf_blocks.py, b12x

Its NOTICE file, verbatim:

```
FlashInfer
Copyright 2025-2026 NVIDIA
Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)

-------------------------------------------------------------------------------------------------
Some of the code in this project are adapted from other open-source projects with different
licenses. This product also bundles some third-party components under other open source licenses.
See licenses/ for text of these licenses.

The persistent CuTe DSL KDA kernels are adapted from humanfia/kda-for-kda-release,
Copyright (c) 2026 KDA Team, under the MIT License (licenses/LICENSE.kda-for-kda).

The static PTX KDA kernels are adapted from NVlabs/kda (260927-kda-for-kda),
Copyright (c) 2026 KDA Team, under the MIT License (LICENSE.kda-for-kda.txt).

The TIRx KDA kernels are adapted from NVlabs/kda (260927-kda-for-kda),
Copyright (c) 2026 KDA Team, under the MIT License (licenses/LICENSE.kda-for-kda).
The split kernel also retains its original Apache-2.0 TIRx-author attribution.
Source: https://github.com/NVlabs/kda/tree/260927-kda-for-kda
```

## Speculators

- Source: https://github.com/vllm-project/speculators
- License: Apache License 2.0 (no NOTICE file)
- Copyright:

  - The repository states no copyright line (its LICENSE appendix is not filled in, its source files have no header); its pyproject.toml names the authors: Red Hat

- reclank files derived from it:

  - `reclank/models/glm53/dspark.py`: src/speculators/models/dspark
  - `scripts/glm53_dspark_reference.py`: src/speculators/models/dspark

## SGLang-Omni

- Source: https://github.com/sgl-project/sglang-omni
- License: Apache License 2.0 (no NOTICE file)
- Copyright:

  - Copyright 2025-2026 SGLang-Omni Team (LICENSE)

- reclank files derived from it:

  - `reclank/models/higgs/tts.py`: sglang_omni/models/higgs_tts
  - `scripts/higgs_reference.py`: higgs_tts sampler.py
  - `scripts/higgs_clone_reference.py`: higgs_tts text_tokenizer.py

## ACE-Step 1.5

- Source: https://github.com/ace-step/ACE-Step-1.5
- License: MIT License (the repository); the model files carry Apache License 2.0 headers
- Copyright:

  - Copyright (c) 2026 ACEStep (LICENSE)
  - Copyright 2025 The ACESTEO Team. All rights reserved. (acestep/models/turbo/modeling_acestep_v15_turbo.py)

- reclank files derived from it:

  - `reclank/models/acestep/dit.py`: modeling_acestep_v15_turbo.py
  - `reclank/models/acestep/pipeline.py`: inference.py, modeling_acestep_v15_turbo.py
  - `reclank/models/acestep/planner.py`: llm_inference.py, constrained_logits_processor.py
  - `reclank/models/acestep/vae.py`: core/generation/handler/vae_decode_chunks.py

## SeedVR

- Source: https://github.com/ByteDance-Seed/SeedVR
- License: Apache License 2.0 (no NOTICE file)
- Copyright:

  - Copyright (c) 2025 Bytedance Ltd. and/or its affiliates (source headers)
  - Copyright 2025 seed (LICENSE)
  - Copyright (c) 2023 HuggingFace Team / Copyright (c) 2025 ByteDance Ltd. and/or its affiliates. (models/video_vae_v3/modules/video_vae.py, modified by ByteDance from diffusers)

- reclank files derived from it:

  - `reclank/models/seedvr2/dit.py`: models/dit_v2, models/dit
  - `reclank/models/seedvr2/pipeline.py`: projects/inference_seedvr2_3b.py, projects/video_diffusion_sr/infer.py
  - `reclank/models/seedvr2/vae.py`: models/video_vae_v3
  - `reclank/models/seedvr2/vae_video.py`: models/video_vae_v3
  - `scripts/seedvr2_reference.py`: the inference code

## Depth Anything 3

- Source: https://github.com/ByteDance-Seed/Depth-Anything-3
- License: Apache License 2.0 (no NOTICE file)
- Copyright:

  - Copyright (c) 2025 ByteDance Ltd. and/or its affiliates (source headers)
  - Copyright 2025 The Depth Anything 3 Team (LICENSE)

- reclank files derived from it:

  - `reclank/models/depth/da3.py`: model/da3.py, api.py, utils/io/input_processor.py
  - `reclank/models/depth/dpt.py`: model/dpt.py
  - `scripts/stereo_reference.py`: src/depth_anything_3

## Video Depth Anything

- Source: https://github.com/DepthAnything/Video-Depth-Anything
- License: Apache License 2.0 (the code and the Small model; no NOTICE file)
- Copyright:

  - Copyright (2025) Bytedance Ltd. and/or its affiliates (source headers)
  - Copyright 2022 The HuggingFace Team. All rights reserved. (video_depth_anything/motion_module/attention.py)

- reclank files derived from it:

  - `reclank/models/depth/vda.py`: video_depth_anything/video_depth.py
  - `reclank/models/depth/dpt.py`: dpt.py, dpt_temporal.py, motion_module
  - `scripts/stereo_reference.py`: video_depth_anything
  - `tests/unit/test_stereo.py`: infer_video_depth's window loop

## MoGe

- Source: https://github.com/microsoft/MoGe
- License: MIT License (its vendored DINOv2: Apache License 2.0)
- Copyright:

  - Copyright (c) Microsoft Corporation.

- reclank files derived from it:

  - `reclank/models/depth/moge.py`: moge/model/v2.py
  - `scripts/stereo_reference.py`: moge

## DINOv2

- Source: https://github.com/facebookresearch/dinov2 (as vendored in Depth Anything 3, Video Depth Anything and MoGe)
- License: Apache License 2.0 (no NOTICE file)
- Copyright:

  - Copyright (c) Meta Platforms, Inc. and affiliates.

- reclank files derived from it:

  - `reclank/models/depth/common.py`: the ViT blocks
  - `scripts/stereo_reference.py`: the ViT

## Real-ESRGAN

- Source: https://github.com/xinntao/Real-ESRGAN
- License: BSD 3-Clause License
- Copyright:

  - Copyright (c) 2021, Xintao Wang

- reclank files derived from it:

  - `reclank/models/realesrgan/model.py`: realesrgan/archs/srvgg_arch.py, realesrgan/utils.py, inference_realesrgan.py
  - `scripts/realesrgan_reference.py`: the same

## BasicSR

- Source: https://github.com/XPixelGroup/BasicSR
- License: Apache License 2.0 (no NOTICE file)
- Copyright:

  - Copyright 2018-2022 BasicSR Authors

- reclank files derived from it:

  - `reclank/models/realesrgan/model.py`: basicsr/archs/rrdbnet_arch.py (RRDBNet)
  - `scripts/realesrgan_reference.py`: basicsr/archs/rrdbnet_arch.py

## torchaudio

- Source: https://github.com/pytorch/audio
- License: BSD 2-Clause License
- Copyright:

  - Copyright (c) 2017 Facebook Inc. (Soumith Chintala), All rights reserved.

- reclank files derived from it:

  - `reclank/models/audio/resample.py`: functional.resample (_get_sinc_resample_kernel)

## MuQ

- Source: https://github.com/tencent-ailab/MuQ
- License: MIT License (the code; the weights' license is the model's)
- Copyright:

  - Copyright (c) Tencent.

- reclank files derived from it:

  - `reclank/models/music/muq.py`: src/muq/muq, src/muq/muq_mulan
  - `reclank/models/music/text.py`: src/muq/muq_mulan/modules/transformer.py

# License texts

Each distinct license once. The MIT and BSD texts carry their copyright lines, so each project's own text is given.

## Practical-RIFE

- Source: https://github.com/hzwer/Practical-RIFE
- License: MIT License
- Copyright:

  - Copyright (c) 2021 hzwer

- reclank files derived from it:

  - `reclank/models/vfi/rife.py`: the v4.25 model's train_log/IFNet_HDv3.py, model/warplayer.py
  - `reclank/models/vfi/video.py`: inference_video.py (the scene-cut test), model/pytorch_msssim/__init__.py (ssim_matlab)
  - `reclank/models/vfi/ref.py`: model/warplayer.py (the grid)
  - `scripts/rife_reference.py`: the v4.25 model's train_log/IFNet_HDv3.py, model/warplayer.py, inference_video.py

## Pillow

- Source: https://github.com/python-pillow/Pillow
- License: MIT-CMU License
- Copyright:

  - Copyright © 1997-2011 by Secret Labs AB
  - Copyright © 1995-2011 by Fredrik Lundh and contributors
  - Copyright © 2010 by Jeffrey 'Alex' Clark and contributors

- reclank files derived from it:

  - `reclank/models/vfi/video.py`: src/libImaging/Resample.c (precompute_coeffs and normalize_coeffs_8bpc of the BOX filter: box_coeffs, probe_plan)
  - `reclank/backends/cuda/csrc/kernels/enc/encoders.cu`: src/libImaging/Resample.c (ImagingResampleHorizontal_8bpc, ImagingResampleVertical_8bpc: k_enc_resample_u8)

## Apache License 2.0

diffusers, transformers, the Nemotron checkpoint code, vLLM, FlashInfer (b12x), Speculators, SGLang-Omni, the ACE-Step 1.5 model files, SeedVR, Depth Anything 3, Video Depth Anything, DINOv2, BasicSR.

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

## MIT License (llama.cpp / ggml)

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## MIT License (ExLlamaV3)

```
MIT License

Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## MIT License (DeepSeek V4.1 Flash reference code)

```
MIT License

Copyright (c) 2023 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## MIT License (ACE-Step 1.5)

```
MIT License

Copyright (c) 2026 ACEStep

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## MIT License (MoGe)

```
MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
```

## MIT License (MuQ)

```
MIT License

Copyright (c) Tencent.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## MIT License (Practical-RIFE)

```
MIT License

Copyright (c) 2021 hzwer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## BSD 3-Clause License (Real-ESRGAN)

```
BSD 3-Clause License

Copyright (c) 2021, Xintao Wang
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## BSD 2-Clause License (torchaudio)

```
BSD 2-Clause License

Copyright (c) 2017 Facebook Inc. (Soumith Chintala), 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## MIT-CMU License (Pillow)

```
The Python Imaging Library (PIL) is

    Copyright © 1997-2011 by Secret Labs AB
    Copyright © 1995-2011 by Fredrik Lundh and contributors

Pillow is the friendly PIL fork. It is

    Copyright © 2010 by Jeffrey 'Alex' Clark and contributors

Like PIL, Pillow is licensed under the open source MIT-CMU License:

By obtaining, using, and/or copying this software and/or its associated
documentation, you agree that you have read, understood, and will comply
with the following terms and conditions:

Permission to use, copy, modify and distribute this software and its
documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appears in all copies, and that
both that copyright notice and this permission notice appear in supporting
documentation, and that the name of Secret Labs AB or the author not be
used in advertising or publicity pertaining to distribution of the software
without specific, written prior permission.

SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
```
