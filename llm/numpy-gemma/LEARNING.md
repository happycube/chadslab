# Learn numpy-gemma

This guide follows the existing runtime. It does not add a second model or change the fast path.

## Choose a model path

Start with the dense Gemma 4 12B path in the [README](README.md). It has a direct decoder-layer path and is the clearest route through the code.

The E4B model adds per-layer embeddings. The 26B model adds a mixture-of-experts block. The [overview page](how-llms-work.html) uses 26B and its 30-layer architecture. Do not use its layer count or dimensions as facts about 12B.

## Follow one token

1. Read [config.py](np_gemma/config.py). It turns model settings into a plan for each layer.
2. Start at `Model.forward` in [model.py](np_gemma/model.py). Follow the embedding lookup, layer loop, and final norm.
3. Follow `_decoder_layer`. Track the residual value, its four norms, the attention result, and the gated MLP result. The hook names show the intermediate values.
4. Follow `_attention`. Note the query, key, and value shapes, RoPE, causal mask, sliding window, and output projection.
5. Read `KVCache.write`, `KVCache.read`, and `Model.prefill`. Compare prompt prefill with one-token decode. They use the same model layers, but different token counts and cache access patterns.

Keep a small shape table as you read. Record token count, hidden width, query heads, key/value heads, and head width. Use the layer plan. Do not assume all layers share the same attention layout.

## Separate math from kernels

Read [ops.py](np_gemma/ops.py) alongside the model. It contains reference math and dispatch for available kernels. For example, compare `linear_int4_numpy` with `linear_int4`.

Then follow the C binding in [cops.py](np_gemma/cops.py) into [bf16_linear.c](np_gemma/csrc/bf16_linear.c). The NumPy expressions explain the operation. The C kernels reduce memory traffic and call overhead. Keep the optimized path as the default.

Kernel selection settings are read when `ops.py` imports. Set them before the process starts. The README lists the supported settings and the cost of each model mode.

## Check each step

- Run [check_gelu.py](scripts/check_gelu.py) to compare the C GELU kernel with a float64 reference. It does not load model weights.
- Run [check_cache.py](scripts/check_cache.py) with a local config and weight file. It compares batch execution with incremental decode.
- Run [check_trace.py](scripts/check_trace.py) with a captured Hugging Face trace. It compares named intermediate tensors.
- Run [check_tokenizer.py](scripts/check_tokenizer.py) with the model snapshot. It compares tokenization with the reference tokenizer.

Use the setup and reference commands in the README. A full trace needs model files and a captured reference trace, so start with the local kernel check if those files are not ready.

## Measure performance

Measure prefill and decode separately. Use the same GGUF file, machine, thread settings, prompt length, and generation length for both runtimes. The README records the reference setup and example results.

For a comparison, run `llama-bench -m FILE -p 512 -n 128 -r 2` and [bench_gguf_models.py](scripts/bench_gguf_models.py) with `--prompt 512 --gen 128 --reps 2`. Repeat paired runs when the machine load varies.

Keep each phase at or above 60 percent of `llama.cpp` throughput on the reference machine. Also compare with the project baseline; the 60 percent floor does not make a regression harmless. The recorded 12B run is about 68 percent on prefill and 74 percent on decode. Other models and machine loads give different ratios.

Change one operation at a time. First check its numerical result. Then check the model trace or cache behavior that uses it. Run the matched benchmark last. Do not change kernel defaults unless the checks pass and both performance measures stay within the guardrail.