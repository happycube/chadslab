# numpy-gemma — a NumPy-only Gemma 4 12B runtime

This project runs the model google/gemma-4-12B-it-qat-q4_0-unquantized.
It uses NumPy only. It does not use PyTorch. It does not use transformers.
The project is the Phase 1 baseline of the learning plan in
../Gemma LLM Runtime Learning Plan.md.

The runtime does five tasks:
1. Read the SafeTensors weights. Convert bfloat16 data to float32 data.
2. Run the 48 decoder layers.
3. Keep a key and value cache. Generate tokens one at a time.
4. Change text into token ids. Change token ids into text.
5. Keep the weights in memory. Then later passes are fast.

## Files

    numpy-gemma/
    ├── np_gemma/
    │   ├── st.py          Read SafeTensors files. Use mmap. Convert bf16 to f32.
    │   ├── config.py      Read the configuration. Make a plan for each layer.
    │   ├── ops.py         Give rms_norm, linear, gelu_tanh, softmax, softcap.
    │   ├── rope.py        Make the default RoPE and the proportional RoPE.
    │   ├── model.py       Give KVCache and Model. Run the forward pass.
    │   └── tokenizer.py   Give the BPE tokenizer and the chat template.
    └── scripts/
        ├── check_trace.py      Compare each intermediate with a HF trace.
        ├── check_cache.py      Compare the KV cache with a batch prefill.
        ├── check_tokenizer.py  Compare the tokenizer with AutoTokenizer.
        ├── generate.py         Generate tokens from token ids.
        ├── chat.py             Generate text from a prompt.
        ├── session.py          Load one time. Then answer many prompts.
        └── gen_ids.py          Write greedy token ids for a HF comparison.

## Start here

Set the paths one time:

    cd numpy-gemma
    PY=../gemma4-12b-qat-pytorch/.venv/bin/python
    SNAP=$(dirname "$(find ../gemma4-12b-qat-pytorch/.cache/huggingface -name model.safetensors | head -1)")

Use PYTHONPATH=. for each command. The scripts import the np_gemma package
from the project root.

Test the tokenizer. This step is fast. It does not load the weights.

    PYTHONPATH=. $PY scripts/check_tokenizer.py --snapshot "$SNAP"

Run one prompt in a resident session. This step loads the weights one time.
Then it answers the prompts.

    PYTHONPATH=. $PY scripts/session.py --snapshot "$SNAP"         --prompts "The capital of France is" --max-new-tokens 8

Run the session without --prompts. The session then reads prompts from the
keyboard. Type :reset to clear the history. Type :q to stop.

    PYTHONPATH=. $PY scripts/session.py --snapshot "$SNAP"

Change the weight format with --dtype. The default is f32.

    PYTHONPATH=. $PY scripts/session.py --snapshot "$SNAP" --dtype bf16 --prompts "Hello"

## Weight modes

The command load_all() loads all layers and the embedding table one time.
The model then keeps the data in memory. Later passes do no file input.

    mode          memory    speed              notes
    f32 (default) 70.1 GB   about 2.1 s/token  Fast. Use this mode first.
    bf16          23.6 GB   about 17.3 s/token Three times less memory. Slower.
    no residency  small     about 5-6 min/token  Read the file for each token.

The f32 mode keeps the weights as float32 values. The bf16 mode keeps the raw
bfloat16 values and converts them during each multiply. The bf16 conversion
makes a float32 block for each group of output rows. Thus only a small float32
block exists at a time. The environment variable NP_GEMMA_BF16_CHUNK sets the
number of rows in one block. The default is 8192.

The bf16 conversion is not a fused kernel. It writes a float32 block and reads
the block again. Thus the bf16 mode uses more memory bandwidth. A native kernel
in the C port will remove most of this cost.

## Compare with Hugging Face

The file check_trace.py compares each intermediate tensor with a trace from the
real model. The reference trace is in
../gemma4-12b-qat-pytorch/traces/ref-france-pos3.

    PYTHONPATH=. $PY scripts/check_trace.py         --config "$SNAP/config.json" --weights "$SNAP/model.safetensors"         --trace ../gemma4-12b-qat-pytorch/traces/ref-france-pos3

Add --layers 1 for a quick test. Layer 0 needs only 0.5 GB of the 24 GB file.

The file check_cache.py compares the key and value cache with a batch prefill.

    PYTHONPATH=. $PY scripts/check_cache.py --config "$SNAP/config.json"         --weights "$SNAP/model.safetensors"         --trace ../gemma4-12b-qat-pytorch/traces/ref-france-pos3 --layers 1

The file gen_ids.py writes greedy token ids. Compare the ids with the HF file
../gemma4-12b-qat-pytorch/scripts/hf_gen_ids.py.

    PYTHONPATH=. $PY scripts/gen_ids.py --snapshot "$SNAP" --dtype bf16         --prompts "Count from 1 to 10." --max-new-tokens 24 --out ids.json

## Test results

    Test                                      Result
    Layer 0, position 0                       16/16 tensors, cosine >= 0.999996
    Layer 0, position 3 (RoPE active)         16/16 tensors, cosine >= 0.99999
    Up to global layer 5 (K=V, p-RoPE)        85/85 tensors, cosine >= 0.999
    All 48 layers and the logits              671/671 tensors, cosine >= 0.999
    Final hidden state                        cosine 0.999678
    First greedy token                        HF 50429 = NumPy 50429
    Key/value cache against a batch prefill   relative L2 4.8e-07
    Tokenizer encode and decode               25/25 and 25/25 exact
    Chat template, 5 message shapes           5/5 exact
    24 greedy tokens, 2 prompts, f32          48/48 ids equal to HF
    24 greedy tokens, 2 prompts, bf16         48/48 ids equal to HF

The f32 mode computes all values as float32. The real model computes most
values as bfloat16. Thus small differences occur. The cosine value stays above
0.999. The generated token ids are equal.

## Model facts

These facts are necessary. A generic transformer will give wrong output.

* The model has 48 layers. Forty layers are sliding layers. They have
  head_dim 256 and 8 key and value heads. Eight layers are global layers. They
  are at positions 5, 11, 17, 23, 29, 35, 41, and 47. They have head_dim 512
  and 1 key and value head.
* A global layer has no value projection. It uses the raw key projection for
  the value. It applies RMSNorm to the value. It does not apply RoPE to the
  value.
* RMSNorm multiplies by the full weight. Do not add 1 to the weight. The weight
  values are large. The mean is 6.6. The maximum is 193.
* The attention scale is 1.0. Do not divide by the square root of head_dim.
  The QK-norm replaces this step.
* A sliding layer uses the default RoPE with theta 1e4. A global layer uses the
  proportional RoPE with partial_rotary_factor 0.25 and theta 1e6. Only 64 of
  256 angle pairs turn. The inverse frequencies have zeros at the end. The
  rotate_half function pairs dimension i with dimension i + head_dim / 2.
* Multiply the input embeddings by the square root of hidden_size. The value is
  the square root of 3840.
* Each layer has a trained scalar. Layer 0 has 0.0544. Layer 5 has 0.365.
  Layer 47 has 0.0496. Multiply the layer output by this scalar.
* The embedding table and the output head are tied. Apply the final logit
  softcap: tanh(logits / 30) * 30.
* The attention mask is causal. A sliding layer also masks keys that are older
  than 1024 positions.
* The tokenizer replaces each space with the character U+2581. It uses BPE with
  byte fallback. The decoder replaces U+2581 with a space, joins the byte
  tokens, and joins the parts. In thinking mode, the chat template opens the
  system turn with the think token. In normal mode, it closes an empty thought
  channel.

## Limits

* The f32 mode needs about 70 GB of memory. Use the bf16 mode when memory is
  small.
* The bf16 mode is about eight times slower than the f32 mode.
* The key and value cache keeps the full sequence. It does not cut the sequence
  to the sliding window.
* The runtime gives the correct first token. The runtime and the reference can
  disagree at later tokens, because one uses float32 and the other uses
  bfloat16. The test above found no disagreement in 48 tokens.

## Next steps

1. Cut the key and value cache to the sliding window. Use a ring buffer.
2. Move the bf16 conversion to a native fused kernel in C.
3. Add bf16 rounding after each operation. Then the float32 mode follows the
   reference more closely.
4. Port the model to C.
