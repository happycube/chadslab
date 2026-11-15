# Nemotron 3.5 Embed and Rerank

## 1. The models

NVIDIA's retrieval models of October 2026 (license OpenMDW-1.1; the base
models Ministral-3 3B and 8B, Apache 2.0):

| model | text model | vision | output |
|---|---|---|---|
| Nemotron-3.5-Embed-1B-BF16 | 16 layers, width 2048, 24/8 heads of 128, FFN 6144 | Pixtral, 0.4B | 2048 floats |
| Nemotron-3.5-Embed-8B-BF16 | 34 layers, width 4096, 32/8 heads of 128, FFN 14336 | Pixtral, 0.4B | 4096 floats |
| Nemotron-3.5-Rerank-8B-BF16 | as the 8B embed | Pixtral, 0.4B | one logit |

- The checkpoints are transformers' Mistral3Model: a Pixtral vision tower,
  the projector (RMS norm, a 2 x 2 patch merger, two linears with GELU),
  and a Ministral3 text model with `is_causal` false (every token sees every
  token of its input).
- The text model is Llama-like: RMS norms, GQA, SiLU-gated FFN, no biases.
  The RoPE is YaRN (factor 16 over 16384 positions, theta 10^6, attention
  factor 1). Llama 4's scale multiplies q by 1 + 0.1 log(1 + floor(pos /
  16384)): it is 1 below 16384 positions. The context is 32768 tokens.
- An embedding is the mean of the last hidden state over all the tokens
  (the prefix and the image tokens too), L2-normalized. Matryoshka: the
  first d values, normalized again.
- A rerank score is the checkpoint's model.py: the same mean, times the
  score row (4096 floats, in float32; no sigmoid). Higher is more relevant.
  2_Dense of sentence-transformers holds the same row.
- The inputs (the checkpoints' chat templates; no BOS, no EOS):
  - embed: `query: ` + text, or `passage: ` + text. A document with an
    image: `passage: ` + the image + ` ` + text.
  - rerank: `query: ` + query + `\n\npassage: ` + document. With an image:
    the image + ` query: ...`, the whole stripped.
  - The text of a user is plain text. The templates put a zero-width
    non-joiner after each `[` and `<`, so that `[IMG]` in a text is not the
    token. The tokenizer's normalizer takes the marks out after the split
    at the special tokens.
- The image: the longest edge at most 1120 pixels (up to 1540), each side
  up to a multiple of 28, bicubic, CLIP's mean and std. An image of h x w
  blocks of 28 pixels is h rows of w `[IMG]` and one `[IMG_BREAK]`; the
  last `[IMG_BREAK]` is `[IMG_END]`. A letter page at 1120 is 40 x 31
  blocks: 1280 tokens.

## 2. The port

- reclank/models/nemotron_embed/model.py: NemotronRetriever (the text
  model, the tokens of the templates, embed_ids and score_ids) and _Vision
  (the tower and the projector). Both run as programs of the records
  ENC_*, as the programs of EmbeddingGemma 2 do. On the CPU they run in C:
  the bfloat16 matrices in groups of 16 rows, float32 products. On the GPU
  each is one CUDA graph, with the matrices in BF12.
- No new records:
  - The RoPE of both towers is ENC_ROPE_TAB (NEOX halves) at a table of the
    cos and sin of each row. Pixtral's 2D RoPE is a table too: angle j the
    row times the even frequencies, then the column times the odd ones.
  - The q tables hold the attention scale (and Llama 4's scale), so the
    attention has scale 1.
  - The patch merger is ENC_GATHER of the 4 patches of each block. The
    merger's columns are put in the order of the gather (the same values).
  - On the GPU the text model copies each KV head to its query heads
    (ENC_GATHER) and takes ENC_ATTN_TILE (float32). ENC_ATTN_RANGE with
    shared heads took 3.3 times as long. The CPU keeps ENC_ATTN_RANGE with
    shared heads.
- The tokenizer is the BPE of models/qwen/tokenizer.py with the pattern of
  the file (Mistral's tekken pattern, the one of transformers'
  fix_mistral_regex).
- The pixels: PIL's bicubic. transformers resizes the uint8 values with
  torchvision's antialiased bicubic: about 0.05 % of the values differ by
  one level.
- A fix of the CPU attention (kernels/attn_prefill.c). The last block of
  keys read up to 32 floats past the end of the transposed keys (kt). The
  kernel masked these values. But when the read went into an unmapped
  page, the process stopped (SIGSEGV). With 1408 rows this occurred in
  about 1 run of 3. Now kt has room for one block of keys at its end. The
  results do not change.
- The server: server/embed.py --nemotron DIR (once for each model), the
  routes /v1/embeddings (input_type query or document; images as base64)
  and /v1/rerank. See the text of server/embed.py.
- The references: scripts/nemotron_embed_reference.py (transformers 5.20,
  float32, on the CPU; the reranker's model.py in the sandbox). The checks:
  scripts/check_nemotron_embed.py and scripts/check_nemotron_rerank.py.

## 3. Status

- **Done:** both Embed sizes and the reranker, text and images, on the CPU
  (float32) and the GPU (BF12), in the server.
- **The data of the checks** (all real):
  - texts: queries and documents of work/embg (English, Chinese, code),
    the text layer of three PDFs (an FOMC press conference, a product sheet,
    a paper's abstract), and two plans of 3324 and 9809 tokens;
  - images: six pages of four PDFs from this machine, rendered at 150 dpi
    (1280 or 1200 tokens at 1120), one at 1540 (2424 tokens), one with text;
  - rerank: three queries, each against 8 or 10 documents (texts, page
    images, an image with text): 26 pairs.
- **The references:** transformers 5.20.2 in float32 with eager attention,
  in the sandbox (the card's sdpa gives the same vectors). The reference of
  the 8B for the 9809-token text is sdpa (eager took over an hour).
- **The token ids** are the same as the reference's for all 49 inputs.
- **Embed, the worst cosine to the reference** (the vector; image rows on
  the reference's pixels; "own": reclank's own pixels):

  | model | device | query | document | image | image + text | image own |
  |---|---|---|---|---|---|---|
  | 1B | CPU | 0.9999999 | 0.9999999 | 0.9999999 | 0.9999999 | 0.9999973 |
  | 1B | GPU | 0.9999999 | 0.9999999 | 1.0000000 | 1.0000000 | 0.9999972 |
  | 8B | CPU | 0.9999999 | 1.0000000 | 1.0000000 | 1.0000000 | 0.9999511 |
  | 8B | GPU | 1.0000000 | 0.9999999 | 0.9999999 | 1.0000000 | 0.9999501 |

  The image rows of the projector: cosine at least 0.999997 on each row.
  The 8B CPU check ran 17 of the 23 inputs (not 5 of the images), plus the
  9809-token text.
- **Rerank, the worst |score - reference|** (the logits are -4 to -1.7):

  | device | text | image | image + text | image own |
  |---|---|---|---|---|
  | CPU | 2.9e-6 | 2.6e-6 | 0 | 1.8e-4 |
  | GPU | 1.2e-4 | 9.2e-5 | 2.0e-5 | 2.8e-4 |

  The order of the documents of each query is the reference's (Kendall
  tau 1.0, the same first document), on both devices and with reclank's
  own pixels.
- **Speed** (CPU: 44 threads of the 2 x 24-core Xeon; GPU: RTX 3090, the
  second run of each size):

  | | 1B CPU | 1B GPU | 8B CPU | 8B GPU |
  |---|---|---|---|---|
  | 14 texts, 7388 tokens, one pass | 12.4 s (598 tokens/s) | 1.13 s (6519 tokens/s) | 81.8 s (90 tokens/s) | 7.27 s (1016 tokens/s) |
  | a query of 16 tokens | 0.07 s | 0.03 s | 0.61 s | 0.18 s |
  | a page (1284 tokens): the vision tower | 6.0 s | 0.56 s | 5.5 s | 0.57 s |
  | a page: the text model | 1.6 s | 0.20 s | 13.2 s | 1.22 s |
  | a page at 1540 (2424 tokens), all | | 1.84 s | | 3.75 s |

  The reranker runs as the 8B Embed: a pair of a query and a 400-token
  text takes 0.5 s on the GPU, 3.5 s on the CPU.
- **Memory:** BF12 on the GPU: 1.3 GB (1B text), 11.3 GB (8B text), 0.66
  GB (vision), plus the programs. On the CPU the bfloat16 matrices are
  packed in groups of 16 rows (the same bytes): about 2.6 GB (1B) and 16 GB
  (8B), and the token table stays in the map.
  A CPU server of the three models (from the HDD, page cache cold) took
  263 s to load; its resident memory was 36 GB (49 GB at the peak of the
  load).
- **On the GPU,** the text attention went from ENC_ATTN_RANGE with shared
  KV heads to ENC_GATHER + ENC_ATTN_TILE: 1B, 1320 tokens, 0.34 -> 0.21 s.
  The linears are k_gemm_bf16_tc on BF12 rows (x as two bfloat16 planes),
  the path of EmbeddingGemma 2.

### The BF12 files

- The models load from the BF12 file of scripts/convert_nemotron_embed_bf12.py
  (OUT/model-bf12.gguf and the small files of the checkpoint), on the CPU
  and on the GPU. The originals are bfloat16: each matrix is BF12, each
  vector (norms, the score row) float32 (its bfloat16 value exactly). There
  are no float32 matrices (F20 would take them). The download (bfloat16
  safetensors) still loads when a directory has no BF12 file.
- Two tensors take the layout of the programs: the patch conv as a linear
  of 608 columns (588 and zeros), and the patch merger's columns in the
  order of ENC_GATHER. With 608 columns the patch conv is BF12 on the GPU
  too.
- The files (MODELS/<name>-BF12): 1B 2.38 GB, 8B 12.83 GB, Rerank
  12.83 GB. 0 matrices kept bfloat16. Zeroed values (the bfloat16 values
  more than 15 binades under the largest of their group of 32): 1B 70,413
  (0.0045 %), 8B 341,923 (0.0041 %), Rerank 343,438 (0.0041 %).
- The loss against the float32 reference (CPU, one thread on CPU 47, a
  few inputs):

  | model | inputs | worst cosine or score difference |
  |---|---|---|
  | 1B | 4 queries, 2 documents, 1 page image | cos 0.9999999 |
  | 8B | 3 queries, 2 documents | cos 1.0000000 |
  | Rerank | 7 pairs of text | |score - ref| 2.6e-6 |

  It is the loss of the float32 products with the bfloat16 weights.
- The GPU from the files, all the inputs of section 3 (the 8B's 9809-token
  text against the sdpa reference): worst cosine 0.9999999 (1B and 8B: query,
  document, image; own pixels 0.9999971 and 0.9999499); rerank worst
  |score - ref| 1.2e-4 (text), 9.1e-5 (image), Kendall tau 1.0 for each
  query. Speed: 1B 4306 tokens/s and 8B 820 tokens/s over 17197 tokens of
  text in two passes (the 9809-token text in one); a 1120 page 0.78 s (1B)
  and 1.84 s (8B), at 1540 1.88 s and 3.87 s.

## 4. Open

- Not done: Nemotron-3.5-Embed-1B-NVFP4. Its weights are NVFP4, which is a
  precision change past BF12 (it needs the owner's approval).
- Options that need approval (precision): one bfloat16 plane of x in the
  GPU's products, or float16 activations; BF12 matrices on the CPU
  (KQ_BF12X16: 12.25 bits a value in place of 16; speed not measured).
- The CPU's vision tower takes about 6 s a page. About half of it is the
  attention over the 4960 patches (ENC_ATTN_RANGE).
- The resize is PIL's: 1 - cos up to 5e-5 against the reference's pixels
  on one page (the 8B). An exact port of torchvision's uint8 resize would
  remove it.
