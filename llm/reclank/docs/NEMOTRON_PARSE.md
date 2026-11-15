# Nemotron 3.5 Parse: document pages to text with layout

reclank runs NVIDIA's Nemotron 3.5 Parse (`nvidia/Nemotron-3.5-Parse`, OpenMDW 1.1, 0.63B parameters) and
its DFlash2 drafter. The model reads one page image. It writes the text of the page in reading order, a
block for each region: the class of the block, its box and its text. Tables come as LaTeX `tabular`,
formulas as LaTeX, the other text as Markdown.

The code:

- `reclank/models/nemoparse/model.py`: the reference in NumPy float32, op for op as the checkpoint's own
  code. It also has the page's preprocessing, the tokens, the weights' reader, the generation rules and
  the structured output (`blocks`).
- `reclank/models/nemoparse/engine.py`: the programs on the GPU and on the CPU (`ParseEngine`).
- `reclank/models/nemoparse/dflash2.py`: the DFlash2 drafter (its NumPy reference and its programs).
- `scripts/convert_nemoparse_f20.py`: the files that reclank loads.
- `scripts/check_nemoparse.py`: the check against the reference.
- `server/parse.py`: the server.

## The model

| Part | What it is |
|---|---|
| Preprocessing | The page in RGB. A page taller than 2048 or wider than 1664 pixels gets PIL's bilinear resize to fit (the aspect kept). White padding at the bottom to a multiple of 16 rows and at the right to a multiple of 64 columns. CLIP's mean and std. Patches of 16 x 16. |
| Encoder | C-RADIOv4-SO400M: a ViT of width 1152, 16 heads of 72, MLP 4304, 27 blocks, 10 prefix tokens (CLS and registers). The position rows come from a learned 128 x 128 table, resized to the content's patches and sampled at each patch's centre (the "content-normalized" frame: the padding patches take the edge's rows). |
| Neck | A 1 x 1 conv and LN on the patch rows; a 1 x 4 conv of stride 4 along each row of patches and LN: (H / 16) x (W / 64) rows of 1024. The summary (tokens 0 and 1) through a linear and LN: one more row. |
| Decoder | mBART, 10 layers of width 1024, no position embeddings. The visual rows are a causal prefix: the sequence is the visual rows, then the prompt's tokens (embeddings x 32), then the output. GQA: 16 query heads, 8 KV heads of 64. FFN 4096 with exact GELU. The head is the token embedding (tied). |
| MTP head | One trained step of multi-token prediction (`extra_proj`, `extra_heads`). |
| DFlash2 drafter | `dflash2/`: 3 Qwen3-style layers (89M parameters) that draft up to 32 tokens a block from the decoder's layers 1, 4 and 8 (below). |

A US letter page at 150 dpi (1275 x 1650) gives 80 x 104 patches and 2081 visual rows. The largest page
(1664 x 2048) gives 3329 rows. The decoder's context is 16384 rows.

The prompt is control tokens only. The default asks for boxes, classes and Markdown:
`</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>`. The model card lists the
other controls (`<no_bbox>`, `<no_classes>`, `<output_plain>`, `<output_no_text>`, `<predict_no_text_in_pic>`).

The output is blocks of the form `<x_a><y_b>TEXT<x_c><y_d><class_C>`. The coordinates are 0..1 of the
page's content (scale x by the image's width and y by its height).

## The files

The download (2.52 GB, float32: the checkpoint's weights are float32, not bfloat16) stays in
`MODELS/Nemotron-3.5-Parse` as the reference. reclank loads the converted directory:

    python scripts/convert_nemoparse_f20.py MODELS/Nemotron-3.5-Parse MODELS/Nemotron-3.5-Parse-F20

| File | What it holds |
|---|---|
| `Nemotron-3.5-Parse-F20.gguf` (1.60 GB) | the model: each of the 177 matrices in F20 (16 significant bits of the float32 value), the vectors float32 |
| `dflash2/Nemotron-3.5-Parse-DFlash2-BF12.gguf` (0.14 GB) | the drafter, stored as bfloat16: its 31 matrices in BF12 (the bfloat16 values exactly), the norms and conv kernels bfloat16 |
| `config.json`, `tokenizer.json`, ... | copies |

No matrix was refused (F20 and BF12 refuse a matrix with a zeroed value above 2^-8 of its RMS). Four tensors
take the programs' layout (the values the same; `model.program_layout`): conv1 as a matrix, conv2 as the
linear of 4 patches' rows side by side, pos_embed as [16384, 1152], RADIO's fc1 and fc2 padded from 4304 to
4320 with zeros (F20 takes columns in multiples of 32).

`ParseEngine(dir)` and `ParseRef(dir)` take either directory. From the download, the CPU runs float32
matrices and the GPU makes the same F20 at load.

`GGUF.dequant` of an F20 tensor needs the F20 case of `rc_kq_rows` (backends/cpu/csrc/kernels/kq/convert.c,
added for this port): before it, F20 rows fell through to the block decoder and gave garbage.

## Generation

reclank follows the card's contract (`docs/serving-reference.md` of the checkpoint): greedy, repetition
penalty 1.1, at most 8744 new tokens, the special tokens kept. The rules are those of transformers'
`generate`:

- The repetition penalty covers each token of the prompt and of the output so far. A positive logit is
  divided by 1.1, a negative one is multiplied by 1.1.
- `forced_eos_token_id`: the last of the 8744 tokens is `</s>`.

The checkpoint also has two logits processors (`hf_logits_processor.py`: a forced `\begin{tabular}` and a
repetition stop). The card's examples do not use them, so reclank does not either.

## Precision

- The matrices: F20 on the CPU and the GPU (from the converted directory). The products sum in float32
  with float32 activations (CPU: `KQ_F20X16`; GPU: `k_kq_f20_gemv` for up to 16 rows, `k_enc_sgemm`'s F20
  tiles for more). The drafter's matrices: BF12 (exact bfloat16).
- The attention: float32 (`ENC_ATTN_TILE` in the GPU's encoder, `ENC_ATTN_RANGE` in C and in the prefill,
  `ENC_ATTN_KV` in a step). The caches of keys and values: float32.

## Drafts

Each draft is checked by the target's own step. A row's logits are the same bits in a step of one row or
of 16 (on the GPU `k_kq_f20_gemv` takes each row alone up to MT_MAX = 16 rows; in C each kernel keeps a
row's sums apart), so the output is exactly the plain greedy output.

### DFlash2 (the default)

The drafter (`dflash2.py`) follows vLLM's native DFlash2 for Parse:

- Its context: for each position the target has run, the residual stream after the target's layers 1, 4
  and 8 (`target_layer_ids`), through `fc` and `hidden_norm`; each drafter layer keeps that position's k
  (k_norm, RoPE) and v. The target's programs write the taps ("tp"); `ParseDrafter.inject_rows` makes the
  context of a prompt's rows and of each round's taken rows.
- The positions are the target's (the visual rows first). No window: a block sees every position before it.
- A round: the block [last token, mask x K] at the next position (the target's scaled embeddings), 3
  layers with the grouped dynamic conv (2 taps, groups of 16 channels) around the attention and the MLP,
  the final norm, the target's head on the K mask slots, the 16 best tokens of each, then the selector's
  greedy walk (on the host). The target runs the last token and the K drafts as one step; the drafts that
  its own choices confirm are taken, with the choice after them.

`RECLANK_NPARSE_DRAFTS` sets K (default 15: a verify step of 16 rows, the most that keeps the GPU's rows
exact). `RECLANK_NPARSE_DFLASH`: `auto` (the default) drafts with DFlash2 on the CPU and with MTP on the
GPU, `1` uses DFlash2 on both, `0` never. `ParseEngine(drafter=False)` or the server's `--no-drafter`
turn DFlash2 off too.

On the GPU (RTX 3090, the six pages, the F20 file) DFlash2 is lossless (the same tokens and text on 6 of 6
pages) but slower than MTP: 543 against 705 tokens/s, 0.304 against 0.332 pages/s. A round takes 9 to 12
tokens, but its host work (the logits of 16 rows and the drafter's 15 rows downloaded, the host's top-16
and the choices) costs more than the GPU's steps that it saves. The drafter's top-16 on the GPU
(`NM_LTOPK`) and the rules on the GPU would remove most of it.

A real page (win-060, 423 tokens; the reference's tokens, so no step of the target): K = 16 took 10.1
tokens a round, 11.1 tokens a target step. On the CPU (2 threads) K = 15 gave 9.2 tokens a step; K = 7 and
K = 31 were slower.

### MTP (GPU, without the drafter)

A GPU step can also run the MTP head on each row: `argmax(head(extra_head(h + extra_proj(embed(a) x
32))))`, `h` the row's final hidden state and `a` the argmax of the row's logits. When the host's choice is
`a`, the next step feeds the token and the draft as two rows. On the six pages below, 92% of the drafts
were taken: about 1.85 tokens a step. `RECLANK_NPARSE_MTP=0` turns it off.

### The CPU's attention of a step

`ENC_ATTN_KV` in C summed each score in order, one (query, head) a task: 16 ms a layer for one row over
1600 keys on 2 threads. Its new operand `simd` (Parse's steps only; the other models' programs do not set
it) sums a score in SIMD lanes and shares each tile of keys among the rows of a KV head (blocks of 4 rows
of the same `lo`). A row's bits still do not depend on the other rows. A step of 16 rows: 360 -> 172 ms on
2 threads.

## Checks

The reference is the checkpoint's own transformers code (`trust_remote_code`) in float32 on the CPU, run in
the bwrap sandbox (`work/gen/sandbox/drivers/nparse_hf.py`, venv `nemoparse`: torch 2.15 CPU, transformers
5.19, timm 1.0.22). The pages are six real PDF pages rendered with `pdftoppm` (150 dpi; one at 200 dpi,
which tests the resize):

| Page | Source | Size | Visual rows | Tokens |
|---|---|---|---|---|
| `im1-120` | Inside Macintosh I (a 1985 scan), p. 120 | 1200 x 1617 | 1939 | 620 |
| `win-060` | Windows 1.03 SDK reference (a scan), p. 60 | 1248 x 1272 | 1601 | 423 |
| `ams-02` | `amsfndoc.pdf` (TeX Live), p. 2 | 1241 x 1754 | 2201 | 1028 |
| `qtm-01` | `qtm-info.pdf` (TeX Gyre), p. 1 | 1241 x 1754 | 2201 | 89 |
| `volc-01` | a report with charts, p. 1 (200 dpi) | 1654 x 2339 | 2945 | 1149 |
| `clib-0030` | Open Watcom C library reference, p. 30 | 1275 x 1651 | 2081 | 786 |

`scripts/check_nemoparse.py MODEL REF_DIR PAGES_DIR [--gpu] [--threads N] [--pages-only a,b]` gives, for
each page:

- the pixels: the processor's patches against reclank's;
- the encoder: the cosine of the visual rows of the reference's patches;
- the decoder: the reference's tokens fed one at a time after the reference's visual rows, the
  KL(reference || reclank) of each step and the steps whose most likely token differs;
- greedy generation from the page file (with the drafts): the tokens and the text against the reference's.

Results:

| Run | Pages | Encoder cosine: whole / worst row | KL mean / max | Top-1 | Greedy output |
|---|---|---|---|---|---|
| GPU, F20 (from the float32 download), MTP | 6 | >= 0.99999963 / 0.99923 | 5.75e-11 / 2.79e-8 | 4095/4095 | the same tokens and text, 6 of 6 |
| CPU, float32 (the download) | ams-02 | 0.99999979 / 0.99979 | 1.30e-11 / 8.75e-10 | 1028/1028 | the same |
| CPU, F20 file, DFlash2 | im1-120 | 0.999999981 / 0.999992 | 2.45e-11 / 1.29e-9 | 620/620 | the same |
| CPU, F20 file, DFlash2 | win-060 | 0.999999977 / 0.999988 | 6.41e-11 / 2.75e-9 | 423/423 | the same |
| GPU, F20 file, DFlash2 | 6 | >= 0.9999997 / 0.99937 | 5.97e-11 / 3.34e-8 | 4095/4095 | the same, 6 of 6 |
| GPU, F20 file, MTP | 6 | the same | the same | 4095/4095 | the same, 6 of 6 |

So F20 costs nothing measurable against float32: the KL to the float32 reference stays near 1e-11 and
every greedy output is the reference's. The few worst encoder rows (about 1 a page under a cosine of
0.9999) are the same rows with float32 weights on the CPU (ams-02: 0.99979 there, 0.99978 on the GPU):
float32 order effects at sensitive rows, not F20.

`model.blocks` gives the same blocks (class, box, pixel box, text) as the checkpoint's
`nemotron_parse.postprocess_document` on all six outputs. Its Markdown is the same where a page has no
table; the official one turns LaTeX tables into HTML.

## Speed

The host was shared with other jobs throughout (load 40 to 110 on 48 CPUs); the CPU runs had 2 CPUs.

| | Encoder | Decoder | Pages/s |
|---|---|---|---|
| GPU (RTX 3090), F20 file, float32 attention, MTP | 1.2 to 3.5 s a page (2.05 mean) | 705 tokens/s | 0.33 |
| GPU, the same with DFlash2 | 2.03 s mean | 543 tokens/s | 0.30 |
| CPU, 2 threads, F20, DFlash2 | 345 to 518 s a page | 14.5 tokens/s (9 to 12 tokens a step) | |
| CPU, 2 threads, F20, no drafts | | 14 tokens/s with 1 row a step; the same page with DFlash2: 1.65 x faster (12.3 against 20.3 s for 422 tokens) | |

On the GPU, RADIO's float32 attention is most of the encoder's time: 2.35 s of the 1472 x 2048 page's
4 s (11786 patches; `k_enc_attn_tile` pads the heads of 72 to 128). A decoder step of one row is
1.27 ms of GPU graph, about 0.2 ms of logits' download, and the host's choice. The GPU with DFlash2
loses to MTP (above).

On 2 CPUs a verify step of 16 rows is compute-bound (172 ms against 47 ms for one row), so the drafter
gains 1.65 x there; with more threads a step is closer to bound by the weights' bytes, where 16 rows cost
little more than one.

Options that change the precision (not approved, measured only):

- `RECLANK_NPARSE_ATTN=f16`: RADIO's attention on the tensor cores with float16 q, k and v (`ENC_ATTN`,
  as Bolt's RADIO). Encoder 1.0 s a page (3.4 x), 0.51 pages/s. The greedy output stayed the same on 6 of 6
  pages, but the worst visual row's cosine fell to 0.57 (whole page 0.99983).
- `ENC_ATTN_X2` (two bfloat16 planes) was not faster than float32 here (3.31 against 3.37 s a page) and
  less precise (worst row 0.975), so it has no switch.

Other ways to speed it up, not done:

- Several pages a step (a batch); the server runs one page at a time.
- A float32 attention kernel for heads of 72 (no padding to 128) on the GPU.
- The drafter's head on the GPU's top-16 (`NM_LTOPK`) instead of the logits' download.

## The server

`python -m server.parse -m MODELS/Nemotron-3.5-Parse-F20 [--port 8101] [--cpu] [--no-drafter]`.
The routes:

- `POST /v1/chat/completions`: OpenAI's chat. The last user message holds one `image_url` part (a data
  URI) and, optionally, a text part with the task prompt. `max_tokens` (default 8744),
  `repetition_penalty` (default 1.1), `stream`. `temperature` (default 0: greedy, the card's contract),
  `top_k`, `top_p` and `seed` (default 0) sample instead, with no drafts (the checkpoint's
  `DocumentParser` samples its crops at 0.1 to 0.2). `message.content` is the raw output (class and
  coordinate tokens kept, `</s>` dropped), as vLLM's with `skip_special_tokens` false. The response's
  `parse` holds `blocks`, `markdown` and `image_size`.
- `POST /v1/parse`: `{"image": base64, "prompt", "max_tokens", "repetition_penalty", "temperature",
  "top_k", "top_p", "seed"}` gives `raw`, `blocks`, `markdown`, `image_size`, `finish_reason`, `usage` and
  `timings`.
- `GET /v1/models`, `GET /health`, and the GPU sharing's admin routes.

The server shares the GPU with the other reclank servers (`server/gpushare.py`): it held about 4 GB in
its test (MTP), and releases it when a peer asks. Its host memory: about 3 GB on the GPU.

A client sends PDF pages as images (render them first, for example `pdftoppm -r 150 -png`). For HTML
tables and charts, run the checkpoint's `nemotron_parse.postprocess_document` on `raw`.
