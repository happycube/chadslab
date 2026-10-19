# LFM2.5-24B-A2B

An evaluation of supporting LiquidAI/LFM2.5-24B-A2B (bfloat16 checkpoint,
48.0 GB, 10 shards, model_type `lfm2_moe`). Sources read: config.json,
the tensor index and headers, tokenizer.json, chat_template.jinja, the
README, and transformers 5.19 `models/lfm2_moe/modeling_lfm2_moe.py`. I
compared these against the Qwen3.5/3.6 path (qwen35.py, gpu35.py) and
gpu38.py.

## 1. What the model is

| | Qwen3.6-35B-A3B (what reclank runs) | LFM2.5-24B-A2B |
|---|---|---|
| layers | 40: 3 Gated DeltaNet + 1 gated attention, repeated | 40: 30 "short conv" + 10 GQA attention (attention at 2, 6, 10, ... 38) |
| hidden | 2048 | 2048 |
| attention | 16 q / 2 kv heads, head 256, gated output, partial rotary 0.25, M-RoPE | 32 q / 8 kv heads, **head 64**, no output gate, NEOX rope on all 64 dims, theta 1e7 |
| q/k norm | RMSNorm (1 + w) per head | RMSNorm (x * w) per head |
| recurrent layer | Gated DeltaNet (conv 4 + delta-rule state 128x128 per head) | **double-gated short conv**: in_proj 2048 -> 3 x 2048 (B, C, x); y = C * conv3(B * x); out_proj. Depthwise conv with kernel 3 and no bias. Its state is the last 2 values of B * x per channel |
| FFN | every layer MoE: 256 experts, top 8, expert 512, shared expert 512 | layers 0-1 **dense SwiGLU 11776**; layers 2-39 MoE: **64 experts, top 4, expert 1536**, no shared expert |
| router | softmax top 8, renormalized | **sigmoid** scores, top 4 of score + `expert_bias` (float32, -0.17 to +0.05, mean 0, selection only), weights = the chosen sigmoid scores / their sum (+1e-6), scale 1.0 |
| norms | RMSNorm (1 + w) | RMSNorm x * w, eps 1e-5. `embedding_norm` is the **final** norm (applied after the last layer, not to the embeddings) |
| head | separate lm_head | **tied** to embed_tokens (128000 x 2048) |
| vocab | 248320 | 128000 rows (124893 BPE + 124 added; the rest are padding) |
| context | 262144 | 131072 |
| MTP / drafter | MTP layer | none in the checkpoint. The README lists "LFM2.5-24B-A2B-DSpark ... coming soon" (SGLang) |
| thinking | optional `<think>` | **always thinks**: the generation prompt ends in `<|im_start|>assistant\n<think>` |

**Sizes (parameters):** experts 22.95B (38 x 64 x 9.44M), conv layers 0.50B,
attention 0.105B, dense MLPs 0.145B, routers 5M, embedding/head 0.262B.
That is 23.97B in total. Active per token: 2.45B, of which experts are 1.43B
(152 experts x 9.44M).

**In BF12:** 36 GB in total, of which experts are 34.4 GB (14.2 MB per
expert, 0.91 GB per layer) and the dense part is 1.5 GB. Each token reads
3.7 GB, of which 2.15 GB is experts.

Checkpoint layout: per-expert tensors
`model.layers.N.feed_forward.experts.E.{w1,w3,w2}.weight` (gate, up, down;
not fused), `feed_forward.gate.weight` [64, 2048] bf16,
`feed_forward.expert_bias` [64] float32, `conv.conv.weight` [2048, 1, 3],
`conv.in_proj`, `conv.out_proj`, `self_attn.{q,k,v,out}_proj`,
`self_attn.{q,k}_layernorm`, `operator_norm`, `ffn_norm`.

## 2. How much of reclank carries over

CPU and GPU run one program (`compile_qwen_step`, qwen35.py:661). The GPU
wraps it in `_Emit` (gpu35.py:1237), so most of the port is new emission in
that one function plus a few new records.

**Reused as is:**
- the program IR and the KQ linear and MoE kernels (any number of experts up
  to 256, top-k up to 16, inner size up to 4096: 64 / 4 / 1536 fit);
- HotCache;
- BF12 (type 57);
- RMS_NORM, the rope tables and the int8/TQ6 KV cache formats;
- the session cache: `QwenEngine._state/_save/_restore` lists `cache.conv`
  generically, and LFM2's state is only 30 x 2 x 2048 floats (480 KB a
  session), plus the K/V of 10 layers;
- the head: gpu38.py's `_head` is a `KQ_LINEAR` of any type plus
  `GP_ARGMAX`, so a BF12 tied head works. The gpu35 head is Q6_K only; use
  the gpu38 version;
- the sampler (repetition_penalty 1.05 is in core/sampling.py);
- the server plumbing.

**Small edits (about 300 lines):**
- the config and name map (`from_gguf`, `_GGUF_NAMES`); `layer_types` with
  a third kind, `"conv"`;
- an `ATTN_PREP` flag for no gate (it always assumes the gated q layout
  today), dropping the `SIGMUL`;
- a dense MLP in layers 0-1, built from existing records (`KQ_LINEAR` x 2
  -> `ENC_SILU_MUL` -> `KQ_LINEAR`);
- guards for the expert layers in gpu35's HotCache and store loops;
- making `KQ_HOT_MOE` work with no shared expert (it assumes
  `pairs = k*t + t`, runner.cu:1520);
- `ARCHS` / `load()`.

**New code (about 900-1300 lines):**

| piece | size | note |
|---|---|---|
| `SHORTCONV` record (NumPy, CPU, GPU) | ~120 | B * x -> causal depthwise conv3 with state -> * C. Clone of `k_ple_conv` (hyperconn.cu:228); B, C and x are column slices of the in_proj output, so the existing MUL does not fit |
| sigmoid router with bias | ~60 | `GP_DS_ROUTER` (dsv41/step.cu:1513) already does score + bias for the choice and s / sum(s) for the weights, with sqrt-softplus. Add a sigmoid mode and a CPU body |
| GPU attention for head 64 | ~300-500 | **the largest item.** `attn_launch` takes only 256/512 (launch.cu:53): decode, group and flash all read in 256-value chunks. The CPU attention is generic. A stopgap is to zero-pad q/k/v to 256 with scale 64^-0.5, which works but makes the K/V 4x larger |
| tokenizer pre-split | ~50 | tokenizer.py hardcodes Qwen's regex. LFM2's is GPT-4-like with `\p{N}{1,3}` (digits in threes; Qwen uses single digits) and `\p{L}+` (no `\p{M}`), with no normalizer (Qwen assumes NFC). Read the pattern from tokenizer.json |
| tool calls | ~80 | `<|tool_call_start|>[f(a=1, b='x')]<|tool_call_end|>`: Pythonic, not Qwen's XML/JSON. Parse with `ast` |
| converter | ~250 | `scripts/convert_lfm25_gguf.py`: safetensors -> GGUF in BF12 with llama.cpp's `lfm2moe` names (shortconv.in_proj/conv/out_proj, exp_probs_b, token_embd_norm for the final norm; verify these against llama.cpp), with the expert stacks unfused as convert_q8_gguf.py does |

Altogether this is the size of the MiniCPM port without its vision tower:
the work is in the attention kernel and the conv record, not in the plumbing.

## 3. GPU memory (RTX 3090, 24.5 GB)

| | GB |
|---|---|
| dense matrices in BF12 (conv, attention, dense MLPs, routers, tied head) | 1.5 |
| K/V: 10 layers x 8 kv heads x 64 x K+V = 20 KB a token in bf16 | 2.7 at 131K (0.7 at 32K; half that in int8) |
| conv state | ~0 |
| programs, arenas | ~3 |
| **hot experts** | **~17-19 of the 34.4 GB (about 32-35 of 64 per layer)** |

The K/V per token is the same as Qwen3.6 and MiniCPM: 10 layers x 2 x 256
is also 10240 values. Half the experts fit on the GPU in BF12.
With the head-64 stopgap, the padded K/V would cost 8-11 GB at 131K and
leave about 10 GB for experts, so the real kernel matters.

## 4. Speed, estimated (to be measured)

- **CPU only:** 3.7 GB a token at about 105 GB/s a node gives a limit of
  about 28 tok/s, and realistically 18-22.
- **GPU + CPU (hot experts on the GPU, cold ones on the CPU from RAM):**
  - the GPU reads 1.1 GB of dense weights plus the hits;
  - the CPU reads the misses: (1 - h) x 2.15 GB;
  - with a hit rate h of 0.75-0.85 (half the experts resident; the skew of
    the 64 experts is not measured yet), the misses are 0.3-0.5 GB a token;
  - that gives roughly **80-130 tok/s plain decode**, with a limit near
    200. The per-layer CPU/GPU handoffs (38 MoE layers) will decide where
    in that range it lands.
- **Compared with the Qwen3.6-35B-A3B class:**
  - LFM2.5 is 32% smaller in total, and 36 GB in BF12 against about 52 GB,
    so twice the share of experts fits on the GPU;
  - but each token reads more expert bytes (4 x 1536 against 8 x 512 per
    layer: 2.15 GB against 1.5 GB);
  - and coarse experts make a hot set less effective, because each miss is
    3x larger.
- **Speculative decoding:** none until Liquid's DSpark ships. When it does,
  scripts/minicpm_dspark_reference.py is the template (if it is the same
  DFlash family), and the MiniCPM numbers (1.6-2.7x) show what to expect.
  There is no MTP layer to fall back on.
- **Prefill** should be fast: the conv layers are cheap (two matrices and a
  3-tap filter), and only 10 layers attend.

## 5. Risks and points to check

- **Router precision:** the expert_bias is float32 and the margins are small
  (bias range 0.22 against sigmoid scores in [0, 1]). Keep the router in
  float32 and check the selections against transformers on real prompts.
- **The conv state holds B * x, not x.** The update must cache the gated
  input (transformers `causal_conv1d_update` on `B * x`).
- **The final norm** is named `embedding_norm` but runs at the end.
- **BOS:** the template starts with `<|startoftext|>` (124894); EOS is
  `<|im_end|>` (124900).
- **Always-on thinking:** turning thinking "off" is not a mode the model was
  trained for. Our think budget/wrap logic applies, but the default answer
  will be longer.
- **The tokenizer** must match exactly: the digit grouping and the missing
  NFC change ids on real text. Check it against HF `tokenizers` on the golden
  prompts.
- **Quality:**
  - by Liquid's own claims it beats Qwen3.6-35B-A3B on agentic and tool use
    (LiveClaw 48.1 against 38.2);
  - the README says it is "not recommended for knowledge-heavy tasks on its
    own";
  - it is a different family from everything we run: no lineage with Qwen,
    and nothing (hot-expert profiles, MTP) carries over.

## 6. Phases

0. **Reference dumps:** a transformers float32 run on the CPU, like
   scripts/minicpm_reference.py (lazy experts). Hidden states per layer,
   routes, logits on the doc / think / code / tool-call prompts, written to
   ~/reclank-golden/lfm25. Also the **expert-use profile**, which settles
   the hit-rate estimate in section 4. About 1-2 hours of CPU.
1. **Converter and the CPU path:** BF12 GGUF (36 GB, next to the checkpoint
   on the NVMe), config and names, the conv record (NumPy and CPU), the
   sigmoid router on the CPU, the dense MLP, the tied head, the tokenizer.
   Exact against the dumps.
2. **GPU:** SHORTCONV and the router on the GPU, the ATTN_PREP no-gate flag,
   the head-64 attention kernels, HotCache with no shared expert, the gpu38
   head. Checked against the CPU path as gpu35 is (check_qwen_gpu-style).
3. **Server:** detect `lfm2moe`, the tool-call parser, start_server.sh
   `M=lfm`.
4. **DSpark,** when Liquid publishes it.

## 7. Recommendation

The port is moderate: about 1.2-1.6K lines in all.

- **Largest items:** a head-64 GPU attention kernel and the short-conv
  record.
- **Easier than MiniCPM:** no vision tower, no M-RoPE, no DeltaNet.
- **Main gain:** the whole model is 36 GB in BF12, so half the experts sit
  on the 3090 at full quality. No other MoE we run gets close to that.
- **Main open question:** how skewed its 64-expert routing is (Phase 0
  measures it), and whether its quality on your workloads justifies a
  fourth family.

## 8. Phase 0 results

- **Dumps:** `scripts/lfm25_reference.py`, in float32 on the CPU, written to
  ~/reclank-golden/lfm25 (1.3 GB). Five real prompts, run greedy:
  - doc (1723 prompt tokens), code (737), tool (177), intl (39) and think
    (21);
  - 256 decode tokens each, except tool, which reached EOS at 181. Every
    run starts with `<think>`;
  - tool emitted two parallel Pythonic calls:
    `<|tool_call_start|>[get_weather(city='Lyon', unit='celsius'), search_files(pattern='SessionPool', path='reclank/')]<|tool_call_end|>`;
  - intl writes Devanagari.
  - The prompts took 141 s and used 2431 of the 2432 experts.
- **Router margins:** the 4th choice of sigmoid + bias beats the 5th by a
  median of 0.012. 6.6% of rows win by less than 1e-3, and 0.07% by less
  than 1e-5. Keep the router matrix in bf16 and compute it in float32;
  otherwise selections will flip.
- **Expert use is flat across prompts:**
  - the 32 experts used most per layer (half of them) take 69% of the uses
    on all rows pooled;
  - a static profile from the other prompts gives a decode hit rate of only
    52% for 32 hot experts (25% / 38% / 52% / 67% / 79% for H = 16 / 24 /
    32 / 40 / 48), barely better than chance.
- **Expert use is sticky within a conversation.** A hot set of the H
  experts used most in the last W rows of the same conversation gives these
  hit rates, and makes this many swaps per token:

  | W | H=24 | H=32 | H=40 | H=48 | swaps a token (H=32 / H=40), BF12 |
  |---|---|---|---|---|---|
  | 32 | 69% | 80% | 87% | 92% | 32 / 27 experts, 449 / 381 MB |
  | 128 | 67% | 78% | 87% | 94% | 12 / 11, 173 / 152 MB |
  | 512 | 62% | 74% | 85% | 92% | 5.8 / 5.0, 82 / 71 MB |

  A swap moves 14.2 MB over PCIe, about 6x the cost of reading a miss on the
  CPU. So the hot set has to change slowly (W around 512: 85% hits for 40
  slots and about 70 MB a token in the background), the way HotCache already
  re-places experts by use. A static profile is not enough.
- **Distinct experts per layer for k consecutive decode tokens:** 4.0, 7.0,
  11.8, 19.1 and 28.5 for k = 1, 2, 4, 8 and 16. A verify of 8 tokens reads
  4.8x the experts of one token, not 8x.
- **What this changes in section 4:**
  - BF12 with 32-35 slots per layer and the adaptive set gives a hit rate
    near 75-80%, which is the estimate already used there;
  - a fixed profile alone would give about 52%: 1 GB a token of misses on
    the CPU, and about 50-70 tok/s.
  - **Options, in order of quality:**
    1. BF12 with HotCache;
    2. RQ8_0 experts: 24.4 GB, 44-46 slots, about 90% hits;
    3. our own NVFP4 experts: 12.9 GB, so the whole model fits on the
       3090.

## 9. Phase 1 results

LFM2-MoE runs on the Qwen3.5 classes (`reclank/models/qwen/qwen35.py`,
`cfg.family == "lfm2"`); there is no new model module.

- **The file:** `scripts/convert_lfm25_gguf.py` wrote
  `~/models/LFM2.5-24B-A2B-BF12-GGUF/LFM2.5-24B-A2B-bf12.gguf`
  (36.7 GB, 428 tensors) in about 3 minutes, with tokenizer.json,
  tokenizer_config.json and chat_template.jinja next to it.
  - Names and metadata follow llama.cpp's lfm2moe, checked against the header
    of the official BF16 GGUF.
  - The matrices and expert stacks are BF12; norms, router, expert bias and
    conv taps are float32. Nothing had to stay bf16.
  - BF12 zeroed 0.045% of the values (largest 1.2e-3 of its matrix RMS).
    Almost all of them are in some experts' down projections, where up to 58%
    of the values are about 1e-6 next to values around 0.03 (dead neurons:
    whole columns). Elsewhere 99.996% of the values are exact, as for
    Qwen3.8.
- **Code (CPU):**
  - `QwenConfig.from_lfm2_gguf`; QwenCache keeps 2 rows of B·x per conv
    layer;
  - the NumPy reference has `short_conv`, `dense_mlp` and `route`, plus
    attention with no gate;
  - `compile_qwen_step` adds a `"conv"` layer type, the dense MLP
    (KQ_LINEAR x2, ENC_SILU_MUL, KQ_LINEAR), ROUTER_SIG, and ATTN_PREP with a
    null gate;
  - `lfm2_gguf_name`, and KQ_MOE with no shared expert, passing the float
    rows to BF12 experts;
  - two new CPU records: SHORT_CONV (190) and ROUTER_SIG (189). A null gate
    in `gp_attn_prep_body` means the q rows hold hd values;
  - Qwen3.6 golden on the CPU: the same bits before and after (64 steps and
    the fast path), at the same speed.
- **Check** (`scripts/check_lfm25_cpu.py` against the Phase 0 dumps):
  - NumPy path: every layer within 5.6e-6 relative of transformers;
  - program, float cache: 3897 rows, decode steps all exact (KL ~1e-9);
  - the one difference is a router tie at doc row 1609 (margin 3e-7 at
    layer 2), which changes the rows after it;
  - program, int16 cache (the LFM2 default): every decode argmax agrees
    (1200 steps), 99.8% of the prompt rows agree, mean KL 9.2e-4;
  - CPU decode: 65 ms a token on 24 threads, cold.
- **Tokenizer:** QwenTokenizer reads the Split pattern and the normalizer
  from tokenizer.json, and follows `ignore_merges`. LFM2 splits digits in
  runs of 1-3, uses no `\p{M}`, applies no NFC, and takes a whole vocabulary
  piece as one token. `check_qwen_tok.py`: 0 of 5039 texts differ for LFM2.5,
  Qwen3.6 and Qwen3.8.
- **Template:** QwenTok accepts the `{% generation %}` tag and passes
  `bos_token`. The 5 reference prompts (one with tools) give HF's text and
  ids exactly.
- **The cache:** 10240 values a token (10 layers x 8 kv heads x 64 x K+V).
  The rows were emulated at fewer bits on 2970 real rows:

  | rows | KB a token | KL to the reference | loss of the real text |
  |---|---|---|---|
  | float32 | 40.0 | 8.0e-4 (the doc tie) | 3.828 |
  | relative noise 1e-5 / 1e-4 | - | 5.0e-3 / 2.1e-2 | 3.832 / 3.828 |
  | int16 (+ a scale per 32) | 21.3 | 6.0e-3 | 3.833 |
  | 12-bit (+ a scale per 32) | 16.3 | 3.5e-2 | 3.820 |
  | int8 (+ a scale per 32) | 11.3 | 8.1e-2 | 3.819 |

  - **Why small errors still show up:** the model is fragile. Router margins
    are tiny, and its copies of earlier text are sharp. Even noise of 1e-5
    flipped one confident copy, at doc row 1153: an indent with p = 0.98
    became the start of another listed line, a KL of 14.
  - **Loss of the real text:** it stays within ±0.007 for every form. So the
    KL measures runs leaving the reference, not a loss of quality.
  - **What int16 is for:** it keeps the runs on the reference, which the
    checks need. A 12-bit form would be no closer to the reference than
    int8, so it is not worth new kernels. int8 is the choice if memory
    matters.

## 10. Phase 2 (the GPU) and phase 3 (the server)

**The GPU** (QwenGPU of gpu35.py; no new model class):
- **New kernels:** `kernels/lfm/lfm2.cu` has:
  - `k_short_conv` (GP_SHORT_CONV): one thread a channel. Only the first
    `nreal` rows of a group update the state, as the DeltaNet conv does; the
    CPU record got the same rule;
  - `k_router_sig` (GP_ROUTER_SIG): one warp a token;
  - `k_attn64` + `k_attn64_join`: attention of one query for head_dim 64,
    over the int16 or int8 cache or the float32 rows of the first attention
    layer. It splits the keys over 64 blocks a KV head, and `attn_launch`
    takes it for GP_ATTN_QC / GP_ATTN_Q8 / GP_ATTN_F32H of one token with
    hd 64.
- **Prompt groups (t > 16):** `k_flash_qc_mt` already handles any head_dim
  that is a multiple of 64. With hd 64 the tensor-core launchers decline,
  so it runs.
- **Changed kernels:**
  - `k_attn_prep` takes a null gate;
  - `kqh_slot`, `k_kqh_sum` and `k_moe_combine` skip the shared expert when
    its gate matrix or slog is null. KQ_GROUP_MOE already did.
- **gpu35.py:**
  - `moe_layers` (the stores, the tables, the prefetch, and the first copies
    of a large group);
  - `_shared(i)`, which gives none for LFM2;
  - the head as a KQ_LINEAR program when it is not Q6_K (`_kq_head`, as in
    Qwen4GPU), which is the BF12 tied head of LFM2.
- **Cache:** the first attention layer of LFM2 keeps float32 rows, as in
  Qwen (KV_FIRST). With that layer in int16 too, the doc prompt left the
  reference at row 253: mean KL 2.6e-2 against 1.5e-3 on the CPU.
- **HotCache:** when its copy jobs reach 48 (the GPU queue holds 64), it
  lands the finished ones and skips new copies for that step. Experts of
  14 MB, 8 a step, filled the queue ("too many cache copies in the queue").
- **Check** (`scripts/check_lfm25_gpu.py` against the dumps):
  - rows as steps from an empty cache: all exact (KL ~1e-7);
  - a GPU prefill, then steps: 99.7% of the argmax agree (the drift of the
    code prompt after its mixed-group prompt), mean KL 2.6e-4;
  - the Qwen3.6 golden on the GPU and the CPU: the same bits before and
    after these changes, at the same speed;
  - decode 52 tok/s with HotCache cold;
  - in the server: 60-64 tok/s on answers of 750-850 tokens (23 hot experts
    a layer with the cache of 131072 tokens on the GPU);
  - prompts: 190-230 tok/s at 700-1700 tokens.

**The server:**
- **Engine:** `engine.py` takes `lfm2moe` (QwenGGUFProgram, plus QwenGPU on
  the GPU, sized for `--ctx`).
- **Defaults** from the model card: temperature 0.1, top_k 50, repetition
  penalty 1.05 (the backend now has a repetition-penalty default), 8192 max
  tokens, model id `lfm2.5-24b-a2b`.
- **Tool calls:** `lfm_parse_output` (chat.py) parses
  `<|tool_call_start|>[f(a=1), g(b='x')]<|tool_call_end|>` with `ast`. It
  accepts positional arguments (named from the tool's schema), JSON words,
  and the JSON list form, and holds a call back while it is incomplete.
- **Thinking:** the template always opens `<think>`, so the reasoning comes
  back as `reasoning_content`, and the think budget of the server applies.
- **Start:** `M=lfm work/start_server.sh` uses the GPU, or `B=cpu` for the
  CPU alone.
- **Tested on the GPU and the CPU:**
  - a plain question;
  - a tool call (`get_weather(city='Lyon', unit='celsius')`, finish
    `tool_calls`), then the tool result, reusing 108 of 157 tokens;
  - a second turn;
  - streams (the reasoning and content deltas).

## 11. RQ8_0 experts (option)

- **The file:** `scripts/convert_lfm25_gguf.py --experts rq8` keeps the dense
  matrices in BF12 and writes the routed experts in RQ8_0: Q8_0 of the rows
  after the TQ6 rotation of each 32 values. The file is
  `LFM2.5-24B-A2B-rq8.gguf`, 26.0 GB (36.7 all BF12); the experts are at
  -45.44 dB against bf16, as Qwen3.8's.
- **Runtime:**
  - QwenGGUFCPU sets `rot_experts` when every routed expert of the file is
    rotated, and the kernels then rotate the act (kq_set_moe_rot; on the GPU
    rc_gpu_set_moe_rot);
  - compile_qwen_step copies and rotates the input of the experts after the
    router (COPY + TQ_ROT, as in Qwen3.8);
  - on the CPU the experts read x as int16 (two int8 planes, act bit 32).
    With int8 x the CPU and the GPU parted at row 9 of a prompt; with int16
    they agree on 96 steps (KL 3e-8).
- **Results:**

  | | BF12 experts | RQ8_0 experts |
  |---|---|---|
  | hot experts a layer (131K cache room) | 27 | 42 |
  | GPU decode, cold HotCache (check) | 52 tok/s | 101 tok/s |
  | GPU prompt, 700-1700 tokens | 195-227 tok/s | 294-430 tok/s |
  | loss of the real text of the prompts (2692 tokens) | 4.1686 | 4.1720 |
  | KL (top 64) to the reference, CPU prompt rows | 1.3e-3 | 1.1e-1 |

  - **What the KL means:** a -45 dB change of the experts, like an int8
    cache, moves this fragile model off the reference run (the router's near
    ties). The loss of the real text moves by +0.003, inside the ±0.007 of
    noise measured for the cache forms.
  - **The checks:** check_lfm25_*.py compare against the reference run, so
    they say FAIL for the RQ8_0 file. Judge it by its loss of text and by
    the GPU-against-CPU agreement.
- **Start:** `X=rq8 M=lfm work/start_server.sh` takes the RQ8_0 file;
  `X=bf12` is the default.

### RQ8_0 dense matrices and head (`--dense rq8`)

- **The file:** `convert_lfm25_gguf.py --experts rq8 --dense rq8` writes the
  conv in_proj and out_proj, the attention, and the dense MLPs in RQ8_0, plus
  the head as its own RQ8_0 matrix (`output.weight`; lfm2_gguf_name takes it
  when the file has one). The embeddings stay BF12: they are the rows the CPU
  reads. The file is `LFM2.5-24B-A2B-rq8all.gguf` (25.9 GB); the dense
  matrices are at -45.45 dB. The program rotates the x of these products
  (`dense_rot` of compile_qwen_step); the GPU head program rotates its x.
- **Correctness:** the Q8_0 products read int8 x on the CPU and, for a step,
  on the GPU (RECLANK_GPU_I8X). With `RECLANK_GPU_I8X=0` (float32 x) the GPU
  matches the float32 NumPy model of the same file on 21 and 32 real rows
  (KL 7e-8 and 5e-8, all the argmax). With the int8 x of the default, the
  runs leave the reference as for the other -45 dB changes.
- **Results** (GPU, cold HotCache, code + doc + think):

  | | RQ8_0 experts | + RQ8_0 dense and head | the same, float32 x |
  |---|---|---|---|
  | hot experts a layer | 42 | 43 | 43 |
  | decode | 100.3 tok/s | 108.3 tok/s | 104.7 tok/s |
  | prompt of 1723 tokens | 428 tok/s | 425 tok/s | 424 tok/s |
  | loss of the real text (CPU, 2692 tokens) | 4.1720 | 4.1600 | - |

- **Start:** `X=rq8all M=lfm work/start_server.sh`.

## 12. The prompt rate, and RQ6_K experts

**Measured** (`work/lfm/bench_prefill.py`: the notes and sources of the
project, warm, each size twice, cache of 131072 tokens):
- **The earlier figures were not warm:** the 290-430 tok/s were short
  prompts, plus the compile of each new mixed-group size.
- **Warm, the all-RQ8 file:** 4K 1687 tok/s, 16K 1220 tok/s.
- **A 4096-row group** took 2.3 s:
  - expert products (k_qmoe_gu/dn, float32) 1.22 s;
  - k_fetch_pull 0.65 s;
  - attention (k_flash_qc_mt, float32, no tensor cores for hd 64) 0.37 s,
    growing by 0.63 s with each 4K of context;
  - dense products 0.13 s;
  - k_short_conv 0.12 s: one thread a channel, serial over the 4096 rows.
- **Copies:** from the map of the file on NVMe they run at 4.2 GB/s (8.4 GB
  a group). The page-locked copies of the weightkeeper and of Optane run at
  12 GB/s.

**Changes:**
- **Tensor-core experts:** the int8 tensor-core path of KQ_GROUP_MOE asked
  for a shared expert in Q8_R or bf16. With no shared expert (LFM2) it now
  takes the routed experts alone. It runs with RECLANK_GPU_MOE_F32=0 (the
  default stays float32 for the Qwen models). On the all-RQ8 file the KL to
  the reference is the same: 4.77e-3 against 4.60e-3, argmax 97.1% against
  97.2%.
- **Short conv:** a group runs every row at once (k_short_conv_rows: row j
  reads B·x of rows j-2..j, from the group or from the state), then the
  state (k_short_conv_state). +4%.
- **Pinned copies:** RECLANK_GPU_NUMA_COPY=1 copies the experts to each node
  and page-locks them. The copies of a group then run at 12 GB/s. It takes
  twice the experts in memory: 49 GB for rq8all.
- **start_server.sh `M=lfm`** sets both switches (NUMA_COPY=0 and MOE_F32=1
  give the old behaviour).

| all-RQ8 file | 4K prompt | 16K prompt |
|---|---|---|
| before | 1687 tok/s | 1220 tok/s |
| tensor-core experts, pinned copies, parallel conv | 2754 tok/s | 1662 tok/s |
| the same, copies from the map of the file | - | 1460 tok/s |

**What is left:** for long prompts, attention. A tensor-core flash kernel for
head_dim 64 (the TC launchers take 256/512 only) would remove most of the
0.63 s per 4K of context. The k_fetch_pull kernel of the copies also takes
SM time.

**RQ6_K experts** (`--experts rq6`; file `LFM2.5-24B-A2B-rq6all.gguf`, with
the dense matrices in RQ8_0):

| | all RQ8 | RQ6_K experts |
|---|---|---|
| file | 25.9 GB | 20.3 GB |
| experts against bf16 | -45.4 dB | -35.0 dB |
| hot experts a layer (131K cache) | 43 | 47 |
| prompt 4K / 16K | 2754 / 1662 tok/s | 2239 / 1474 tok/s |
| decode (cold, check) | 108 tok/s | 112 tok/s |
| loss of the real text | 4.1600 | 4.1739 |

RQ6_K is not worth it here:
- its prompts are slower: the unpacking of Q6_K in the tensor-core kernel
  costs more than the smaller copies save;
- decode gains only 4%;
- the experts are 10 dB worse, and the loss is at the edge of the noise.

A study of best-of-4 rotations for RQ6 is running.

**RQ4_K experts** (`--experts rq4`; RQ5_K too with `rq5`, the encoder
`cops.kq_to_qk`; file `LFM2.5-24B-A2B-rq4all.gguf`, 14.4 GB, with the dense
matrices in RQ8_0):
- the experts are at -22.9 dB against bf16 (RQ6_K -35.0, RQ8_0 -45.4);
- all 64 experts of every layer fit on the GPU (the cache of 131072 tokens
  included), so the CPU computes no expert;
- GPU decode 132 tok/s (cold); prompt 4K 2906 tok/s, 16K 1746 tok/s.

**The loss of the prompt text is too short to judge these files.** Paired
against BF12 over 2692 tokens, the standard error of the difference is
0.014-0.024 nats:

| | difference to BF12 |
|---|---|
| rq8 | +0.004 ± 0.014 |
| rq8all | -0.009 ± 0.018 |
| rq6all | +0.005 ± 0.019 |
| rq4all | -0.048 ± 0.024 |

Each run leaves the reference path, so the loss of a token moves by
0.33-0.65 nats either way, and the mean resolves only about 0.05. The
"±0.007" quoted in sections 9 and 11 was too small. A larger evaluation
(`work/lfm/eval_nll.py`: 10 chunks of 4096 tokens spread over the notes and
sources) runs for bf12, rq8all, rq6all and rq4all.

**The larger evaluation** (`work/lfm/eval_nll.py`: 10 chunks of 4096 tokens
spread over the notes and sources, 40950 tokens; CPU program, teacher
forced; each file paired against BF12, token by token):

| | loss | difference to BF12 | perplexity |
|---|---|---|---|
| bf12 | 2.3803 | - | - |
| rq8all | 2.3821 | +0.0018 ± 0.0011 | +0.2% |
| rq6all | 2.3779 | -0.0024 ± 0.0013 | -0.2% |
| rq4all | 2.3880 | +0.0077 ± 0.0019 | +0.8% |

The standard errors over tokens and over the 10 chunks agree.
- **RQ8_0 and RQ6_K** cannot be told from BF12: less than about 0.2% of
  perplexity.
- **RQ4_K experts** cost a measurable 0.8%.
- **Caveat:** the text is the notes and sources of the project; chat,
  reasoning, and other languages are not covered.
- **The choice:**
  - rq8all for speed with no measurable loss (decode about 108 tok/s, 1.7x
    BF12);
  - rq4all for 132 tok/s at +0.8% (all 64 experts on the GPU; the split
    of each MoE layer to the CPU still runs, and a resident path with no
    handoff should make it faster).

## 13. RQ4_K decode

**A step of the rq4all file** (6.8 ms, 134.7 tok/s; nsys, the graph nodes):
- the hot experts 2.9 ms (k_kqh_gu, k_kqh_dn: the generic kernel on float32
  x, about 300 GB/s);
- the dense products 1.35 ms (k_kq_linear_i8, about 800 GB/s);
- add+norm 0.6 ms;
- the split to the CPU about 0.6 ms (hot split, signal, await, combine),
  though no expert is cold.

**Changes:**
- **Int8 x for the hot experts:** RECLANK_GPU_KQH_X=i8 (k_kqh_gu_i8) asked
  for a shared expert of an int8 type; with no shared expert it now runs.
  The int16 form (the default, x2) has no Q4_K, so Q4_K took the float32
  kernel. 134.7 -> 154.8 tok/s.
- **The resident MoE:** when every expert has a slot (QwenGPU.resident;
  RECLANK_GPU_RESIDENT=0 keeps the split), a step and a small group emit
  GP_KQ_HOT_MOE alone into the output: no split, no CPU part, no HotCache.
  154.8 -> 166.7 tok/s.
- **Checks:**
  - with float32 x everywhere the resident GPU matches the NumPy model of
    the same file on 53 real rows (KL ~1e-8, all the argmax);
  - the KL to the reference is the same with int8 or int16 x for the
    experts (1.38e-2 against 1.31e-2).
- **Start:** start_server.sh sets KQH_X=i8 for X=rq4all.

| rq4all decode (131K cache) | tok/s |
|---|---|
| before | 134.7 |
| + int8 x for the hot experts | 154.8 |
| + the resident MoE | 166.7 |

## 14. A better K-quant encoder

`cops.kq_to_qk_opt` (convert.c `rc_kq_to_qk_opt`) is now the converter's
default for the K-quant experts (`--ggml-encoder` keeps the old one). It
changes only how the codes are chosen, so the format, the size, and the
kernels stay the same.
- **qkx2 with unit weights:** no weighting by the magnitudes.
- **A search of d and dmin:** ±3% around the start, in steps of 1%.
- **A neighbour search:** ±1 on each 6-bit scale and min.

- **Weight error against the ggml encoder:** Q4_K -0.21 dB, Q5_K -0.43 dB.
- **Converting is slower:** 49 min for the rq4all file instead of 10, partly
  because a simulation shared the CPUs.

**The rq4all file, 40K-token evaluation:**

| encoder | loss | difference to BF12 |
|---|---|---|
| ggml (the old file) | 2.3880 | +0.0077 ± 0.0019 |
| opt (now `LFM2.5-24B-A2B-rq4all.gguf`) | 2.3842 | +0.0039 ± 0.0019 |

Paired against the old file, the new one is -0.0039 ± 0.0018. That halves
the loss of RQ4_K experts to about +0.4% of perplexity, at the same 14.4 GB
and 167 tok/s.

**An imatrix** (weighting by the activations) adds only about 0.04 dB on top
of this encoder, so it is not used.

**LiquidAI's QAD Q4_0 GGUF** (`LFM2.5-24B-A2B-QAD-Q4_0.gguf`, 13.6 GB: Q4_0 matrices, the
embeddings and the tied head in Q6_K, F32 norms). The products had no Q4_0 (only the Gemma path read
it, in its int4 layout): the CPU kernels gave NaN. Now KQ_Q4_0 (type 2, w = d (q - 8)) runs as Q5_1
with m = -8 d: the CPU values (every tier), the AVX2 products (dot2/avx2.c), the VNNI products and
4 x 4 tiles; on the GPU the row kernels already had it, and the tensor-core and dp4a tiles of the
experts take it (KT_Q40). The file carries its tokenizer and template: QwenTokenizer and QwenTok take
the data of formats.gguf.bpe_tokenizer_data (byte-level BPE, the pre-tokenizer lfm2: the pattern and
ignore_merges of the tokenizer.json of the checkpoint), the same 124 added tokens and the same ids on
70175 tokens of real text; the engine reads the template of a GGUF when no chat_template.jinja is
beside it (it took Qwen3.8's for a file alone). Against the reference of the original checkpoint
(the QAD model is its own model: the loss of the prompt text 3.73 nats against 4.16 for BF12): the
GPU 165 rows argmax 98.2%, KL 0.012 (decode rows; the CPU program the same); with the i8 experts the
same. All 64 experts of every layer on the GPU: decode 186 tok/s (i8 experts; 161 float), prompt
2K 2465 tok/s, 4K 2358. start_server.sh M=lfm takes it by default (X= the other files, now on the
HDD).
