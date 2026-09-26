# How to quantize the MTP drafter and test it in llama.cpp

This file gives the steps to make a small drafter file for the Gemma 4 26B
model and to measure it in llama.cpp. The plan for MTP in numpy-gemma is in
[MTP_PLAN.md](MTP_PLAN.md).

## The idea in short

A decode step of the target model reads all of its weights for one token. A
drafter is a small model. It proposes the next few tokens cheaply. The target
then checks all of them in one pass. Each accepted draft is a token that did
not need its own pass of the target.

The drafter must be cheap. In bf16, the 26B drafter reads 801 MiB for each
draft token. The target reads about 2390 MB for each token. Thus three bf16
drafts cost about one full target step, and most of the gain is lost. A 4-bit
drafter reads about 213 to 225 MiB.

## The files

    file                                                    size      bits/weight
    gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant-bf16    801 MiB   16.00
    ...-q8_0     int8, 32 values and one f16 scale per block   425 MiB    8.50
    ...-q4_0     int4, 32 values and one f16 scale per block   225 MiB    4.50
    ...-nvfp4    FP4 E2M1, 16 values and one FP8 scale         225 MiB    4.50
    ...-mxfp4    FP4 E2M1, 32 values and one power-of-2 scale  213 MiB    4.25

All the files are in `models/assistants`. The directory `models/` is not in
git. Use the steps below to make them again.

The 26B target is QAT. Google says that a QAT target needs the QAT drafter.
Thus use `gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant` and not
`gemma-4-26B-A4B-it-assistant`.

## Step 1: download the drafter

The drafter comes as safetensors, about 840 MB. Use the download script of the
adjacent project. It writes to the project-local Hugging Face cache.

    cd ../gemma4-12b-qat-pytorch
    .venv/bin/python scripts/download_model.py \
        --model google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant
    cd ../numpy-gemma

The script prints the snapshot path. The commands below call it `SNAP`.

## Step 2: convert to a bf16 GGUF

The converter of llama.cpp knows the architecture `gemma4-assistant`. Use
bf16, because the checkpoint is bf16. The output then has no rounding.

    PY=../gemma4-12b-qat-pytorch/.venv/bin/python
    D=models/assistants
    B=gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant
    mkdir -p $D
    $PY ../llama.cpp/convert_hf_to_gguf.py $SNAP --outtype bf16 --outfile $D/$B-bf16.gguf

## Step 3: quantize

`llama-quantize` needs a file type as the last argument. It then selects a
type for each tensor. The drafter needs one flag more than a normal model,
for two reasons:

- The drafter ties the output head to the token embedding. That table is
  262144 x 1024, which is 512 MiB of the 801 MiB. By default, the tool keeps
  the output head at a higher precision. Set `--output-tensor-type` and
  `--token-embedding-type` to quantize it too.
- The tool has no file type for FP4 on every tensor. The type `MXFP4_MOE`
  puts only the expert tensors in FP4 and the rest in Q8_0. The drafter has no
  experts. Thus use `--tensor-type ".*=mxfp4"` to give FP4 to every tensor.

The norm weights are one-dimensional. The tool keeps them in f32 for every
type.

    Q=../llama.cpp/build/bin/llama-quantize

    # int8 and int4: --pure gives the same type to every weight
    $Q --pure --output-tensor-type q8_0 --token-embedding-type q8_0 \
        $D/$B-bf16.gguf $D/$B-q8_0.gguf Q8_0
    $Q --pure --output-tensor-type q4_0 --token-embedding-type q4_0 \
        $D/$B-bf16.gguf $D/$B-q4_0.gguf Q4_0

    # FP4: a pattern for every tensor, plus the head and the embedding
    $Q --tensor-type ".*=mxfp4" --output-tensor-type mxfp4 --token-embedding-type mxfp4 \
        $D/$B-bf16.gguf $D/$B-mxfp4.gguf MXFP4_MOE
    $Q --tensor-type ".*=nvfp4" --output-tensor-type nvfp4 --token-embedding-type nvfp4 \
        $D/$B-bf16.gguf $D/$B-nvfp4.gguf MXFP4_MOE

The last line of each run gives the size, for example
`quant size = 212.71 MiB (4.25 BPW)`.

## Step 4: run one test by hand

Use `llama-server`. The example program `llama-speculative` cannot load this
drafter. It stops with "Gemma4Assistant requires ctx_other to be set". The
drafter reads the key and value cache of the target, so it needs a link to the
target context. Only the common speculative code of the server makes that
link.

    ../llama.cpp/build/bin/llama-server \
        -m models/gemma-4-26B-qat-q4_0/gemma-4-26B_q4_0-it.gguf \
        -md $D/$B-q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 3 \
        -c 4096 --port 18089

The server prints the same error message one or two times at the start. The
message says "this warning is normal during memory fitting". It is not a
failure. Wait for the health check:

    curl -s localhost:18089/health

Then send a request with greedy selection:

    curl -s localhost:18089/v1/chat/completions -d '{
      "messages": [{"role": "user", "content": "Explain why the sky is blue."}],
      "max_tokens": 200, "temperature": 0}' | python3 -m json.tool

The `timings` part of the answer gives the result:

    predicted_per_second   the decode rate in tokens/s
    draft_n                the count of draft tokens
    draft_n_accepted       the count of draft tokens that the target accepted

## Step 5: run the benchmark

`scripts/bench_mtp_llamacpp.py` does step 4 for each drafter file. It also
runs the target without a drafter. It uses four prompts: code, prose, a list,
and arithmetic. It compares the text of each run with the text of the run
without a drafter. With greedy selection, a drafter must not change the text.

    python3 scripts/bench_mtp_llamacpp.py
    python3 scripts/bench_mtp_llamacpp.py --quants q4_0 mxfp4 --n-max 2 3 4

One run of all five files takes about seven minutes on jackal. Do not run a
second heavy job at the same time. The decode is memory bound, and a second
job changes the result.

## Results

jackal, Xeon W-2295, 18 cores, llama.cpp c550d2f60, CPU only. The target is
the 26B QAT Q4_0 file. Greedy selection, 200 tokens for each prompt, three
drafts for each step. The ALL row is the total of the four prompts.

    drafter  file      tok/s   gain   accepted     code   prose   list   math
    none        -      17.01   1.00         -     17.18   17.02  16.72  17.14
    bf16     801 MiB   19.62   1.15   547 (73%)   22.29   17.23  19.22  20.44
    q8_0     425 MiB   21.60   1.27   546 (73%)   24.69   18.75  21.25  22.58
    nvfp4    225 MiB   21.93   1.29   542 (71%)   24.25   19.68  20.87  23.59
    q4_0     225 MiB   23.29   1.37   543 (72%)   25.58   19.96  23.48  25.01
    mxfp4    213 MiB   24.69   1.45   537 (69%)   26.63   21.33  24.63  27.05

The text of every run with three drafts is the same as the text without a
drafter.

The count of drafts for each step changes the result. Three is the best value
for both 4-bit types:

    drafter   n=2     n=3     n=4
    q4_0     21.01   23.29   22.55
    mxfp4    22.21   24.69   23.11

The runs with two and four drafts print only the totals. The script did not
compare their text.

What the numbers show:

- The size of the drafter decides the gain, and the acceptance changes little.
  From bf16 to 4-bit, the acceptance falls from 73 to 69 or 72 per cent. The
  rate rises from 19.62 to more than 23 tokens/s.
- The QAT drafter keeps its quality at 4 bits. Google trained it for Q4_0, and
  the q4_0 file accepts 543 drafts against 547 for bf16.
- NVFP4 has the same size as q4_0 but is slower. The cause is probably the
  CPU kernel. NVFP4 has an FP8 scale for each 16 values, so the kernel does
  more work for each block. This test did not measure the kernel alone.
- MXFP4 is the fastest here. It is 5 per cent smaller than q4_0 and its x86
  kernel is fast. It loses a little acceptance on code and prose.
- Prose gains the least. A drafter guesses prose less well than code or
  arithmetic.

Each value comes from one run. Two runs of the target alone gave 17.01 and
17.13 tokens/s. Thus a difference of less than about 2 per cent is noise.

## Which type to use

In llama.cpp, use the mxfp4 file with three drafts. The q4_0 file is the
second choice, about 6 per cent slower.

For numpy-gemma, use int4 (the Q4_0 layout) first:

- numpy-gemma already has the int4 GEMV and the int4 output head for this
  layout: groups of 32 values with one scale. The drafter needs no new
  kernel. MXFP4 needs a new kernel for a gain of about 6 per cent.
- The drafter is a QAT checkpoint for Q4_0. The int4 file keeps the
  acceptance of bf16.
- int8 reads 1.9 times more bytes for no gain in acceptance. One draft step
  is memory bound, like a decode step. Use int8 only as a check if the int4
  acceptance is low.

An MXFP4 kernel is a later option. The E2M1 values need a table of 16 entries.
The same byte shuffle that unpacks the int4 nibbles can do the lookup.
