# The types of the dense matrices of Qwen3.6

The dense matrices of Qwen3.6-35B-A3B (the attention of the 10 full
layers, the projections and the output of the 30 DeltaNet layers: 130
matrices, 2.7 GB in bfloat16) stay on the GPU; the experts mostly do not.
On a small GPU their bytes are context: on a card of 4 GB each 0.1 GB is
about 9K tokens of the int8 cache. This file gives the procedure that
chooses a type for each matrix for a budget of bytes, the tools, and the
measures behind it.

## The types

| type | bits a weight | the form |
|---|---|---|
| Q4_K, Q5_K, Q6_K, Q8_0 | 4.5, 5.5, 6.56, 8.5 | the blocks of ggml (the quantizers: `cops.kq_to_qk`, `kq_to_q6_k`, `kq_to_q8_0`) |
| RQ4_K, RQ5_K, RQ6_K, RQ8_0 | the same | the blocks of the rows after the TQ6 rotation of each 32 values (`reclank.formats.gguf.ROT_BASE`); the runtime rotates the x of the product (`KMat.rot`, `compile_qwen_step`) |
| BF12 | 12.25 | the bfloat16 values, exact but for the values more than 15 binades under the largest of their group of 32 |

The rotation spreads a large weight of a group over the group: against the
checkpoint it gains 0.4 to 3 dB at each bit depth (the most on the keys
and the values of the full layers).

## The procedure

1. **The inputs.** `scripts/qwen_dense_study.py collect` runs real text
   (the docs of the repository, 2048 tokens) through the Python path of a
   GGUF and keeps the mean square of each input channel of each dense
   product.

2. **The error of each matrix in each type.** `qwen_dense_study.py score`
   quantizes each matrix of the checkpoint (`Qwen/Qwen3.6-35B-A3B`, in the
   order of the GGUF: the DeltaNet value heads are tiled there) in each
   type and gives the error of its output, the error of each weight
   weighted by the mean square of its input channel, in dB against the
   output itself (`study.json`). A channel with a large input makes the
   coarse weights of that channel cost more: the output of the DeltaNet of
   layers 0 and 2 has a channel with 3900 times the mean input power.

3. **The sensitivity of the model to each layer.** The same error of the
   output of a matrix does not cost the model the same in each layer.
   `scripts/qwen_dense_layers.py` loads a file with every dense matrix in
   BF12 (the reference), and for each layer gives its dense matrices the
   values of one type (RQ4_K) in place on the GPU (still BF12: the
   programs stay), runs 4096 teacher-forced steps after a prompt of 128
   tokens, and measures the KL of the reference to it at each position.
   The runs repeat to the bit (a second run of the reference: KL 6e-10).
   The sensitivity of layer l is a_l = KL_l / (its summed output error).
   The kinds of matrix in a layer differ too: `scripts/qwen_dense_eval.py`
   on files of `convert_q36_dense.py --plan` with one kind in RQ4_K gives
   the relative sensitivity of each kind (`qwen_dense_plan.py fit`), and
   `qwen_dense_plan.py fit-layers` combines them: s of a matrix = a_l r_K.

4. **The plan.** `qwen_dense_plan.py plan` gives each matrix the type that
   makes the predicted KL (sum of s times the output error) least for a
   budget of bytes, among the types given: greedy upgrades along the
   convex hull of the bytes and the error of each matrix (the Lagrangian
   of the budget).

5. **The file, and its check.** `scripts/convert_q36_dense.py --plan
   plan.json` writes the GGUF (the other tensors of the template stay);
   `qwen_dense_eval.py run` and `compare` measure it against the BF12
   reference end to end.

## The measures

The text: the docs of the repository. The reference: the file with every
dense matrix BF12 (exact but for 61285 of the 1.36 G values, each more than
15 binades under the largest of its group: -120 to -130 dB). The experts and
the cache (int8) are those of the UD-Q4_K_M file in every file, so the KL
measures the dense matrices alone; it has a floor near 3e-3 for any change
of the weights (a small difference changes the choice of an expert of 8 of
256 now and then).

**The error of the outputs** (step 2; the median over the matrices of a
kind, dB, weighted by the inputs):

| kind | matrices | MB bf16 | RQ4_K | RQ5_K | RQ6_K | RQ8_0 | Q8_0 |
|---|---|---|---|---|---|---|---|
| attn_gate | 30 | 503 | -24.1 | -30.1 | -36.2 | -46.7 | -46.2 |
| attn_qkv | 30 | 1007 | -22.5 | -28.5 | -34.7 | -45.1 | -44.2 |
| ssm_out | 30 | 503 | -22.5 | -28.4 | -34.6 | -45.1 | -44.3 |
| attn_q | 10 | 336 | -23.5 | -29.5 | -35.6 | -46.0 | -45.5 |
| attn_output | 10 | 168 | -22.5 | -28.4 | -34.6 | -45.0 | -44.4 |
| attn_k | 10 | 21 | -21.7 | -27.6 | -33.8 | -44.4 | -43.3 |
| attn_v | 10 | 21 | -19.6 | -25.5 | -31.8 | -42.3 | -41.0 |

The worst matrices: the output of the DeltaNet of layers 0 and 2 (RQ6_K
-25.5 and -27.0 dB: a large input channel).

**The sensitivity of each layer** (step 3: its dense matrices in RQ4_K,
the KL of 4096 positions, a_l = KL / the summed output error):

| layers | a_l |
|---|---|
| 0 to 14 | 0.11 to 0.35 (layer 4 0.35, layer 1 0.25, layer 0 0.21) |
| 15 to 26 | 0.05 to 0.15 |
| 27 to 39 | 0.02 to 0.05 |

The output of the model is ten times as sensitive to the first layers as
to the last: the late layers take the lower types. In each group of four
layers the full layer (3, 7, 11, ...) has the most output error and the
least a_l. The kinds (one kind in RQ4_K, the rest as the best file below;
the relative s of the fit): attn_q 9.3e-2, attn_output 8.1e-2, attn_v
6.7e-2, attn_qkv 6.4e-2, ssm_out 4.8e-2, attn_gate 3.6e-2, attn_k 1.6e-2.

**The files** (step 5; a prompt of 2048 tokens and 4096 teacher-forced
steps on the 3090, the KL to the reference; the paired errors of the
differences about 1.7e-4; "plan" is `qwen_dense_plan.py plan` with the
types RQ4_K, RQ5_K, RQ6_K, RQ8_0, BF12 for the bytes of the 130 matrices):

| dense | matrices GB | GPU GB | KL | prompt, decode tok/s |
|---|---|---|---|---|
| Q8_0 (the UD-Q4_K_M of Unsloth) | 1.36 | 1.60 | 4.32e-3 | 905, 46.7 |
| RQ8_0 | 1.36 | 1.60 | 3.21e-3 | 841, 49.0 |
| plan, 1.359 GB (105 RQ8_0, 19 BF12, 6 RQ6_K) | 1.36 | 1.60 | **2.73e-3** | 857, 52.4 |
| RQ8_0, 26 BF12 (the worst output errors) | 1.39 | 1.63 | 2.73e-3 | 818, 48.2 |
| RQ6_K | 1.05 | 1.29 | 6.07e-3 | 742, 49.7 |
| plan, 1.049 GB (80 RQ6_K, 25 RQ8_0, 21 RQ5_K, 4 BF12) | 1.05 | 1.29 | **5.40e-3** | 753, 51.5 |
| RQ6_K, 26 BF12 | 1.10 | 1.34 | 5.00e-3 | 719, 49.9 |
| plan, 0.95 GB (52 RQ5_K, 52 RQ6_K, 17 RQ8_0, 9 RQ4_K) | 0.95 | 1.19 | 7.64e-3 | 711, 39.5 |

At the bytes of RQ8_0 the plan is 15% nearer the reference, as the hand
choice of 26 BF12 matrices with 30 MB more; at the bytes of RQ6_K, 11%
(-6.6e-4, about 4 times the error). The NLL of the text is the same in all
of them (+-0.002). RQ6_K of the matrices gives 0.31 GB of the GPU: with
4 GB in Pascal mode Qwen3.6 then holds a prompt of 118K tokens with the
int8 cache (96K with the UD file). The RQ4_K and RQ5_K matrices decode
slower (the plan of 0.95 GB: 39.5 tok/s).

## The commands

    # 1, 2: the inputs and the errors of every matrix in every type
    python scripts/qwen_dense_study.py collect --gguf UD.gguf --out act.npz
    python scripts/qwen_dense_study.py score --orig ORIG --gguf UD.gguf --act act.npz --out study.json
    # 3: the reference, and the sensitivity of each layer
    python scripts/convert_q36_dense.py ORIG UD.gguf BF12.gguf --type BF12
    python scripts/qwen_dense_layers.py --gguf BF12.gguf --tokenizer-gguf UD.gguf --orig ORIG \
        --type RQ4_K --study study.json --out layers.json
    #    (and the kinds: files with one kind in RQ4_K, qwen_dense_eval.py run, then
    #    qwen_dense_plan.py fit --run PLAN.json:RUN.npz ... --out sens.json)
    python scripts/qwen_dense_plan.py fit-layers --study study.json --layers layers.json \
        --kinds sens.json --out sens_layers.json
    # 4: the plan for a budget of the 130 matrices
    python scripts/qwen_dense_plan.py plan --study study.json --sens sens_layers.json \
        --budget-gb 1.05 --out plan.json
    # 5: the file, and its check
    python scripts/convert_q36_dense.py ORIG UD.gguf OUT.gguf --plan plan.json
    python scripts/qwen_dense_eval.py run --gguf OUT.gguf --tokenizer-gguf UD.gguf --out out.npz
    python scripts/qwen_dense_eval.py run --gguf BF12.gguf --tokenizer-gguf UD.gguf --out ref.npz
    python scripts/qwen_dense_eval.py compare ref.npz out.npz

Step 3 takes the longest: 40 runs of 4096 steps, about 80 minutes on the
3090; the rest a few minutes each (a conversion about 1.5 minutes, an
evaluation about 3.5).

## Limits

- One text (the docs of the repository) for the inputs, the sensitivities,
  and the checks.
- The sensitivity of a layer is measured with all its dense matrices in
  RQ4_K and the model is linear in the output errors: the plans with
  RQ5_K and RQ6_K are an extrapolation, which the checks of step 5 hold.
- The experts stay as they are in the template (Q4_K, Q5_K of UD); the
  shared experts stay Q8_0 (the MoE kernels of Qwen3.6 do not rotate).
