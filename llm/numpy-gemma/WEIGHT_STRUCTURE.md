# Weight structure: four negative results

**Status:** measured, not usable. Recorded so the ideas are not tried again
without new evidence.

The question was whether the weights hold **linear-algebraic structure** that a
post-training transform could remove, so that the model needs fewer parameters
*and fewer calculations* for the same width.

The context that made it worth asking:

* Quantization cuts **bytes**. It does not cut **calculations**.
* The int4 tile is bound by instruction throughput, not by the multiply. The
  VNNI test says so: replacing one vpdpbusd with vpmaddubsw + vpmaddwd costs
  only **1.3x**, when a pure dot-product bound would cost about 2x.
* Prefill runs at about **500 GFLOP/s** out of a machine roof near 4 TFLOP/s,
  about 12 percent.
* So a transform that removes multiply work attacks the actual bottleneck in a
  way quantization cannot. And the two compose: rank r at 4 bits costs
  0.5625 * r * (m + n) bytes.

Four families were measured. All four fail on this model.

Measured on the Q4_0 checkpoint gemma-4-26B_q4_0-it.gguf: 30 layers, hidden
2816, 16 query heads, 128 experts with 8 active, expert width 704, dense MLP
width 2112. Layers 0 and 15, expert 0, representative matrices.

---

## 1. Low-rank within a matrix

**Method.** For a matrix W of shape m x n, find the rank r that keeps the
relative Frobenius error under a tolerance. Truncated SVD is the optimal
rank-r approximation in that norm (Eckart-Young), so this is the best any
low-rank method can do in weight space.

**Why the least-squares version is the right one.** The layer does not have to
reproduce W; it has to reproduce W x. So the objective that matters is

    min over rank-r AB of  E || W x - A B x ||^2
      = min tr( (W - AB) C (W - AB)^T ),   C = E[ x x^T ]

whose closed form is: whiten by C^(1/2), take the truncated SVD, unwhiten.
That is activation-aware SVD. Alternating least squares -- fix A, solve B, fix
B, solve A -- refines it further. Both are post-training and need only
calibration activations, which the hook mechanism already collects.

**Result, layer 15.** Weight-space error at a given rank, as a fraction of the
number of singular values.

| matrix | shape | err at 10% rank | at 25% | at 50% | rank for 5% err | rank for 10% err |
|---|---|---|---|---|---|---|
| self_attn.q_proj | 4096 x 2816 | 70.8% | 47.7% | 24.4% | 87.8% of n | 75.5% of n |
| self_attn.o_proj | 2816 x 4096 | 77.5% | 57.1% | 32.2% | 91.9% | 82.2% |
| mlp.gate_proj | 2112 x 2816 | 71.7% | 49.2% | 24.8% | 86.2% | 74.3% |
| mlp.down_proj | 2816 x 2112 | 73.3% | 53.1% | 28.8% | 89.7% | 78.9% |
| experts.gate_up_proj (expert 0) | 1408 x 2816 | 76.8% | 58.6% | 36.2% | 96.2% | 88.8% |
| experts.down_proj (expert 0) | 2816 x 704 | 83.9% | 66.9% | 44.1% | 98.0% | 93.6% |

Layer 0 is the same to within a few points. The spectra are essentially flat:
the matrices are near full rank.

**The storage paradox.** A rank-r factorization costs r * (m + n) numbers
against m * n for the dense matrix. For a 4096 x 2816 matrix the break-even is
r = m * n / (m + n) = 1669, that is **59 percent of n**. Below that rank the
factors are smaller; above it they are larger. At 50 percent rank the error is
24 to 44 percent, against roughly 1 to 2 percent for the existing 4-bit
weights. So in the useful regime the factorization is worse on both axes: more
storage **and** more error. Only combining it with quantization of the factors
makes it competitive on bytes, and it still pays the error.

**Verdict: dead.**

---

## 2. A shared basis across the 128 experts

A flat spectrum for one matrix does not rule out redundancy *between* the 128
experts of a layer. They are trained together under a load-balancing loss, so
they could be variations on a common set of directions. If so, a shared basis
of k directions plus a coefficient vector for each expert stores the block in
about k / 128 of the space. That is a Tucker / CP decomposition of the expert
tensor, a different object from a low-rank factorization of any single matrix.

**Result, layer 15.** Energy of the 128 expert matrices, split into the mean
and the deviation from it, and the energy of the deviation captured by a shared
basis of k.

    gate_up    128 experts, 1408 x 2816   mean energy 1.0%, deviation 99.0%
        basis  1 ->  0.8% of deviation energy
        basis  2 ->  1.6%
        basis  4 ->  3.2%
        basis  8 ->  6.4%
        basis 16 -> 12.8%
        basis 32 -> 25.5%
        basis 64 -> 50.7%

    down       128 experts, 2816 x 704    mean energy 0.8%, deviation 99.2%
        basis  8 ->  6.6%
        basis 32 -> 26.0%
        basis 64 -> 51.4%

The captured energy tracks k / 128 exactly: 8/128 = 6.3 percent measured
6.4 percent; 32/128 = 25.0 percent measured 25.5 percent. That is what 128
**orthogonal, isotropic** directions give. There is no shared basis, and the
mean expert holds under 1 percent of the energy, so there is no common expert
either. The experts are maximally diverse.

**Verdict: dead.**

---

## 3. Transform coding (DCT / FFT / wavelet)

**Method.** A 2D separable transform writes a matrix as a sum of rank-1 outer
products a_j * b_j. Store the coefficients and drop the rest. Two forms were
measured: the **low-frequency corner** (a K x K block, whose index costs
nothing to store), and the **best k coefficients by magnitude** (unrealistic,
since it ignores the cost of storing the indices, but it is the upper bound for
the method).

**Result, layer 15.** Error at a budget of stored numbers.

| budget | DCT corner | DCT best-k | SVD, same budget |
|---|---|---|---|
| **self_attn.q_proj 4096 x 2816** | | | |
| 16384 | 99.9% | 99.4% | 98.8% |
| 65536 | 99.7% | 98.2% | 97.4% |
| 262144 | 98.9% | 94.3% | 93.2% |
| 1048576 | 95.4% | 83.0% | 81.0% |
| **experts.gate_up_proj expert 0** | | | |
| 262144 | 96.6% | 86.7% | 86.0% |
| 1048576 | 85.7% | 61.8% | 66.9% |
| **random control 2816 x 2816** | | | |
| 262144 | 98.3% | 92.4% | 96.9% |
| 1048576 | 93.1% | **77.5%** | **88.3%** |

**The control is the point.** On random data the DCT *beats* the SVD at the
same budget, because a fixed basis costs one coefficient per term while the SVD
costs m + n (two basis vectors). The method is not broken. On the model's
q_proj the SVD edges it out, 83.0 against 81.0 percent, and both are useless.
So the weights are only marginally more concentrated than noise.

**Verdict: dead.**

---

## 4. Convolution structure (the case where FFT wins)

If W is Toeplitz or circulant, the Fourier basis *diagonalizes* it: W x becomes
an elementwise product, **O(n log n)** instead of **O(n^2)**. That is the one
place a frequency-domain transform is a large win, and it needs no
training-time change if the weights already have the structure.

**Test.** The best Toeplitz approximation in Frobenius norm averages W along
each diagonal. Measure the residual.

    attn.q_proj          residual about 100.0%
    expert0 gate_up      residual about 100.8%
    random control       residual about 99.9%

The model's matrices are not measurably more convolutional than noise.

**Verdict: dead.**

---

## Why this happens, and what it implies

Two structural facts, in plain terms:

1. **The training fills the space it is given.** Gradient descent on a dense
   parameterization has no reason to leave a matrix low-rank, and every reason
   not to: a full-rank solution fits the data better. QAT then squeezes what is
   left without changing the rank. Structure has to be *imposed* and lived in
   during training, not recovered afterwards.
2. **A fixed transform cannot beat the SVD per term.** Because separable basis
   elements are rank-1, a k-term transform representation is a matrix of rank
   at most k, and the SVD is the optimal rank-k approximation. So when the SVD
   spectrum is flat, no DCT, FFT or wavelet basis can concentrate it. The
   measured SVD spectrum is therefore a certificate for all of them at once.

The methods that do get large wins from this family all earn it in training:
FNet (attention replaced by an unparameterized Fourier transform, 92 to 97
percent of BERT accuracy), Hyena (implicit long convolutions through the FFT),
Monarch and butterfly matrices (O(n sqrt(n))), Toeplitz neural networks.
Post-hoc conversion of a densely trained model to any of them destroys it.

The frequency idea is already present in this model where it pays: **RoPE**
encodes position by rotating dimension pairs at a geometric ladder of
frequencies, and the global layers use a proportional variant in which only 64
of 256 angle pairs rotate. That is a frequency-domain construction, and it
works because the architecture imposes it and training goes through it.

---

## What is left

| approach | status |
|---|---|
| bit width (4-bit, QAT) | banked, about 8x on bytes, small quality cost |
| prefix reuse | **proven, 74 percent on a 4-turn conversation, zero quality cost** |
| low-rank within a matrix | dead (section 1) |
| shared basis across experts | dead (section 2) |
| transform coding | dead (section 3) |
| convolution / Toeplitz | dead (section 4) |
| distillation into a smaller dense model | the only real 5 to 10x reduction in calculations, needs a training pipeline |

If the goal is fewer calculations rather than fewer bytes, the two live options
are to find redundancy in the **workload** (prefix reuse, already measured) or
to spend compute on **distillation**. Nothing in the weights is available for
free.

---

## Caveats

* These are **weight-space** errors, not functional ones. A 5 percent weight
  error may be benign or fatal; the measurements are a gate, not a verdict.
  Errors of 60 to 99 percent are safely beyond any functional tolerance.
* The matrices were read from the Q4_0 GGUF and dequantized. The spectra
  therefore include quantization noise of roughly 1 to 2 percent, which lifts
  the tail. The true spectra are slightly steeper than measured, but not by the
  factors these conclusions would need.
* Two layers, one expert, and a representative set of matrices were sampled.
* The random control uses Gaussian entries.

---

## Reproduce

    cd numpy-gemma
    PYTHONPATH=. OMP_NUM_THREADS=18 python scripts/check_spectrum.py
    PYTHONPATH=. OMP_NUM_THREADS=18 python scripts/check_expert_basis.py
    PYTHONPATH=. OMP_NUM_THREADS=18 python scripts/check_freq.py

Related: the performance side of the same question is in the README section
"Prefill experiments", and the engineering record of the runtime is in
pipeline.html.
