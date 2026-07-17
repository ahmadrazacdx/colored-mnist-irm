# IRM vs ERM on Colored MNIST

![ERM vs IRM](figures/erm_vs_irm_accuracy.png)

ERM memorizes the spurious color shortcut and collapses to **29% test accuracy**. IRM recovers to **66%** by penalizing features that shift across environments.

## Background

[Colored MNIST](https://arxiv.org/abs/1907.02893) is a synthetic benchmark designed to isolate one question: can a model tell apart causal features from spurious correlations? Each MNIST digit is colored red or green. In the training environments, color strongly predicts the label — but at test time, that correlation flips. A model that relies on color will fail. A model that learns shape will generalize.

This repo reproduces the core experiment from Arjovsky et al. (2019) and adds representation probing to show what each model actually learned internally. Everything runs in under 2 minutes on a free Colab T4.

## Setup

```bash
git clone https://github.com/<user>/colored-mnist-irm.git
cd colored-mnist-irm
pip install -r requirements.txt
```

## Data

Binary MNIST (digits 0–4 → class 0, digits 5–9 → class 1) with 25% label noise. Each digit is colored red or green, where color correlates with the noisy label at different strengths per environment:

| Environment | Samples | Color–Label Correlation |
| --- | --- | --- |
| Env1 (train) | 25,000 | 90% |
| Env2 (train) | 25,000 | 80% |
| Test | 10,000 | 10% (reversed) |

```bash
python data.py
```

![Sample Environments](figures/sample_envs.png)

The 25% label noise is important — it caps the Bayes-optimal accuracy at 75% even for a perfect shape-based classifier. This prevents trivial solutions and makes the spurious shortcut (color at ~85% pooled correlation) genuinely tempting for the optimizer.

## Model

2-hidden-layer MLP with 390 units each (ReLU activations), matching the original IRM paper. Input is a 2-channel 14×14 image (downsampled MNIST, one channel per color). Output is a single logit for binary classification. The network is split into an `encoder` (hidden layers) and a `head` (final linear layer) — this split is built in from the start so we can extract frozen features for probing later without modifying anything.

## Training

```bash
python train.py --mode erm
python train.py --mode irm
```

Both train over 3 random seeds and report mean (std).

**ERM** (Empirical Risk Minimization) pools all training data and minimizes cross-entropy. It has no mechanism to detect which features are spurious — it simply follows the strongest gradient signal, which is color.

**IRM** (Invariant Risk Minimization) adds a per-environment penalty: for each environment, it checks whether scaling the logits by a dummy scalar w=1.0 is already optimal. If the gradient of the loss with respect to this scalar is non-zero, it means the representation encodes something environment-specific. Penalizing the squared norm of this gradient forces the model to find features where a single classifier works everywhere. The penalty is activated after 190 warmup steps of pure ERM.

### Results

| | Env1 | Env2 | Test |
| --- | --- | --- | --- |
| ERM | 97.6% (0.4%) | 96.1% (0.3%) | **29.4% (1.8%)** |
| IRM | 73.0% (1.2%) | 72.8% (1.1%) | **66.2% (0.1%)** |

ERM achieves near-perfect training accuracy by exploiting color. This backfires at test time where color is anti-correlated with the label — the model is confidently wrong. IRM sacrifices ~25 points of training accuracy but gains genuine out-of-distribution generalization, approaching the 75% shape-only ceiling. The standard deviation on IRM's test accuracy is just 0.1%, meaning the result is highly stable across seeds.

### Training Dynamics

![Training Dynamics](figures/training_dynamics.png)

The phase transition at step 190 is the most informative part of this experiment. During warmup (steps 1–190), IRM behaves identically to ERM — both latch onto color, and test accuracy declines. The moment the invariance penalty activates, IRM's test accuracy reverses direction and climbs sharply while its training accuracy drops. This is the trade-off at work: the model is abandoning a feature that helps on the training distribution but hurts on the test distribution.

## Representation Probing

```bash
python probe.py
```

I extracted the 390-dimensional features from the penultimate layer of both trained models and fit logistic regression probes to classify either the spurious feature (color) or the causal feature (digit class).

| Probe Target | ERM Features | IRM Features |
| --- | --- | --- |
| Color (spurious) | 100.0% (0.0%) | 100.0% (0.0%) |
| Digit (causal) | 85.6% (0.3%) | 74.3% (0.1%) |

![Probe Results](figures/probe_results.png)

Both representations retain perfect color information — a linear probe decodes color at 100% from both ERM and IRM features. This is expected and worth discussing: IRM-v1 is a **predictor alignment** method, not an information bottleneck. It makes the classifier head ignore color, but it doesn't force the representation to erase it. The color information is still present in the 390-dimensional space; it's just orthogonal to the prediction direction. A method like DANN (domain-adversarial training) would actively try to remove color from the representation, which is a fundamentally different approach.

The digit probe confirms that IRM's representation retains causal shape information (74.3%), closely matching its actual test accuracy. ERM's digit probe is higher (85.6%) because its representation encodes everything aggressively — both color and shape — but it uses the wrong one at test time.

## Notes & Limitations

- **IRM stability.** IRM-v1 was remarkably stable here (test std = 0.1%). This is not always the case. [Gulrajani & Lopez-Paz (2021)](https://arxiv.org/abs/2007.01434) showed that IRM can be highly sensitive to hyperparameters on more complex benchmarks like DomainBed. The stability we observe is partly due to the simplicity of Colored MNIST.

- **Gap from shape ceiling.** IRM reaches ~66% vs the theoretical 75% ceiling. This gap comes from the interaction between 25% label noise and IRM's optimization dynamics. Longer training, learning rate tuning, or stronger penalty weights could close it.

- **The 100% color probe.** The fact that IRM's representation perfectly encodes color while ignoring it for prediction is a real finding, not a failure. It tells us that IRM-v1 operates at the level of the classifier, not the representation. Whether this is desirable depends on the use case — in safety-critical settings, you might want the representation itself to be scrubbed of spurious features (which would require a different method).

## References

- Arjovsky, M., Bottou, L., Gulrajani, I., & Lopez-Paz, D. (2019). *Invariant Risk Minimization.* [arXiv:1907.02893](https://arxiv.org/abs/1907.02893)
- Gulrajani, I., & Lopez-Paz, D. (2021). *In Search of Lost Domain Generalization.* ICLR 2021. [arXiv:2007.01434](https://arxiv.org/abs/2007.01434)
