# IRM vs. ERM on Colored MNIST

> Invariant Risk Minimization recovers out-of-distribution generalization by learning representations that discard spurious color features. Test accuracy jumps from ~12% (ERM) to ~66% (IRM).

<!-- Key result figure will be added after experiments -->

## What & Why

Standard models (ERM) exploit spurious correlations that happen to hold in training data, like digit color correlating with class label. When this correlation reverses at test time, accuracy collapses below chance. [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893) (Arjovsky et al., 2019) penalizes representations where the optimal predictor differs across environments, forcing the model to rely only on invariant (causal) features like digit shape.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Build dataset and visualize samples
python data.py

# Train ERM baseline
python train.py --mode erm

# Train with IRM penalty
python train.py --mode irm

# Run representation probes (after training both)
python probe.py
```

## Results

*Coming after experiments are run.*

## Notes & Limitations

*Coming after experiments are run.*

## References

- Arjovsky et al., [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893), 2019
- Gulrajani & Lopez-Paz, [In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434), 2021
