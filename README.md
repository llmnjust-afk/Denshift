# denshift

Source-free, gradient-free test-time adaptation for 3D point clouds under
density shift.

A frozen point-cloud classifier sees fewer and fewer points as a sensor is
throttled, a scene empties, or a scanner is swapped. Its logits move, and they
move along a direction the model learned from the *sampling rate* of its
training data rather than from shape. This repository asks a narrow question:
**can that density-induced logit offset be measured and removed without labels,
without source data, and without gradients through the model?**

The answer this code currently supports is a qualified yes: the offset is
measurable from paired views of the same cloud, but only along a
low-dimensional subspace that the intervention actually excites. That
qualification turns out to be the interesting part.

## Status

Research code, pre-publication. The library is stable and tested; the headline
result is not yet strong enough for a main-track submission. Numbers below are
reproducible with `python scripts/smoke_test.py`.

## Method

Write the frozen model's logits on a cloud `x` as

```
z(x) = s(x) + B . phi(d(x))
```

where `s` is the semantic score we want, `d` is a density descriptor computed
from the input alone, and `phi` is a small basis expansion. `B` is what the
model silently memorised from its training density.

Take two views of the *same* cloud at different sampling rates. The semantic
content is identical, so the logit difference is pure density nuisance:

```
z(x_dense) - z(x_sparse) = B . (phi(d_dense) - phi(d_sparse))
```

That is a linear regression with no labels, no source data, and no gradients
through the model. It is solved by ridge regression on accumulated sufficient
statistics, so it streams in O(1) memory. Inference subtracts the fitted bias:

```
z_hat(x) = z(x) - alpha * B . phi(d(x))
```

The basis is a second-order expansion of the six-dimensional descriptor. The
constant column of `phi` is not identifiable from paired differences, but the
resulting ambiguity shifts every class by the same amount and therefore leaves
`argmax` unchanged. Centring the correction across samples subtracts a
*class-dependent* vector instead, and measurably hurts; the estimator does not
do it.

## What the evidence says

Four things were measured, and three of them are negative.

**Resampling the input does not work.** The obvious alternative — normalise
every test cloud to the source density before feeding it to the model — loses
9.5 to 14.5 points across every regime. Resampling perturbs the shape
statistics the classifier depends on, so the cure damages the very signal it is
meant to protect. The correction has to happen in logit space.

**The nuisance is only partly identifiable.** Reproduce with
`scripts/diagnose_identifiability.py`. The paired design reaches rank 19 of 21
basis directions with a condition number of 230 (decoupled) to 330 (hard) on
the identified subspace. The fit explains 88% of the pair variance, yet the
recovered coefficient is 66 to 72% wrong in absolute space
(`||Phi B - f|| / ||f|| = 0.66`), under-scaling by about 30% per class, with a
per-sample cosine of 0.48 to 0.55. Error on the differences does not imply a
correct coefficient away from the subspace the intervention explored.

**Truncation is the fix, and it is small.** Restricting the correction to the
*top two* directions of the pair Gram matrix is the only setting that helps
(0.910 against a 0.895 baseline on the decoupled stream); from k=3 onward
accuracy falls monotonically, reaching 0.850 when the full coefficient is used.
So the honest claim is not "we remove the bias" but "we remove the part of it
that the data identifies".

**What it buys.** With truncation the correction is near-neutral in the mild
regime, gains 1 to 2 points when density is decoupled from class, and loses 3
to 9 points when the entire density marginal shifts. The last case is not a bug
to be tuned away; it is the regime where the assumption behind the estimator is
false.

### Synthetic harness

`seeds 0 and 1, 150 samples, ten classes`

| adapter | none | decoupled | decoupled_hard | global |
|---|---|---|---|---|
| source_only | 1.000 | 0.933 | 0.853 | 0.753 |
| entropy_bias | 1.000 | 0.933 | 0.853 | 0.753 |
| global_bias | 1.000 | 0.927 | 0.853 | 0.753 |
| dsc | 0.933 | 0.920 | 0.887 | 0.613 |
| dsc_top2 | 0.987 | 0.933 | 0.873 | 0.720 |

| adapter | none | decoupled | decoupled_hard | global |
|---|---|---|---|---|
| source_only | 1.000 | 0.940 | 0.907 | 0.807 |
| dsc | 0.973 | 0.893 | 0.960 | 0.520 |
| dsc_top2 | 1.000 | 0.933 | 0.920 | 0.713 |

`dsc_top2` is the recommended configuration: it gives up most of `dsc`'s gain on
the hard regime in exchange for never falling far below the frozen baseline
elsewhere.

`decoupled` draws density independently of class while preserving the marginal;
`decoupled_hard` widens that gap; `global` shifts the whole marginal, which is
where the estimator breaks.

## Install

```
python3 -m pip install -r requirements.txt
```

`torch` is only needed for the black-box wrapper and the real-data backends; the
calibration core is pure NumPy.

## Quick start

```
python3 scripts/smoke_test.py --n-samples 150 --seed 0
```

or, in a notebook:

```python
from denshift import DensityBiasModel, density_descriptor, stratified_subsample

model = DensityBiasModel(n_classes=10, warmup=32)
for cloud, logits in stream:
    sparse = stratified_subsample(cloud, ratio=0.5, bins=4, rng=rng)
    model.observe_pair(
        density_descriptor(cloud), density_descriptor(sparse),
        logits, frozen(sparse),
    )
    corrected = model.correct(logits, density_descriptor(cloud))
```

## Layout

```
denshift/geometry.py          density descriptor, subsampling
denshift/streaming.py         running standardiser, streaming quantiles
denshift/calibrate.py         the estimator
denshift/changepoint.py       residual-ratio shock detector, descriptor detector
denshift/adapter.py           adapters: source-only, entropy, global bias, DSC ablations
denshift/blackbox.py          frozen-model wrapper that freezes and audits
denshift/metrics.py           accuracy, mCE, mRR, ECE
denshift/synthetic.py         controlled testbed and benchmark driver
denshift/backends/dgcnn.py    DGCNN matching the PointCloud-C checkpoints
denshift/data/modelnet_c.py   ModelNet-C split discovery and loading
scripts/                      smoke test, identifiability diagnostic, ModelNet-C eval
```

## Running on ModelNet-C

Download the merged release from `ldkong1205/PointCloud-C` and point the eval
script at it. The DGCNN weights from that repository load directly:

```
python scripts/eval_modelnet_c.py \
    --root data/ModelNet40-C \
    --checkpoint weights/dgcnn_modelnet40.t7 \
    --corruptions add_global jitter scale rotation \
    --severity 1 2 3 4 5 \
    --adapter dsc_top2 \
    --n-samples 200
```

The script prints accuracy per split and mCE against the clean split. Budget
roughly a second per cloud on CPU.

Two cautions. ModelNet-C and Robo3D normalise mCE differently, so never place
their numbers in the same table. Robo3D's baseline protocol also forbids
test-time adaptation, so its published numbers are not comparable to anything
here without re-running the baseline.

## Findings worth keeping

Two measurement bugs changed the conclusions of earlier runs and are now
covered by tests.

Fitting a classifier and reading `coef_` without `intercept_` silently destroys
the semantic pathway: the same features score 0.99 with the intercept and 0.08
without. Every "the features are degenerate" conclusion drawn that way was an
artefact.

Comparing a dense cloud with a subsample of itself changes the *shape*
descriptor as well as the density. With radial-quantile descriptors the drift is
3.3% under a 2x subsample, which is enough to make the paired regression
attribute shape drift to density. Including the shape difference as a covariate
raises the fit to R^2 = 1.0, which is exactly how the confounding was found.

## Citation

If you use this code, please cite the paper once it exists. Until then, point at
the repository.

## License

MIT. See `LICENSE`.
