# SNN evaluation code

Reference implementation for **Subjective Neural Networks (SNN): Trust-Aware
Bayesian Dropout with a Principled Uncertainty Decomposition**. Implements the
SNN head (Beta–Bernoulli dropout with Kumaraswamy reparameterisation and a
Concrete-relaxed mask), modern UQ baselines, the law-of-total-variance (LoTV)
decomposition, and the three experiments.

## Install
```bash
pip install -r requirements.txt          # torch, numpy, scipy
# torchvision / timm only needed for real frozen-backbone features
```

## Two backbone modes
- `--backbone synthetic` — controlled features that mimic post-LayerNorm
  penultimate states; runs anywhere with no downloads. Use for sanity checks.
- `--backbone dinov2_vits14` (or a `timm` model name) — extracts **frozen**
  foundation-model features once for CIFAR-10/100 / SVHN. Only the small SNN
  head is trained, so this is laptop-feasible. First run downloads weights/data.

## Run the experiments
```bash
# Exp 1: predictive / calibration / OOD + decomposition (u vs H vs neg_b) + LoTV
python -m snn_eval.run_exp1 --backbone dinov2_vits14 --dataset cifar10 --ood svhn

# Exp 2: two-threshold abstention frontier vs 1D rules, multi-seed + CIs
python -m snn_eval.run_exp2 --seeds 10

# Exp 3: trust-discounted SL fusion across reliable + adversarial sources
python -m snn_eval.run_exp3
```

## What maps to what in the paper
- `models.SubjectiveHead` — the SNN head; `sample_p` is Eq. (Kumaraswamy),
  `sample_mask` is the Concrete relaxation (Eq. Concrete), `kl` is the
  closed-form Beta–Beta KL. `train_head(..., is_snn=True)` is the ELBO with
  KL warmup and `beta_max`.
- `inference.snn_nested_samples` + `sl_signals` — nested sampling, Dirichlet
  moment matching, the three signals `u, H, neg_b`, and the LoTV trace split
  (`aleatoric`, `epistemic`, `total`). This is where Proposition 1 is checked.
- `fusion.py` — SL cumulative / averaging / trust-discounted operators + scalar
  baselines (Exp 3).
- `laplace.LastLayerLaplace` — diagonal-GGN last-layer Laplace baseline.
- `attacks.py` — FGSM / PGD on head inputs (see caveat below).

## Diagnostics and known sensitivities (read before trusting numbers)
- **Dead-mechanism check.** After SNN training the code prints
  `std(E[p])`. If it is `< 0.01`, the variational posterior collapsed (the
  loss-balance pitfall). Raise `--beta_max`, shrink `--d_hidden`, or use an
  informative prior. On the trivially separable synthetic data this value is
  *expected* to be tiny because masks need not vary; it engages on real
  features and in the train-on-easy extrapolation setting of Exp 2.
- **Exp 1 reproduces the headline cleanly** even on synthetic data: `SNN (u)`
  OOD-AUROC collapses while `SNN (H)` and `SNN (neg_b)` stay high, and the LoTV
  split shows OOD epistemic stays low while OOD aleatoric rises — i.e. `u`
  tracks epistemic only (Prop. 1).
- **Exp 2 significance is config-dependent.** The 2D frontier reliably beats
  `u`, but whether it *significantly* beats `neg_b` (the paper's `p<0.01`
  claim) depends on regime construction, `beta_max`, bottleneck width, and seed
  count. Re-establish this on your real features with the full 10 seeds before
  citing the p-values; the smoke test with 3 synthetic seeds does not settle it.
- **Exp 3 trust-discounted fusion** preserves accuracy where logit averaging
  collapses, but its NLL can be sensitive to the opinion→η reconstruction when
  concentrations get extreme; the headline metric is accuracy preservation.
- **Adversarial caveat.** With a frozen backbone, `attacks.py` perturbs the
  penultimate features, a relaxation of pixel-space attacks. For end-to-end
  attacks, backprop through the frozen backbone; the head API is unchanged.

## Reproducing the paper tables
Each `run_expN.py` prints the table for its section. Wrap `run_exp1` in a loop
over seeds and datasets to fill Tables 1–2; `run_exp2`/`run_exp3` already
aggregate. Cache feature tensors (the `cache=` arg in `data.extract_features`)
so the frozen backbone runs only once per dataset.

## Route 2: SNN as a FULL network (ResNet-18 / WRN-28-10)

Trains the SNN as a genuine Bayesian-dropout network so the trust variables live
*throughout* the backbone (channel-wise Beta-Bernoulli dropout), not just in a
head. This defends the "neuron-level trust" claim and, with a modern recipe,
fixes the undertrained-baseline objection.

Files:
- `backbones.py` — `StochasticDrop2d` (channel-wise Beta-Bernoulli / Dropout2d /
  identity), SNN-instrumented `ResNet18` (CIFAR stem, drop after each block;
  11.2M params) and `WRN(28,10)` (drop in the WRN dropout slot; 36.5M params),
  plus `build_net(arch, K, method)`.
- `fullnet_inference.py` — deterministic / MC-dropout / EDL / SNN-nested
  inference over data loaders; reuses `inference.sl_signals` and `metrics`.
- `train_fullnet.py` — modern recipe: SGD + Nesterov, cosine LR, weight decay
  5e-4, label smoothing 0.1, RandomCrop+Flip+Cutout; KL warmup + `beta_max` for
  SNN; supports `snn / mcdropout / edl / deterministic / ensemble`.

```bash
# sanity check anywhere (random data, no downloads):
python -m snn_eval.train_fullnet --arch resnet18 --method snn --dataset synthetic --smoke

# real headline runs (laptop GPU):
python -m snn_eval.train_fullnet --arch resnet18 --method snn --dataset cifar10  --epochs 200
python -m snn_eval.train_fullnet --arch resnet18 --method snn --dataset cifar100 --epochs 200
python -m snn_eval.train_fullnet --arch wrn2810  --method snn --dataset cifar10  --epochs 200
# baselines on the SAME architecture/recipe (kills the "undertrained baseline" objection):
python -m snn_eval.train_fullnet --arch resnet18 --method mcdropout  --dataset cifar10 --epochs 200
python -m snn_eval.train_fullnet --arch resnet18 --method edl        --dataset cifar10 --epochs 200
python -m snn_eval.train_fullnet --arch resnet18 --method ensemble   --ensemble_size 5 --dataset cifar10 --epochs 200
```

Expected with the modern recipe (200 epochs): ResNet-18 ~95% / ~77% on
CIFAR-10 / -100; WRN-28-10 ~96% / ~81%. If your numbers are far below this the
recipe is misconfigured (the original 88.8% was undertrained).

### Tuning the SNN full-net (important)
- A full net has thousands of trust channels, so the summed KL is large
  (~6k for ResNet-18, ~14k for WRN). Use a **small `--beta_max`** (default 1e-2)
  and watch the `E[keep] std` diagnostic printed during training: if it stays
  ~0 the posterior collapsed (raise `beta_max`); if accuracy tanks, lower it.
- `--init_keep` (default 0.9): channel dropout is aggressive; keep it high.
- `--Np --Nm` (default 10/10 = 100 forward passes at test, same cost as MC
  Dropout-100). Reduce on CPU.

### Hardware notes
- ResNet-18 trains comfortably on a single laptop GPU (~1–2 h, 200 epochs) and
  runs (slowly) on CPU.
- **WRN-28-10 (36.5M params) needs a GPU.** On CPU it is impractically slow and
  may exhaust memory at batch 128 — use a GPU, or `--bs 32` for a CPU smoke test.
- SNGP / DDU require spectral normalisation inside the backbone and are not
  implemented here; they only fit Route 2 (full net), not the frozen-feature
  route. Deep Ensembles + last-layer Laplace cover the rest.

## Augmented SL opinion (NEW estimator — replaces Dirichlet MLE/moment-matching)

`augmented.py` ports the Dirichlet-playground construction exactly: per trust
sample, inner mask predictions are COUNTED into a Dirichlet (counts + prior);
then varE = Var_i[means] (epistemic), eVar = E_i[m(1-m)] (aleatoric), and

    u* = sum(varE) / (sum(varE)+sum(eVar))   # vacuity = epistemic SHARE
    b  = (1-u*) * pooled_direction ;  P = b + u*/K
    S* = mean_k max(1, P(1-P)/eVar - 1) ; alpha* = P*S*

This bakes the law-of-total-variance split INTO the opinion: it fixes the
Prop.-1 failure (old u collapsed for confident heads) by making vacuity a ratio,
and preserves aleatoric spread in the concentration S*.

**Important correction found while porting:** the playground's eVar uses the
per-sample Dirichlet *posterior* variance, which shrinks as 1/Nm exactly like
the multinomial noise in varE — so its u saturates near 0.5+ even in purely
aleatoric regimes (verify with `--Np 200 --Nm 50` in `sim_units`). The default
here uses the categorical variance E[m(1-m)] (LoTV-faithful), which restores the
clean regime separation; `aleatoric="posterior"` reproduces the playground.
Also use `prior = 1/K` (not 1) — P = b + u/K already base-rate-smooths.

### Unit-regime mechanism (validated)
`python -m snn_eval.sim_units` replicates the playground's 2-unit network
(P(0)=(1-p)p, P(1)=p, P(2)=(1-p)^2) across the five regimes and checks against
analytic LoTV. Result (u_LoTV column): trusted 0.03, distrusted 0.04,
aleatoric (a=b=10) 0.06 LOW, epistemic-U (a=b=0.1) 0.71 HIGH — confirming the
mechanism: a=b>>1 injects within-draw (aleatoric) variance, a=b<<1 injects
across-draw (epistemic) variance. Note: a=b=1 (flat) is analytically MIXED and
aleatoric-leaning, not purely epistemic.

### Evaluation ladder (small models first)
0. `python -m snn_eval.sim_units`            — 2-unit analytic validation (seconds)
1. `python -m snn_eval.run_mnist`            — full SNN MLP on MNIST pixels, OOD =
   FashionMNIST, CPU minutes; reports old vs augmented signals + learned regimes
2. `python -m snn_eval.run_exp1 --backbone dinov2_vits14 --dataset cifar10`
   — frozen-feature head (CPU-feasible); now also reports `SNN (u* aug)` rows
3. `python -m snn_eval.train_fullnet --arch resnet18|wrn2810` — full nets (GPU)

All evaluations now report BOTH the legacy moment-matched signals and the
augmented opinion, so the paper can show the diagnose->fix narrative directly.

## MNIST rotation + ECE evaluation (rung 1, extended)
`python -m snn_eval.run_mnist` now (a) trains SNN, MC Dropout and EDL on the
same MLP and prints Acc/NLL/ECE/OOD-AUROC for all three, and (b) sweeps
rotations 0..180 deg, reporting per angle: SNN u* (epistemic share), u_e, u_a,
H; the same LoTV split computed from MC Dropout's T samples; and EDL u and H.
Saves results/rotation_sweep.csv and .png. Expected on real MNIST: accuracy
falls with angle while SNN u_e (and u*) rise smoothly — the epistemic component
tracks covariate shift; u_a stays comparatively flat on ambiguous-but-familiar
angles. Flags: --rot_step, --rot_n, --T, --epochs.
