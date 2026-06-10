# CLAUDE.md — project handoff (SNN paper + evaluation code)

## What this is
Code + LaTeX for "Subjective Neural Networks: Trust-Aware Bayesian Dropout with a
Principled Uncertainty Decomposition" (UAI resubmission; prior reviews rejected
#283). Merges two earlier drafts (full SNN paper + frozen-backbone Subjective
Head paper). Python package: `snn_eval/`. Paper: `snn_unified.tex` (plain
article class; port to UAI style later; `[fill]`/`\todo` cells need run numbers;
`%VERIFY` citations need checking).

## Method (current, post-pivot)
SNN = Beta-Bernoulli dropout: per-unit trust p_j~Beta(α_j,β_j), Bernoulli mask;
Kumaraswamy reparam + Concrete relaxation (explicit gradient path — reviewers
flagged this). Inference = nested sampling: Np outer trust draws × Nm inner
masks. **The opinion is now the AUGMENTED SL opinion (`augmented.py`), NOT the
old Dirichlet MLE/moment-matching**: per-trust-sample Dirichlets from counts+
prior; varE=Var_i[means] (epistemic), eVar=E_i[m(1-m)] (aleatoric, categorical);
u* = ΣvarE/(ΣvarE+ΣeVar) (epistemic SHARE); b=(1-u*)·pooled dir; P=b+u*/K;
S*=mean_k max(1, P(1-P)/eVar - 1); α*=P·S*. Use prior=1/K. mode="soft" for
calibrated P, "counts" for opinion semantics.

## Key findings so far (validated)
1. Prop. 1: old u=K/(S+K) is epistemic-only → collapses on OOD for confident
   heads (measured: OOD-AUROC 0.17–0.23 for old u vs 1.00 for H/neg_b/u*).
2. Playground HTML bug found: its eVar used Dirichlet POSTERIOR variance, which
   shrinks 1/Nm like the noise in varE → u saturates ~0.5–0.75 even in pure
   aleatoric regime. Fixed with categorical variance (sim_units validates:
   trusted .03 / distrusted .04 / aleatoric .06 / epistemic-U .71). Note:
   α≈β=1 (flat) is MIXED aleatoric-leaning, not purely epistemic.
3. Real-MNIST rotation sweep (user-run): acc .97→.14 at 90°; u_e ×10 vs u_a ×2
   → epistemic share tracks covariate shift; partial recovery at 180°. Narrative:
   H is the better detector, u* is the better EXPLAINER.
4. EDL baseline was broken (59% MNIST) — fixed via Sensoy-style KL annealing
   (lam = min(1, ep/10)) in models.train_head; EDL needs ≥25 epochs.
5. Exp2 caveat: 2D (max b, u) frontier reliably beats u; the "beats neg_b at
   p<0.01" claim did NOT reproduce in 3-seed synthetic smoke — re-establish on
   real features with 10 seeds before citing.

## Evaluation ladder (small → big)
0. `python -m snn_eval.sim_units` — 2-unit regime validation (seconds).
1. `python -m snn_eval.run_mnist` — SNN/MCD/EDL MLPs, ECE table, rotation sweep
   (CSV+PNG in results/), training logs on. CPU-fine. OOD=FashionMNIST.
2. `python -m snn_eval.run_exp1 --backbone dinov2_vits14 --dataset cifar10
   --ood svhn` — frozen-feature head; reports old + augmented signals.
3. `python -m snn_eval.train_fullnet --arch resnet18|wrn2810 --method snn|...
   --dataset cifar10|cifar100` — full-net SNN, modern recipe; **GPU required**
   (~0.1 steps/s on CPU = days). beta_max small (1e-2) for full nets; watch
   E[keep] std diagnostic (<0.01 = collapsed posterior).

User environment: Windows, Python 3.14, CPU-only laptop (sll venv). Use
num_workers=0. Full-net runs → Colab/Kaggle GPU.

## Open tasks (priority order)
1. Re-run run_mnist with EDL fix + soft-P rows; confirm EDL ~98%, SNN(mean,H)
   OOD ~0.92, soft-P ECE near 0.006.
2. Per-digit rotation trajectories (single rotated "1"/"9", old Fig. 2/3 style)
   — small variant of run_mnist sweep.
3. Swap augmented_opinion into run_exp2 (abstention: use u* and α*) and
   run_exp3 (fusion consumes α* directly).
4. Frozen-DINOv2 CIFAR runs (rung 2), 10 seeds; fill paper tables.
5. GPU full-net runs (rung 3) incl. baselines on same recipe.
6. Paper: port to UAI style; update Exp sections to augmented estimator
   everywhere; verify %VERIFY citations (Kwon2020, Depeweg2018, DINOv2, CLIP);
   sanity-check Prop. 1 wording vs implementation (varE across Beta samples).
7. Optional: end-to-end pixel attacks (attacks.py currently perturbs head
   inputs); run_all.py emitting LaTeX tables; per-layer regime histograms.

## Reviewer-objection → fix map (for the rebuttal)
Gradient path unspecified → Concrete relaxation, explicit (§Method).
Bayesian-over-weights overclaim → stated: Bayesian over trust probs only.
Efficiency false claim → frozen backbone + wall-clock reporting.
Undertrained baselines → modern recipe, same arch for all methods; EDL annealed.
Missing baselines → Deep Ensembles, last-layer Laplace added (SNGP/DDU need
spectral-norm backbones; noted out of scope).
Novelty vs Lee et al. → contribution = LoTV-grounded augmented opinion +
unit-regime mechanism + abstention/fusion operations, not the dropout itself.
Kumaraswamy gap unquantified → TV/KL grid in appendix (still to run).
