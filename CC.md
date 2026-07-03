# CLAUDE.md — project handoff (SNN paper + evaluation code)

## What this is
Code + LaTeX for "Subjective Neural Networks: Trust-Aware Bayesian Dropout with a
Principled Uncertainty Decomposition" (UAI resubmission; prior reviews rejected
#283). Merges two earlier drafts (full SNN paper + frozen-backbone Subjective
Head paper). Python package: `snn_eval/`. Paper: `snn_unified.tex` (plain
article class; port to UAI style later; `[fill]`/`\todo` cells need run numbers;
`%VERIFY` citations need checking).

## Method (current, post-pivot #2: BALD/MI opinion)
SNN = Beta-Bernoulli dropout: per-unit trust p_j~Beta(α_j,β_j), Bernoulli mask;
Kumaraswamy reparam + Concrete relaxation (explicit gradient path — reviewers
flagged this). Inference = nested sampling: Np outer trust draws × Nm inner
masks. **The opinion is now the BALD/mutual-information opinion
(`augmented.py:augmented_opinion`), ported from the dirichlet_playground_lastv2.html
Dirichlet-tab `computeOpinionBALD`, REPLACING the earlier law-of-total-variance
(varE/eVar) formulation** (2026-06-16): per-trust-sample raw frequencies
rawMean_i=counts_i/Nm (no prior, entropy input) and prior-smoothed means
mean_i=(counts_i+prior)/S (belief-direction input); H_total=H(E_i[rawMean_i])
(total predictive entropy), eH=E_i[H(rawMean_i)] (aleatoric), MI=max(0,H_total-eH)
(epistemic, BALD score); u*=MI/H_total (epistemic SHARE); b=(1-u*)·E_i[mean_i];
P=b+u*/K; S*=clip(K/(eH/log K), K, 1e4); α*=P·S*. Use prior=1/K. mode="soft" for
calibrated P, "counts" for opinion semantics. Dict keys `u_e`/`u_a` now alias
MI/eH (no longer variance sums) for backward-compat with rotation-sweep plots.

## Key findings so far
**CAUTION — items 1-3 below were measured under the PRE-2026-06-16 variance/LoTV
formula (u*=ΣvarE/(ΣvarE+ΣeVar)). The opinion mechanism has since been replaced
with the BALD/MI formula above; sim_units regime ordering re-validated post-swap
(aleatoric u=0.056 << epistemic_u u=0.769, trusted/distrusted both low — see
`python -m snn_eval.sim_units` output), but the specific OOD-AUROC/rotation
numbers below need a re-run before citing in the paper.**
1. Prop. 1: old u=K/(S+K) is epistemic-only → collapses on OOD for confident
   heads (measured: OOD-AUROC 0.17–0.23 for old u vs 1.00 for H/neg_b/u*).
2. Playground HTML bug found (now moot — variance formula retired): its eVar used
   Dirichlet POSTERIOR variance, which shrinks 1/Nm like the noise in varE → u
   saturated ~0.5–0.75 even in pure aleatoric regime. Was fixed with categorical
   variance before the BALD swap; BALD sidesteps the bug class entirely since it
   never touches per-sample posterior variance.
3. Real-MNIST rotation sweep (user-run, pre-BALD): acc .97→.14 at 90°; u_e ×10 vs
   u_a ×2 → epistemic share tracks covariate shift; partial recovery at 180°.
   Narrative: H is the better detector, u* is the better EXPLAINER. Re-run needed
   under BALD u_e=MI / u_a=eH to confirm the same trend holds.
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
1. Re-run run_mnist (now BALD opinion) with EDL fix + soft-P rows; re-confirm
   EDL ~98%, SNN(mean,H) OOD ~0.92, soft-P ECE near 0.006 under the new u*/α*.
2. Per-digit rotation trajectories (single rotated "1"/"9", old Fig. 2/3 style)
   — small variant of run_mnist sweep.
3. Swap augmented_opinion into run_exp2 (abstention: use u* and α*) and
   run_exp3 (fusion consumes α* directly). u*/α* are now BALD-derived.
4. Frozen-DINOv2 CIFAR runs (rung 2), 10 seeds; fill paper tables.
5. GPU full-net runs (rung 3) incl. baselines on same recipe.
6. Paper: port to UAI style; update Exp sections to the BALD/MI estimator
   everywhere (replace all LoTV/varE/eVar wording with H_total/eH/MI); verify
   %VERIFY citations (Kwon2020, Depeweg2018, DINOv2, CLIP); rewrite Prop. 1
   wording to match the entropy-based implementation (no more "varE across
   Beta samples").
7. Optional: end-to-end pixel attacks (attacks.py currently perturbs head
   inputs); run_all.py emitting LaTeX tables; per-layer regime histograms.

## Reviewer-objection → fix map (for the rebuttal)
Gradient path unspecified → Concrete relaxation, explicit (§Method).
Bayesian-over-weights overclaim → stated: Bayesian over trust probs only.
Efficiency false claim → frozen backbone + wall-clock reporting.
Undertrained baselines → modern recipe, same arch for all methods; EDL annealed.
Missing baselines → Deep Ensembles, last-layer Laplace added (SNGP/DDU need
spectral-norm backbones; noted out of scope).
Novelty vs Lee et al. → contribution = BALD/MI-grounded augmented opinion +
unit-regime mechanism + abstention/fusion operations, not the dropout itself.
Kumaraswamy gap unquantified → TV/KL grid in appendix (still to run).
