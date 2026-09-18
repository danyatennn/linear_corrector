# Distribution-Free Selective Correctors for Recommenders under MNAR Feedback

A **selective corrector** wraps an existing recommender. For each recommended item it decides **accept** (show it) or **reject** (abstain), backed by a finite-sample, distribution-free guarantee on the resulting error rate — valid for any sample size, with no assumptions on the data distribution.

This extends the AI-error-corrector framework of Tyukin et al., *"Coping with AI Errors with Provable Guarantees"* (Information Sciences, 2024) — a 1-D score projection, an abstain/accept rule, and Dvoretzky–Kiefer–Wolfowitz (DKW) confidence bounds — to the regime that actually occurs in recommender systems: **MNAR implicit feedback**, where the probability an item is *observed* correlates with the model's own score *within* each relevance class.

## Three results

1. **Propensity-corrected DKW bound.** A variance-aware, martingale-based tail replaces the plain DKW tail, giving a valid guarantee even though relevance is inferred from biased clicks rather than clean labels.
2. **Soft propensity weights.** Instead of hard "clicked / not clicked" sets, every recommended item gets a signed weight (`C/e`, `1 - C/e`). Under a standard examination model this makes both the relevant- and irrelevant-item score distributions identifiable from clicks alone.
3. **Precision@K decomposition.** Precision splits into base-rate-independent conditional rates times the (movable) base rate, so the guarantee can be recalibrated after a popularity/base-rate drift without retraining.

All three are validated on a full-size **SASRec** ranker trained on **Amazon Movies & TV**, under a semi-synthetic MNAR observation model, against conformal risk control (De Toni et al., RecSys 2025) as a baseline.

## Install & run

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/prepare_data.py     # k-core, labels, sequences -> data_cache/
.venv/bin/python scripts/train_sasrec.py     # train SASRec, cache projection scores
.venv/bin/python scripts/run_all.py          # run every experiment, save figures/
```

Place the raw Amazon Movies & TV review dump (`Movies_and_TV.csv.gz`) in the repo root before running `prepare_data.py`. Each experiment can also be run on its own, e.g. `.venv/bin/python experiments/exp1_decisive.py`. Tests: `.venv/bin/pytest -q`. Dataset size, model architecture, and the `Δ` / `e_min` grids are all set in `config.py`.

## Layout

```
config.py                 central configuration (data / model / experiment grids)
recsys_guarantees/         corrector library
  bounds.py                 DKW bounds + variance-aware propensity-corrected bounds
  cdf.py                     weighted empirical CDF, pseudo-inverse, isotonic projection
  corrector.py               Algorithm 1 (Type-I) & Algorithm 2 (Type-II)
  precision.py                precision@K decomposition, risk-coverage
  propensity.py               propensity model + noisy/mis-specified variant
  simulate.py                  semi-synthetic MNAR observation, base-rate reshaping
  conformal.py                  De Toni et al. conformal risk control baseline
  data.py, sasrec.py             data loading and the SASRec ranker
scripts/                   data prep, training, end-to-end orchestration
experiments/               exp0 .. exp8, each producing one figure
tests/                     unit tests
```
