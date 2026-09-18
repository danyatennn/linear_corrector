import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bounds import rho, psi, rho_e, psi_e, rho_e_hoeffding, psi_e_hoeffding

FIG_DIR = "figures"

_Y_REJECT = "Probability of correct rejection of errors"
_X_CARD = r"Cardinality of the datasets $S_{-,j}$"


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_bound_panels(deltas, n_grid, e_min=None, name="exp0_bounds.png"):
    n_grid = np.asarray(list(n_grid))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, delta in zip(axes.ravel(), deltas):
        lo = np.array([rho(delta, int(n)) for n in n_grid])
        hi = np.array([psi(delta, int(n)) for n in n_grid])
        ax.fill_between(n_grid, lo, hi, alpha=0.30, color="tab:blue",
                        label=r"clean DKW $[\rho,\psi]$")
        ax.plot(n_grid, lo, color="tab:blue", lw=1.0)
        ax.plot(n_grid, hi, color="tab:blue", lw=1.0)
        if e_min is not None:
            lo_e = np.array([rho_e(delta, int(n), e_min) for n in n_grid])
            hi_e = np.array([psi_e(delta, int(n), e_min) for n in n_grid])
            ax.fill_between(n_grid, lo_e, hi_e, alpha=0.18, color="tab:red")
            ax.plot(n_grid, lo_e, color="tab:red", lw=1.0, ls="--",
                    label=rf"propensity $[\rho_e,\psi_e]$ ($e_{{\min}}={e_min}$)")
            ax.plot(n_grid, hi_e, color="tab:red", lw=1.0, ls="--")
        ax.axhline(delta, color="k", lw=0.8, ls=":")
        ax.set_title(rf"$\Delta = {delta}$")
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel(_X_CARD)
    for ax in axes[:, 0]:
        ax.set_ylabel(_Y_REJECT)
    fig.suptitle("Distribution-free rejection guarantee vs negative-pool size", y=1.00)
    return _save(fig, name)


def plot_band_width_vs_m(m_grid, delta, e_min, name="exp0_bandwidth.png"):
    m = np.asarray(list(m_grid), dtype=float)
    clean = np.array([psi(delta, int(n)) - rho(delta, int(n)) for n in m])
    vb = np.array([psi_e(delta, int(n), e_min) - rho_e(delta, int(n), e_min) for n in m])
    hoeff = np.array([psi_e_hoeffding(delta, int(n), e_min)
                      - rho_e_hoeffding(delta, int(n), e_min) for n in m])

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(m, hoeff, color="tab:red", lw=1.8, marker="^", ls="--",
            label=r"propensity, loose Hoeffding+union $\psi_e-\rho_e$")
    ax.plot(m, vb, color="tab:green", lw=2.0, marker="s",
            label=r"propensity, variance-aware $\psi_e-\rho_e$ (this work)")
    ax.plot(m, clean, color="tab:blue", lw=1.8, marker="o",
            label=r"clean DKW $\psi-\rho$")
    ax.set_xscale("log")
    ax.set_xlabel(r"Cardinality of $S_{-,j}$  (log scale)")
    ax.set_ylabel(r"Band width  $\psi - \rho$")
    ax.set_title(rf"Guarantee band width vs negative-pool size  ($\Delta={delta}$, "
                 rf"$e_{{\min}}={e_min}$)")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper right", fontsize=9)
    return _save(fig, name)

def plot_claimed_vs_achieved(x, claimed, achieved, xlabel="Δ", ylabel="rejection rate",
                             title="Claimed guarantee vs achieved",
                             name="exp1_decisive.png"):
    x = np.asarray(x)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(x, claimed, "k-", lw=2.2, marker="o", label="claimed lower bound")
    for label, payload in achieved.items():
        y, ok = payload if isinstance(payload, tuple) else (payload, True)
        ax.plot(x, np.asarray(y), lw=1.8, marker="s" if ok else "x",
                ls="-" if ok else "--", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    return _save(fig, name)


def plot_risk_coverage(curves, title="Risk-coverage", name="exp2_risk_coverage.png"):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for label, payload in curves.items():
        cov, prec = payload[0], payload[1]
        ls = payload[2] if len(payload) > 2 else "-"
        ax.plot(cov, prec, ls, lw=1.8, marker=".", label=label)
    ax.set_xlabel("Coverage (fraction of items accepted)")
    ax.set_ylabel("Precision of accepted items")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    return _save(fig, name)

def plot_violation_vs_sigma(sigmas, rates, title="Graceful degradation under mis-specified e",
                            name="exp3_misspecified.png"):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    if isinstance(rates, dict):
        for label, r in rates.items():
            ax.plot(sigmas, r, lw=1.8, marker="o", label=label)
        ax.legend(loc="best", fontsize=9)
    else:
        ax.plot(sigmas, rates, lw=1.8, marker="o", color="tab:red")
    ax.set_xlabel(r"Propensity noise $\sigma$")
    ax.set_ylabel("Guarantee violation rate")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    return _save(fig, name)


def plot_drift_bars(methods, before, after, ylabel="Achieved precision",
                    hline=None, hline_label="target",
                    before_label="before drift", after_label="after drift",
                    title="Drift robustness (base-rate halved)",
                    name="exp4_drift.png"):
    methods = list(methods)
    x = np.arange(len(methods))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.bar(x - w / 2, before, w, label=before_label, color="tab:blue")
    ax.bar(x + w / 2, after, w, label=after_label, color="tab:orange")
    if hline is not None:
        ax.axhline(hline, color="k", ls="--", lw=1.2, label=hline_label)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="best", fontsize=9)
    return _save(fig, name)


def plot_bound_width_vs_emin(emins, widths, title=r"Bound width vs propensity floor $e_{\min}$",
                             name="exp5_emin.png"):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    if isinstance(widths, dict):
        for label, wv in widths.items():
            ax.plot(emins, wv, lw=1.8, marker="o", label=label)
        ax.legend(loc="best", fontsize=9)
    else:
        ax.plot(emins, widths, lw=1.8, marker="o", color="tab:purple")
    ax.axvline(0.05, color="k", ls=":", lw=1.0)
    ax.set_xlabel(r"Propensity floor $e_{\min}$")
    ax.set_ylabel(r"Band width $\psi_e - \rho_e$")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    return _save(fig, name)
