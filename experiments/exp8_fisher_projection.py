import _common  # noqa: F401  (sets sys.path so config / recsys_guarantees import)

import os

import numpy as np
import torch

from config import DATA, SASREC
from recsys_guarantees import data, sasrec
from recsys_guarantees import plotting

from scripts.train_sasrec import _shift_splits, _build_pools


@torch.no_grad()
def feature_pairs(model, histories, candidates, maxlen, device, batch_size=512):
    out = np.empty((len(candidates), model.d), dtype=np.float32)
    for start in range(0, len(candidates), batch_size):
        end = min(start + batch_size, len(candidates))
        seqs = np.stack([sasrec._left_pad(histories[j], maxlen) for j in range(start, end)])
        rep = model.seq_repr(torch.from_numpy(seqs).to(device))                 # (B, d)
        cand = torch.from_numpy(np.asarray(candidates[start:end], dtype=np.int64)).to(device)
        cand_emb = model.item_emb(cand)                                         # (B, d)
        out[start:end] = (rep * cand_emb).cpu().numpy()
    return out


def fisher_lda(Phi, y, ridge=1e-3):
    Phi = np.asarray(Phi, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    mu_p, mu_n = Phi[y].mean(0), Phi[~y].mean(0)
    Sw = np.cov(Phi[y], rowvar=False) * (y.sum() - 1) \
        + np.cov(Phi[~y], rowvar=False) * ((~y).sum() - 1)
    Sw /= (len(y) - 2)
    Sw += ridge * np.trace(Sw) / Sw.shape[0] * np.eye(Sw.shape[0])              # stabilise
    w = np.linalg.solve(Sw, mu_p - mu_n)
    return w


def auc(score, y):
    score, y = np.asarray(score), np.asarray(y).astype(bool)
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(len(score))
    n_p, n_n = int(y.sum()), int((~y).sum())
    return float((ranks[y].mean() - (n_p - 1) / 2) / n_n)


def lr_table(score, R, deltas):
    rel, irr = score[R == 1], score[R == 0]
    pi = float((R == 1).mean())
    rows = []
    for D in deltas:
        theta = np.quantile(irr, D)
        p_acc_err = float((irr > theta).mean())
        p_acc_good = float((rel > theta).mean())
        lr = p_acc_good / max(p_acc_err, 1e-9)
        post = (pi / (1 - pi)) * lr
        rows.append((D, 1 - p_acc_good, 1 - p_acc_err, lr, post / (1 + post)))
    return rows


def oracle_curve(score, R, n_points=40):
    th = np.quantile(score, np.linspace(0.0, 0.98, n_points))
    cov, prec = [], []
    n = len(score)
    for t in th:
        acc = score > t
        if acc.sum() == 0:
            continue
        cov.append(int(acc.sum()) / n)
        prec.append(float(R[acc].mean()))
    return np.array(cov), np.array(prec)


def main():
    cache = data.load_cache(DATA)
    splits = _shift_splits(data.leave_last_out(data.build_sequences(cache["df"])))
    cal, test = _build_pools(splits)

    device = sasrec.resolve_device(SASREC.device)
    model = sasrec.SASRec(cache["n_items"], SASREC).to(device)
    ckpt = torch.load(os.path.join(DATA.cache_dir, "sasrec.pt"), map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    Phi_cal = feature_pairs(model, cal["histories"], cal["candidates"], SASREC.maxlen, device)
    Phi_test = feature_pairs(model, test["histories"], test["candidates"], SASREC.maxlen, device)
    R_cal = np.asarray(cal["R"]); R_test = np.asarray(test["R"])

    h_raw = Phi_test.sum(1)
    w = fisher_lda(Phi_cal, R_cal)
    h_fisher = Phi_test @ w
    if auc(h_fisher, R_test) < 0.5:
        h_fisher = -h_fisher

    auc_raw, auc_fish = auc(h_raw, R_test), auc(h_fisher, R_test)
    print(f"[exp8] AUC(test):  raw score = {auc_raw:.3f}   Fisher LDA = {auc_fish:.3f}")

    deltas = [0.5, 0.7, 0.9, 0.95]
    for tag, sc in [("raw   ", h_raw), ("fisher", h_fisher)]:
        print(f"\n[exp8] {tag}  {'Delta':>5} {'P(rej|err)':>10} {'P(rej|good)':>11} "
              f"{'LR':>5} {'precision':>9}")
        for D, prej_good, prej_err, lr, prec in lr_table(sc, R_test, deltas):
            print(f"{'':14}{D:6.2f} {prej_err:10.3f} {prej_good:11.3f} {lr:5.2f} {prec:9.3f}")

    cov_r, prec_r = oracle_curve(h_raw, R_test)
    cov_f, prec_f = oracle_curve(h_fisher, R_test)
    path = plotting.plot_risk_coverage(
        curves={
            f"raw SASRec score (AUC {auc_raw:.2f})": (cov_r, prec_r, "-"),
            f"Fisher LDA projection (AUC {auc_fish:.2f})": (cov_f, prec_f, "-"),
        },
        title="Fisher LDA projection separates relevance -> precision the corrector can lift",
        name="exp8_fisher.png")
    print(f"\n[exp8] saved -> {path}")


if __name__ == "__main__":
    main()
