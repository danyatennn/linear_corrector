import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from config import DATA, SASREC
from recsys_guarantees import data
from recsys_guarantees import sasrec


def _shift_splits(splits):
    shifted = {}
    for u, sp in splits.items():
        shifted[u] = dict(
            train=sp["train"] + 1,
            val=(sp["val"][0] + 1, sp["val"][1]),
            test=(sp["test"][0] + 1, sp["test"][1]),
        )
    return shifted


def _build_pools(splits):
    cal = dict(users=[], histories=[], candidates=[], R=[])
    test = dict(users=[], histories=[], candidates=[], R=[])
    for u, sp in splits.items():
        val_item, val_R = sp["val"]
        test_item, test_R = sp["test"]

        cal["users"].append(u)
        cal["histories"].append(sp["train"])
        cal["candidates"].append(val_item)
        cal["R"].append(val_R)

        test["users"].append(u)
        test["histories"].append(np.append(sp["train"], val_item))
        test["candidates"].append(test_item)
        test["R"].append(test_R)
    return cal, test


def main():
    cache = data.load_cache(DATA)
    df, n_items = cache["df"], cache["n_items"]
    print(f"[train] loaded cache: {len(df):,} interactions, "
          f"{cache['n_users']:,} users, {n_items:,} items")

    sequences = data.build_sequences(df)
    splits = data.leave_last_out(sequences)
    splits = _shift_splits(splits)
    print(f"[train] usable users (>=3 interactions): {len(splits):,}")

    os.makedirs(DATA.cache_dir, exist_ok=True)
    ckpt_path = os.path.join(DATA.cache_dir, "sasrec.pt")

    if os.path.exists(ckpt_path):
        device = sasrec.resolve_device(SASREC.device)
        model = sasrec.SASRec(n_items, SASREC).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[train] loaded existing checkpoint {ckpt_path} "
              f"(epoch {ckpt.get('epoch', '?')}, val NDCG@10={ckpt.get('val_ndcg', float('nan')):.4f})"
              f" -- skipping training")
    else:
        model, device = sasrec.train_sasrec(
            n_items, splits, SASREC, verbose=True, ckpt_path=ckpt_path, ckpt_every=3)
        print(f"[train] saved checkpoint -> {ckpt_path}")

    eval_users = list(splits.keys())
    test_metrics = sasrec.evaluate(
        model, eval_users, splits, n_items, SASREC.maxlen, device, use="test")
    random_hr = SASREC.num_neg_eval and 10.0 / (SASREC.num_neg_eval + 1)
    print(f"[train] TEST  NDCG@10={test_metrics['ndcg']:.4f}  "
          f"HR@10={test_metrics['hr']:.4f}  (random HR@10 ~ {random_hr:.3f})")

    cal, test = _build_pools(splits)
    h_cal = sasrec.score_pairs(model, cal["histories"], cal["candidates"],
                               SASREC.maxlen, device)
    h_test = sasrec.score_pairs(model, test["histories"], test["candidates"],
                                SASREC.maxlen, device)
    scores_path = os.path.join(DATA.cache_dir, "scores.npz")
    np.savez(
        scores_path,
        u_cal=np.asarray(cal["users"], dtype=np.int64),
        h_cal=h_cal.astype(np.float32),
        R_cal=np.asarray(cal["R"], dtype=np.int64),
        u_test=np.asarray(test["users"], dtype=np.int64),
        h_test=h_test.astype(np.float32),
        R_test=np.asarray(test["R"], dtype=np.int64),
    )
    print(f"[train] saved corrector pools -> {scores_path}  "
          f"(cal={len(h_cal):,}, test={len(h_test):,}, "
          f"relevant share cal={np.mean(cal['R']):.3f} test={np.mean(test['R']):.3f})")


if __name__ == "__main__":
    main()
