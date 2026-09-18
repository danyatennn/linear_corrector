import json
import os

import numpy as np
import pandas as pd


def factorized_load(cfg):
    user_to_id, item_to_id = {}, {}
    u_parts, i_parts, r_parts, t_parts = [], [], [], []
    reader = pd.read_csv(
        cfg.csv_path,
        compression="gzip",
        usecols=["user_id", "parent_asin", "rating", "timestamp"],
        dtype={"user_id": str, "parent_asin": str,
               "rating": "float32", "timestamp": "int64"},
        chunksize=2_000_000,
    )
    for chunk in reader:
        u_parts.append(np.fromiter(
            (user_to_id.setdefault(u, len(user_to_id)) for u in chunk["user_id"]),
            dtype=np.int64, count=len(chunk)))
        i_parts.append(np.fromiter(
            (item_to_id.setdefault(it, len(item_to_id)) for it in chunk["parent_asin"]),
            dtype=np.int64, count=len(chunk)))
        r_parts.append(chunk["rating"].to_numpy(np.float32))
        t_parts.append(chunk["timestamp"].to_numpy(np.int64))
    return pd.DataFrame({
        "u": np.concatenate(u_parts),
        "i": np.concatenate(i_parts),
        "rating": np.concatenate(r_parts),
        "ts": np.concatenate(t_parts),
    })


def k_core_filter(df, k):
    while True:
        uc = df["u"].value_counts()
        ic = df["i"].value_counts()
        keep_u = uc.index[uc >= k]
        keep_i = ic.index[ic >= k]
        new = df[df["u"].isin(keep_u) & df["i"].isin(keep_i)]
        if len(new) == len(df):
            return new.reset_index(drop=True)
        df = new


def cap_users(df, max_users, seed):
    if max_users is None:
        return df
    users = df["u"].unique()
    if len(users) <= max_users:
        return df
    rng = np.random.default_rng(seed)
    keep = rng.choice(users, size=max_users, replace=False)
    return df[df["u"].isin(keep)].reset_index(drop=True)


def add_labels(df, relevant_threshold):
    out = df.copy()
    out["R"] = (out["rating"] >= relevant_threshold).astype(np.int64)
    return out


def reindex(df):
    out = df.copy()
    out["u"], u_uniques = pd.factorize(out["u"])
    out["i"], i_uniques = pd.factorize(out["i"])
    return out, len(u_uniques), len(i_uniques)


def build_sequences(df):
    df = df.sort_values(["u", "ts"], kind="stable")
    sequences = {}
    for u, grp in df.groupby("u", sort=False):
        sequences[int(u)] = dict(
            items=grp["i"].to_numpy(np.int64),
            rating=grp["rating"].to_numpy(np.float32),
            R=grp["R"].to_numpy(np.int64),
        )
    return sequences


def leave_last_out(sequences):
    splits = {}
    for u, seq in sequences.items():
        items, R = seq["items"], seq["R"]
        if len(items) < 3:
            continue
        splits[u] = dict(
            train=items[:-2],
            val=(int(items[-2]), int(R[-2])),
            test=(int(items[-1]), int(R[-1])),
        )
    return splits

def prepare(cfg, verbose=True):
    df = factorized_load(cfg)
    if verbose:
        print(f"[data] raw interactions: {len(df):,}")
    df = k_core_filter(df, cfg.k_core)
    df = cap_users(df, cfg.max_users, cfg.seed)
    df = k_core_filter(df, cfg.k_core)
    df = add_labels(df, cfg.relevant_threshold)
    df, n_users, n_items = reindex(df)
    if verbose:
        print(f"[data] after {cfg.k_core}-core + cap: {len(df):,} interactions, "
              f"{n_users:,} users, {n_items:,} items, "
              f"relevant share = {df['R'].mean():.3f}")
    os.makedirs(cfg.cache_dir, exist_ok=True)
    np.savez(os.path.join(cfg.cache_dir, "interactions.npz"),
             u=df["u"].to_numpy(), i=df["i"].to_numpy(),
             rating=df["rating"].to_numpy(), R=df["R"].to_numpy(),
             ts=df["ts"].to_numpy())
    with open(os.path.join(cfg.cache_dir, "meta.json"), "w") as f:
        json.dump(dict(n_users=n_users, n_items=n_items, n_inter=len(df)), f)
    return load_cache(cfg)


def load_cache(cfg):
    npz = np.load(os.path.join(cfg.cache_dir, "interactions.npz"))
    with open(os.path.join(cfg.cache_dir, "meta.json")) as f:
        meta = json.load(f)
    df = pd.DataFrame({k: npz[k] for k in ["u", "i", "rating", "R", "ts"]})
    return dict(df=df, n_users=meta["n_users"], n_items=meta["n_items"])
