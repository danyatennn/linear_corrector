import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _Block(nn.Module):

    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.attn_ln = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ffn_ln = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask, key_padding_mask):
        q = self.attn_ln(x)
        a, _ = self.attn(q, x, x, attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask, need_weights=False)
        a = torch.nan_to_num(a)
        x = x + self.dropout(a)
        x = x + self.ffn(self.ffn_ln(x))
        return x


class SASRec(nn.Module):
    def __init__(self, num_items, cfg):
        super().__init__()
        d = cfg.hidden_dim
        self.maxlen = cfg.maxlen
        self.item_emb = nn.Embedding(num_items + 1, d, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.maxlen, d)
        self.emb_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [_Block(d, cfg.num_heads, cfg.dropout) for _ in range(cfg.num_blocks)]
        )
        self.last_ln = nn.LayerNorm(d)
        self.d = d

    def log2feats(self, seq):
        B, L = seq.shape
        x = self.item_emb(seq) * (self.d ** 0.5)
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = x + self.pos_emb(positions)
        x = self.emb_dropout(x)
        pad_mask = (seq == 0)
        causal = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), 1)
        for block in self.blocks:
            x = block(x, attn_mask=causal, key_padding_mask=pad_mask)
        return self.last_ln(x)

    def forward(self, seq, pos, neg):
        feats = self.log2feats(seq)
        pos_e = self.item_emb(pos)
        neg_e = self.item_emb(neg)
        pos_logits = (feats * pos_e).sum(-1)
        neg_logits = (feats * neg_e).sum(-1)
        return pos_logits, neg_logits

    def seq_repr(self, seq):
        return self.log2feats(seq)[:, -1, :]
    
def _left_pad(seq, maxlen):
    seq = seq[-maxlen:]
    out = np.zeros(maxlen, dtype=np.int64)
    out[maxlen - len(seq):] = seq
    return out


class SeqDataset(Dataset):

    def __init__(self, train_sequences, num_items, maxlen, seed=0):
        self.samples = [s for s in train_sequences if len(s) >= 2]
        self.num_items = num_items
        self.maxlen = maxlen
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        inp = _left_pad(s[:-1], self.maxlen)
        pos = _left_pad(s[1:], self.maxlen)
        neg = np.zeros(self.maxlen, dtype=np.int64)
        real = pos != 0
        neg_vals = self.rng.integers(1, self.num_items + 1, size=real.sum())
        neg[real] = neg_vals
        return (torch.from_numpy(inp), torch.from_numpy(pos), torch.from_numpy(neg))


def _ndcg_hr_at_k(rank, k=10):
    if rank < k:
        return 1.0 / np.log2(rank + 2), 1.0
    return 0.0, 0.0


@torch.no_grad()
def evaluate(model, eval_users, splits, num_items, maxlen, device,
             num_neg=100, use="val", seed=0, max_users=5000):
    model.eval()
    rng = np.random.default_rng(seed)
    users = eval_users[:max_users]
    ndcgs, hrs = [], []
    for u in users:
        sp = splits[u]
        hist = sp["train"] if use == "val" else np.append(sp["train"], sp["val"][0])
        target = sp["val"][0] if use == "val" else sp["test"][0]
        if len(hist) == 0:
            continue
        seq = torch.from_numpy(_left_pad(hist, maxlen)).unsqueeze(0).to(device)
        rep = model.seq_repr(seq)
        negs = rng.integers(1, num_items + 1, size=num_neg)
        cands = np.concatenate([[target], negs])
        cand_emb = model.item_emb(torch.from_numpy(cands).to(device))
        scores = (rep @ cand_emb.T).squeeze(0).cpu().numpy()
        rank = int((scores > scores[0]).sum())
        n, h = _ndcg_hr_at_k(rank, k=10)
        ndcgs.append(n); hrs.append(h)
    return dict(ndcg=float(np.mean(ndcgs)), hr=float(np.mean(hrs)))


def save_checkpoint(path, state_dict, num_items, cfg, epoch, val_ndcg):
    torch.save(dict(state_dict=state_dict, num_items=num_items, cfg=vars(cfg),
                    epoch=epoch, val_ndcg=val_ndcg), path)


def train_sasrec(num_items, splits, cfg, verbose=True, ckpt_path=None, ckpt_every=5):
    device = resolve_device(cfg.device)
    torch.manual_seed(0)
    model = SASRec(num_items, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98))
    bce = nn.BCEWithLogitsLoss(reduction="none")

    train_seqs = [splits[u]["train"] for u in splits]
    dataset = SeqDataset(train_seqs, num_items, cfg.maxlen)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    eval_users = list(splits.keys())

    best_ndcg, best_state, bad = -1.0, None, 0
    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        total = 0.0
        for inp, pos, neg in loader:
            inp, pos, neg = inp.to(device), pos.to(device), neg.to(device)
            pos_logits, neg_logits = model(inp, pos, neg)
            mask = (pos != 0).float()
            loss = (bce(pos_logits, torch.ones_like(pos_logits)) * mask).sum() / mask.sum()
            loss = loss + (bce(neg_logits, torch.zeros_like(neg_logits)) * mask).sum() / mask.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        metrics = evaluate(model, eval_users, splits, num_items, cfg.maxlen, device, use="val")
        if verbose:
            print(f"[sasrec] epoch {epoch:3d}  loss={total/len(loader):.4f}  "
                  f"val NDCG@10={metrics['ndcg']:.4f}  HR@10={metrics['hr']:.4f}")
        improved = metrics["ndcg"] > best_ndcg
        if improved:
            best_ndcg = metrics["ndcg"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ckpt_path and best_state is not None and (improved or epoch % ckpt_every == 0):
            save_checkpoint(ckpt_path, best_state, num_items, cfg, epoch, best_ndcg)
        if bad >= cfg.patience:
            if verbose:
                print(f"[sasrec] early stop at epoch {epoch} (best val NDCG@10={best_ndcg:.4f})")
            break
    model.load_state_dict(best_state)
    if ckpt_path:
        save_checkpoint(ckpt_path, best_state, num_items, cfg, epoch, best_ndcg)
    return model, device


@torch.no_grad()
def score_pairs(model, histories, candidates, maxlen, device, batch_size=512):
    model.eval()
    out = np.empty(len(candidates), dtype=np.float32)
    for start in range(0, len(candidates), batch_size):
        end = min(start + batch_size, len(candidates))
        seqs = np.stack([_left_pad(histories[j], maxlen) for j in range(start, end)])
        seq_t = torch.from_numpy(seqs).to(device)
        rep = model.seq_repr(seq_t)
        cand_t = torch.from_numpy(np.asarray(candidates[start:end], dtype=np.int64)).to(device)
        cand_emb = model.item_emb(cand_t)
        out[start:end] = (rep * cand_emb).sum(-1).cpu().numpy()
    return out
