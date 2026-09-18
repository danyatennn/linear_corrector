import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA
from recsys_guarantees import data


def main():
    cache = data.prepare(DATA, verbose=True)
    df = cache["df"]
    print(f"[data] cached -> {DATA.cache_dir}/interactions.npz")
    print(f"[data] users={cache['n_users']:,}  items={cache['n_items']:,}  "
          f"interactions={len(df):,}")
    lengths = df.groupby("u").size()
    print(f"[data] seq length: median={lengths.median():.0f}  "
          f"p95={lengths.quantile(0.95):.0f}  max={lengths.max()}")


if __name__ == "__main__":
    main()
