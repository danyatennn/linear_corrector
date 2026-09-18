from dataclasses import dataclass


@dataclass
class DataConfig:
    csv_path: str = "Movies_and_TV.csv.gz"
    cache_dir: str = "data_cache"

    k_core: int = 5
    max_users: int | None = 60000

    relevant_threshold: float = 4.0
    unwanted_threshold: float = 3.0

    seed: int = 0


@dataclass
class SASRecConfig:
    maxlen: int = 50
    hidden_dim: int = 128
    num_blocks: int = 2
    num_heads: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 80
    patience: int = 15
    num_neg_eval: int = 100
    device: str = "auto"


@dataclass
class ExperimentConfig:
    top_k: int = 20
    deltas: tuple = (0.8, 0.85, 0.9, 0.95)
    e_min: float = 0.1
    bias_strength: float = 4.0
    seed: int = 0


DATA = DataConfig()
SASREC = SASRecConfig()
EXP = ExperimentConfig()
