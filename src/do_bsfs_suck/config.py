from dataclasses import dataclass, field
from typing import Literal

# Sparsity is parameterized by active dims A = k*b, not k, and G*b is held
# fixed -- so equal A means equal nonzeros and equal decoder params, and b=1 is
# the SAE baseline at L0 = A.

Variant = Literal["vanilla", "grassmann", "group_lasso", "shuffled", "topk_sae"]

Condition = Literal[
    "trained",
    "rand_excl_emb",
    "rand_incl_emb",
    "step0",
    "step0_excl_emb",
    "control",
]


@dataclass(frozen=True)
class FeaturizerConfig:
    d_in: int
    variant: Variant = "vanilla"
    dict_dims: int = 16384
    block_dim: int = 1
    active_dims: int = 64
    lasso_coeff: float = 1e-3  # group lasso only
    seed: int = 0

    @property
    def n_blocks(self) -> int:
        if self.dict_dims % self.block_dim:
            raise ValueError(
                f"dict_dims={self.dict_dims} not divisible by block_dim={self.block_dim}"
            )
        return self.dict_dims // self.block_dim

    @property
    def k(self) -> int:
        if self.active_dims % self.block_dim:
            raise ValueError(
                f"active_dims={self.active_dims} not divisible by "
                f"block_dim={self.block_dim}"
            )
        k = self.active_dims // self.block_dim
        if k > self.n_blocks:
            raise ValueError(f"k={k} exceeds n_blocks={self.n_blocks}")
        return k

    @property
    def name(self) -> str:
        return (
            f"{self.variant}_b{self.block_dim}_A{self.active_dims}"
            f"_G{self.n_blocks}_s{self.seed}"
        )


@dataclass(frozen=True)
class StreamConfig:
    model: str = "EleutherAI/pythia-160m"
    condition: Condition = "trained"
    layers: tuple[int, ...] = (3, 6, 9)
    dataset: str = "monology/pile-uncopyrighted"
    n_tokens: int = 50_000_000
    seq_len: int = 512
    batch_seqs: int = 16
    # position 0 of a residual stream is an outlier and would dominate recon
    drop_first: int = 1
    # model randomization seed; changing it changes WHICH random model you get
    seed: int = 0
    # corpus order only. held-out eval data must vary this, never `seed`, or the
    # eval runs against a differently-randomized model than training did
    data_seed: int = 0


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 3e-4
    batch_tokens: int = 4096
    warmup_frac: float = 0.02
    dead_after_tokens: int = 4_000_000
    aux_k: int = 64
    aux_coeff: float = 1.0 / 32.0
    grad_clip: float = 1.0
    log_every: int = 200


@dataclass(frozen=True)
class RunConfig:
    stream: StreamConfig = field(default_factory=StreamConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    featurizers: tuple[FeaturizerConfig, ...] = ()
