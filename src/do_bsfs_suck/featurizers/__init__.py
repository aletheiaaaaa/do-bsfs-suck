import torch

from do_bsfs_suck.config import FeaturizerConfig
from do_bsfs_suck.featurizers.base import (
    Featurizer,
    block_soft_threshold,
    block_topk,
)
from do_bsfs_suck.featurizers.grassmann import GrassmannBSF, qr_retract
from do_bsfs_suck.featurizers.group_lasso import GroupLassoBSF
from do_bsfs_suck.featurizers.shuffled import ShuffledBSF
from do_bsfs_suck.featurizers.topk_sae import TopKSAE
from do_bsfs_suck.featurizers.vanilla import VanillaBSF

VARIANTS: dict[str, type[Featurizer]] = {
    "vanilla": VanillaBSF,
    "grassmann": GrassmannBSF,
    "group_lasso": GroupLassoBSF,
    "shuffled": ShuffledBSF,
    "topk_sae": TopKSAE,
}


def build(
    cfg: FeaturizerConfig, directions: torch.Tensor | None = None
) -> Featurizer:
    """`directions` is required by, and only used by, the shuffled control."""
    if cfg.variant == "shuffled":
        if directions is None:
            raise ValueError("the shuffled control needs a trained b=1 decoder")
        return ShuffledBSF(cfg, directions)
    return VARIANTS[cfg.variant](cfg)


__all__ = [
    "VARIANTS",
    "Featurizer",
    "GrassmannBSF",
    "GroupLassoBSF",
    "ShuffledBSF",
    "TopKSAE",
    "VanillaBSF",
    "block_soft_threshold",
    "block_topk",
    "build",
    "qr_retract",
]
