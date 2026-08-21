from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from do_bsfs_suck.arch import embedding_prefixes
from do_bsfs_suck.config import Condition



@dataclass(frozen=True)
class RandomizeSpec:
    """Two readings of Heap et al. left open; both are exposed rather than picked."""

    resample_layernorm: bool = False
    freeze_unembed: bool = True
    # control: fresh noise per token occurrence, not per vocabulary entry
    control_per_occurrence: bool = True


def _is_layernorm(name: str) -> bool:
    return "layernorm" in name or "layer_norm" in name


def _frozen_prefixes(model, condition: Condition, spec: RandomizeSpec) -> tuple[str, ...]:
    if condition != "rand_excl_emb":
        return ()
    return embedding_prefixes(model, spec.freeze_unembed)


@torch.no_grad()
def resample_(model: nn.Module, condition: Condition, spec: RandomizeSpec, seed: int) -> list[str]:
    """Gaussian-resample parameters in place, matching each tensor's own mean/var."""
    if condition not in ("rand_excl_emb", "rand_incl_emb"):
        return []

    frozen = _frozen_prefixes(model, condition, spec)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    touched = []

    for name, p in model.named_parameters():
        if name.startswith(frozen):
            continue
        if _is_layernorm(name) and not spec.resample_layernorm:
            continue
        noise = torch.randn(p.shape, generator=gen, dtype=torch.float32)
        p.copy_((noise * p.std().item() + p.mean().item()).to(p.dtype))
        touched.append(name)

    return touched


STEP0 = ("step0", "step0_excl_emb")


@torch.no_grad()
def splice_trained_embeddings_(
    model: nn.Module, name: str, spec: RandomizeSpec, dtype: torch.dtype
) -> None:
    """Copy trained embeddings onto step-0 weights."""
    trained = AutoModelForCausalLM.from_pretrained(name, revision="main", dtype=dtype)
    src = dict(trained.named_parameters())
    dst = dict(model.named_parameters())
    for prefix in embedding_prefixes(model, spec.freeze_unembed):
        key = f"{prefix}.weight"
        if key in src and key in dst:
            dst[key].copy_(src[key])
    del trained


def load_model(
    name: str,
    condition: Condition = "trained",
    spec: RandomizeSpec = RandomizeSpec(),
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    revision = "step0" if condition in STEP0 else "main"
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype)
    if condition == "step0_excl_emb":
        splice_trained_embeddings_(model, name, spec, dtype)
    resample_(model, condition, spec, seed)
    return model.eval()


def control_hook(spec: RandomizeSpec, seed: int):
    """Replace input embeddings with i.i.d. standard Gaussian noise at inference."""
    gen_state = {"gen": None}

    def hook(module: nn.Module, args, output: torch.Tensor) -> torch.Tensor:
        if gen_state["gen"] is None:
            gen_state["gen"] = torch.Generator(device=output.device).manual_seed(seed)
        if spec.control_per_occurrence:
            return torch.randn(
                output.shape, generator=gen_state["gen"],
                device=output.device, dtype=output.dtype,
            )
        # one fixed random embedding per token id
        ids = args[0]
        table = torch.randn(
            module.weight.shape, generator=gen_state["gen"],
            device=output.device, dtype=output.dtype,
        )
        return table[ids]

    return hook
