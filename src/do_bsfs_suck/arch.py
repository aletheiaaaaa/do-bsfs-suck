from torch import nn

# checked in order; the fallback below covers anything not listed
BLOCK_PATHS = (
    "gpt_neox.layers",
    "transformer.h",
    "model.layers",
    "model.decoder.layers",
)


def _resolve(model: nn.Module, path: str) -> nn.Module | None:
    node = model
    for part in path.split("."):
        node = getattr(node, part, None)
        if node is None:
            return None
    return node


def _module_name(model: nn.Module, target: nn.Module) -> str:
    for name, mod in model.named_modules():
        if mod is target:
            return name
    raise ValueError(f"module not found in {type(model).__name__}")


def blocks(model: nn.Module) -> nn.ModuleList:
    """The decoder block list, whatever the architecture calls it."""
    for path in BLOCK_PATHS:
        found = _resolve(model, path)
        if isinstance(found, nn.ModuleList):
            return found

    n_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if n_layers is not None:
        for _, mod in model.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) == n_layers:
                return mod

    raise ValueError(f"no decoder blocks found in {type(model).__name__}")


def embedding_prefixes(model: nn.Module, include_unembed: bool = True) -> tuple[str, ...]:
    """Parameter prefixes for the embeddings, for freezing or resampling.

    Tied models (gpt2) share one tensor between input and output, so there is no
    separate unembedding parameter and only one prefix comes back.
    """
    inp = model.get_input_embeddings()
    names = [_module_name(model, inp)]

    if include_unembed:
        out = getattr(model, "get_output_embeddings", lambda: None)()
        if out is not None and out.weight is not inp.weight:
            names.append(_module_name(model, out))

    return tuple(names)
