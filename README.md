# do-bsfs-suck

Evaluates the three block-sparse featurizer variants of
[2606.25234](https://arxiv.org/abs/2606.25234) on LM residual streams, against
matched SAE baselines, on trained vs randomly initialized transformers
([2501.17727](https://arxiv.org/abs/2501.17727)) — measuring reconstruction, MDL,
block geometry, feature absorption
([2409.14507](https://arxiv.org/abs/2409.14507)) and oversplitting
([2405.08366](https://arxiv.org/abs/2405.08366v3)).

```bash
uv run do-bsfs-suck sweep configs/smoke.yaml     # CPU-sized, exercises everything
uv run do-bsfs-suck sweep configs/sweep.yaml     # the real run
uv run do-bsfs-suck config configs/sweep.yaml    # same, with defaults filled in
uv run do-bsfs-suck figures --results results/sweep.json
```

Multi-GPU shards runs across processes rather than data-parallelising each one:
`uv run accelerate launch -m do_bsfs_suck sweep configs/sweep.yaml`.

Sparsity is parameterized by active dims `A = k*b` with `G*b` fixed, so equal `A`
means equal nonzeros and equal decoder params, and `b=1` is the SAE baseline.
Absorption is reported on both the matched-`A` and matched-`k` arms, since at
fixed `A` raising `b` shrinks `k = A/b` and moves the metric on its own.
