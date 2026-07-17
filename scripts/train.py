"""Train the projection head with the lazy hybrid data flow.

- A FROZEN backbone produces labels (per-gene ScoreCache) and the (mut, wt)
  embeddings (content-addressed EmbeddingCache, misses computed live). Embeddings
  are read either resident (bulk-load, small datasets) or streamed from H5 with
  worker prefetch (large datasets) -- auto-selected by --max_resident_gb.
- With --use_lora, a SEPARATE LoRA backbone embeds sequences live each step.
- Device is auto-detected (GPU by default; CPU opt-in by submitting to a CPU node
  once caches are warm). --warm_only fills the caches on GPU and exits.

Heavy on first (cold-cache) use; cheap once warm. Run via sbatch.
"""

from __future__ import annotations

from gsr.args import parse_args
from gsr.backbone.esm import ESMBackbone
from gsr.cache.embedding_cache import EmbeddingCache
from gsr.cache.score_cache import ScoreCache
from gsr.data.dataset import VariantDataset
from gsr.data.seq_dataset import SequenceVariantDataset
from gsr.data.stream_dataset import StreamingVariantDataset
from gsr.data.training_data import (build_metadata, fill_embeddings,
                                     select_embeddings_mode)
from gsr.data.uniref import load_wildtypes
from gsr.eval.dms_cache import build_or_load_dms_cache, dms_cache_exists
from gsr.train.trainer import Trainer
from gsr.utils.device import resolve_device
from gsr.utils.seeding import seed_everything
from gsr.utils.stats import print_dataset_stats


def build_evaluator(args, backbone):
    if args.eval_per_epoch == 0 and args.eval_every_steps == 0:
        return None
    from gsr.eval.centroid import CentroidDMSEvaluator
    return CentroidDMSEvaluator.from_args(args, backbone=backbone)


def _cpu_guard(msg: str) -> None:
    raise SystemExit(
        f"[train] {msg} and device=cpu (ESM on CPU is impractical). Run on a GPU "
        "node first to warm the caches (a normal GPU run, or --warm_only), then "
        "re-submit to a CPU node.")


def main():
    args = parse_args()
    args.device = resolve_device(args.device)
    seed_everything(args.seed)
    print(f"[train] device={args.device} embeddings_mode={args.embeddings_mode}")

    records = load_wildtypes(args.fasta_path, max_seq_len=args.max_seq_len)
    records.sort(key=lambda r: r.gene_id)
    if args.limit_genes:
        records = records[: args.limit_genes]

    frozen = ESMBackbone(args.esm_model, device=args.device, use_lora=False)
    score_cache = ScoreCache(args.esm_model, args.scorer)

    if args.device == "cpu":
        missing = score_cache.missing_genes([r.gene_id for r in records])
        if missing:
            _cpu_guard(f"{len(missing)} genes lack cached scores")

    df = build_metadata(records, args, frozen, score_cache)
    print_dataset_stats(df, title=f"train ({len(records)} genes)")

    emb_cache = EmbeddingCache(args.esm_model, args.embedding_layer, args.pooling)

    # --- warm-only: fill all caches on GPU, then exit ----------------------
    if args.warm_only:
        fill_embeddings(df, args, frozen, emb_cache, resident=False)
        build_or_load_dms_cache(args, backbone=frozen)
        print("[train] --warm_only: caches filled; exiting.")
        return

    # --- CPU guards for the training/eval caches ---------------------------
    if args.device == "cpu" and not args.use_lora:
        if emb_cache.missing_ids(df["variant_id"].tolist()):
            _cpu_guard("some embeddings are not cached")
        if (args.eval_per_epoch or args.eval_every_steps) and \
                not dms_cache_exists(args):
            _cpu_guard("the DMS eval cache is not built")

    evaluator = build_evaluator(args, backbone=frozen)

    if args.use_lora:
        lora = ESMBackbone(
            args.esm_model, device=args.device, use_lora=True,
            lora_cfg=dict(rank=args.lora_rank, alpha=args.lora_alpha,
                          dropout=args.lora_dropout,
                          target_modules=args.lora_target_modules))
        dataset = SequenceVariantDataset(df)
        print(f"[train] LoRA live path: {len(dataset)} items")
        Trainer(args, dataset, evaluator=evaluator, backbone=lora).fit()
        return

    # --- frozen path: pick resident vs streaming ---------------------------
    N, D = len(df), frozen.output_dim(args.pooling)
    pool_gb = N * D * 4 * 2 / 1e9
    mode = select_embeddings_mode(N, D, args.max_resident_gb, args.embeddings_mode)
    print(f"[train] pool={N} items, {pool_gb:.1f} GB (mut+wt) -> mode={mode}")

    if mode == "ram":
        mut, wt = fill_embeddings(df, args, frozen, emb_cache, resident=True)
        dataset = VariantDataset(df, mut, wt)
    else:
        fill_embeddings(df, args, frozen, emb_cache, resident=False)  # ensure cached
        dataset = StreamingVariantDataset(df, emb_cache)
    print(f"[train] frozen path: {len(dataset)} items, dim={dataset.input_dim}")
    Trainer(args, dataset, evaluator=evaluator, backbone=None).fit()


if __name__ == "__main__":
    main()
