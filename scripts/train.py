"""Train the projection head with the lazy hybrid data flow.

- A FROZEN backbone produces labels (via the per-gene ScoreCache) and, for the
  frozen path, the (mut, wt) embeddings (via the content-addressed EmbeddingCache,
  computing + caching misses live). Embeddings are held resident so the training
  loop does no per-batch I/O.
- With --use_lora, a SEPARATE LoRA backbone embeds sequences live each step
  (labels still come from the frozen backbone).

Everything runs from the human UniRef90 FASTA directly -- no separate build step.
Heavy: run via sbatch.
"""

from __future__ import annotations

from gsr.args import parse_args
from gsr.backbone.esm import ESMBackbone
from gsr.cache.embedding_cache import EmbeddingCache
from gsr.cache.score_cache import ScoreCache
from gsr.data.dataset import VariantDataset
from gsr.data.seq_dataset import SequenceVariantDataset
from gsr.data.training_data import build_metadata, fill_embeddings
from gsr.data.uniref import load_wildtypes
from gsr.train.trainer import Trainer
from gsr.utils.seeding import seed_everything
from gsr.utils.stats import print_dataset_stats


def build_evaluator(args, backbone):
    if args.eval_per_epoch == 0 and args.eval_every_steps == 0:
        return None
    try:
        from gsr.eval.centroid import CentroidDMSEvaluator
    except Exception as e:
        print(f"[train] centroid evaluator unavailable ({e}); training only.")
        return None
    return CentroidDMSEvaluator.from_args(args, backbone=backbone)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = args.device

    records = load_wildtypes(args.fasta_path, max_seq_len=args.max_seq_len)
    records.sort(key=lambda r: r.gene_id)
    if args.limit_genes:
        records = records[: args.limit_genes]

    # Frozen backbone: labels + (frozen) embeddings + DMS eval cache.
    frozen = ESMBackbone(args.esm_model, device=device, use_lora=False)
    score_cache = ScoreCache(args.esm_model, args.scorer)

    df = build_metadata(records, args, frozen, score_cache)
    print_dataset_stats(df, title=f"train ({len(records)} genes)")

    evaluator = build_evaluator(args, backbone=frozen)

    if args.use_lora:
        lora = ESMBackbone(
            args.esm_model, device=device, use_lora=True,
            lora_cfg=dict(rank=args.lora_rank, alpha=args.lora_alpha,
                          dropout=args.lora_dropout,
                          target_modules=args.lora_target_modules))
        dataset = SequenceVariantDataset(df)
        print(f"[train] LoRA live path: {len(dataset)} items")
        Trainer(args, dataset, evaluator=evaluator, backbone=lora).fit()
    else:
        emb_cache = EmbeddingCache(args.esm_model, args.embedding_layer, args.pooling)
        mut, wt = fill_embeddings(df, args, frozen, emb_cache)
        dataset = VariantDataset(df, mut, wt)
        print(f"[train] frozen path: {len(dataset)} items, dim={dataset.input_dim}")
        Trainer(args, dataset, evaluator=evaluator, backbone=None).fit()


if __name__ == "__main__":
    main()
