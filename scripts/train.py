"""Train the projection head on a built VariantStore.

Reads <scratch>/store/<dataset_name>, trains the head with the configured
contrastive loss, and (Phase 4+) runs the zero-shot DMS centroid evaluator during
and after training. Heavy -- run via sbatch.
"""

from __future__ import annotations

from gsr import paths
from gsr.args import parse_args
from gsr.data.dataset import VariantDataset
from gsr.data.seq_dataset import SequenceVariantDataset
from gsr.scoring.store import VariantStore
from gsr.train.trainer import Trainer
from gsr.utils.seeding import seed_everything
from gsr.utils.stats import print_dataset_stats


def build_evaluator(args, dataset):
    """Attach the zero-shot DMS centroid evaluator when requested.

    Returns None (train contrastive-only) if evaluation is disabled or the
    evaluator dependencies are not available yet.
    """
    if getattr(args, "eval_every_steps", 0) == 0:
        return None
    try:
        from gsr.eval.centroid import CentroidDMSEvaluator
    except Exception as e:  # eval module not present yet (pre-Phase-4)
        print(f"[train] centroid evaluator unavailable ({e}); training only.")
        return None
    return CentroidDMSEvaluator.from_args(args)


def main():
    args = parse_args()
    seed_everything(args.seed)

    store = VariantStore(paths.SCRATCH_ROOT / "store" / args.dataset_name)
    # LoRA finetunes the backbone -> embeddings can't be cached, serve sequences.
    if args.use_lora:
        dataset = SequenceVariantDataset(store)
        print("[train] LoRA live path: serving sequences (backbone finetuned).")
    else:
        dataset = VariantDataset(store)
    print_dataset_stats(dataset.df, title=f"train ({args.dataset_name})")
    print(f"[train] genes={dataset.df['gene_id'].nunique()} items={len(dataset)} "
          f"use_lora={args.use_lora}")

    evaluator = build_evaluator(args, dataset)
    Trainer(args, dataset, evaluator=evaluator).fit()


if __name__ == "__main__":
    main()
