"""Standalone evaluation of embeddings (backbone or a trained projection).

Runs any subset of {centroid, dimreduction, regression} on DMS data. With no
checkpoint it evaluates the raw ESM backbone (the baseline); with --checkpoint it
applies that run's projection head. Because it consumes cached backbone
embeddings + a project_fn, the SAME code runs here and inside training.

Heavy (loads ESM to build the DMS embedding cache the first time) -- run via sbatch.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from gsr import paths
from gsr.args import DISTANCE_CHOICES, POOLING_CHOICES
from gsr.eval.dms_cache import build_or_load_dms_cache
from gsr.utils.seeding import seed_everything


def build_parser():
    p = argparse.ArgumentParser(description="Standalone embedding evaluation.")
    p.add_argument("--out_name", required=True,
                   help="Output subdir under <scratch>/eval/<out_name>/.")
    p.add_argument("--checkpoint", default=None,
                   help="Path to a run checkpoint (best.pt/final.pt). "
                   "Omit to evaluate the raw backbone (identity projection).")
    p.add_argument("--evaluators", nargs="+",
                   default=["centroid", "dimreduction", "regression"])
    # backbone / embedding config (must match the checkpoint's if given)
    p.add_argument("--esm_model", default="esmc_600m")
    p.add_argument("--embedding_layer", type=int, default=-1)
    p.add_argument("--pooling", default="concat", choices=POOLING_CHOICES)
    p.add_argument("--score_batch_size", type=int, default=16)
    # DMS data
    p.add_argument("--dms_selection_types", nargs="+",
                   default=["Activity", "Binding", "Expression", "Stability",
                            "OrganismalFitness"])
    p.add_argument("--dms_max_per_assay", type=int, default=200)
    # centroid eval
    p.add_argument("--eval_distance", default="cosine", choices=DISTANCE_CHOICES)
    p.add_argument("--held_out_gene_frac", type=float, default=0.2)
    p.add_argument("--centroid_subsample", type=int, default=0)  # 0 = all (standalone)
    # dimreduction
    p.add_argument("--dimred_methods", nargs="+", default=["pca", "tsne", "umap"])
    p.add_argument("--dimred_max_points", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p


def load_project_fn(checkpoint, backbone_dim, pooling, device):
    """Return a project_fn; identity if no checkpoint."""
    if checkpoint is None:
        print("[eval] no checkpoint -> evaluating raw backbone (identity)")
        return lambda emb: np.asarray(emb, dtype=np.float32)

    from gsr.models.projection_head import ProjectionHead
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    if ckpt["input_dim"] != backbone_dim:
        raise ValueError(
            f"checkpoint input_dim {ckpt['input_dim']} != backbone/pooling dim "
            f"{backbone_dim}; pooling/model mismatch with the trained head.")
    head = ProjectionHead(
        input_dim=ckpt["input_dim"], hidden_dims=cfg["head_hidden_dims"],
        out_dim=cfg["head_out_dim"], dropout=cfg["head_dropout"],
        activation=cfg["head_activation"], norm=cfg["head_norm"])
    head.load_state_dict(ckpt["head_state"])
    head.eval().to(device)
    print(f"[eval] loaded projection head from {checkpoint}")

    def fn(emb):
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(emb, dtype=np.float32)).to(device)
            return head(x).cpu().numpy()
    return fn


def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    emb, meta = build_or_load_dms_cache(args)
    project_fn = load_project_fn(args.checkpoint, emb.shape[1], args.pooling, device)

    out_dir = paths.EVAL_DIR / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}

    if "centroid" in args.evaluators:
        from gsr.eval.centroid import CentroidDMSEvaluator
        ev = CentroidDMSEvaluator(args, emb, meta)
        ev.save_splits(out_dir / "gene_splits.json")
        m = ev.evaluate(project_fn)
        metrics.update(m)
        print(f"[eval] centroid spearman_mean={m['centroid/spearman_mean']:.4f} "
              f"axis_mean={m['centroid/spearman_axis_mean']:.4f}")

    if "regression" in args.evaluators:
        from gsr.eval.regression import evaluate_regression
        m = evaluate_regression(emb, meta, project_fn=project_fn, seed=args.seed)
        metrics.update(m)
        print(f"[eval] regression spearman_mean={m['regression/spearman_mean']:.4f} "
              f"over {int(m['regression/n_assays'])} assays")

    if "dimreduction" in args.evaluators:
        from gsr.eval.dimreduction import make_figures
        Z = project_fn(emb)
        tag = "projected" if args.checkpoint else "backbone"
        make_figures(Z, meta, out_dir, tag=tag, methods=args.dimred_methods,
                     max_points=args.dimred_max_points, seed=args.seed)

    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[eval] wrote metrics -> {out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
