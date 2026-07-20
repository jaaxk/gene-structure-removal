#!/usr/bin/env python3
"""Append one run's result to tune/results.jsonl from its resolved_args.json +
final_full_metrics.json. Usage: tune_log_result.py <run_name> [hyp_tag]"""
import json
import sys
from pathlib import Path

RUNS_DIR = Path("/scratch/jv2807/gene_structure_removal/runs")
RESULTS = Path("/scratch/jv2807/gene_structure_removal/tune/results.jsonl")

SWEPT_KEYS = [
    "loss_type", "distance_metric", "use_learnable_scale", "ntxent_temperature",
    "triplet_margin", "head_hidden_dims", "head_out_dim", "head_dropout",
    "head_activation", "head_norm", "lr", "weight_decay", "epochs", "batch_size",
    "batch_mode", "genes_per_batch", "use_lora", "lora_rank", "lora_alpha",
    "lora_dropout", "lora_target_modules", "esm_lr", "limit_genes",
    "variants_per_gene",
]


def main():
    run_name = sys.argv[1]
    hyp_tag = sys.argv[2] if len(sys.argv) > 2 else ""
    run_dir = RUNS_DIR / run_name
    args_path = run_dir / "resolved_args.json"
    metrics_path = run_dir / "final_full_metrics.json"
    if not metrics_path.exists():
        print(f"NO METRICS for {run_name} (run may have failed)", file=sys.stderr)
        sys.exit(1)
    args = json.loads(args_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    is_lora = "best_source" in metrics
    if is_lora:
        # LoRA runs report both head-projected and raw (post-LoRA) backbone
        # metrics; best_source/best_spearman_mean pick the winner.
        head_m = {k[len("head/"):]: v for k, v in metrics.items() if k.startswith("head/")}
        backbone_m = {k[len("backbone/"):]: v for k, v in metrics.items() if k.startswith("backbone/")}
        best_source = metrics["best_source"]
        best_m = backbone_m if best_source == "backbone" else head_m
        row = {
            "run_name": run_name,
            "hyp": hyp_tag,
            "hparams": {k: args.get(k) for k in SWEPT_KEYS},
            "best_source": best_source,
            "macro_spearman_mean": metrics["best_spearman_mean"],
            "head_macro_spearman_mean": head_m.get("centroid/spearman_mean"),
            "backbone_macro_spearman_mean": backbone_m.get("centroid/spearman_mean"),
            "per_type_diff": {
                k.split("/")[-1]: v for k, v in best_m.items()
                if k.startswith("centroid/spearman_diff/")
            },
        }
    else:
        row = {
            "run_name": run_name,
            "hyp": hyp_tag,
            "hparams": {k: args.get(k) for k in SWEPT_KEYS},
            "macro_spearman_mean": metrics.get("centroid/spearman_mean"),
            "per_type_diff": {
                k.split("/")[-1]: v for k, v in metrics.items()
                if k.startswith("centroid/spearman_diff/")
            },
        }
    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"logged {run_name}: macro={row['macro_spearman_mean']:.4f}")


if __name__ == "__main__":
    main()
