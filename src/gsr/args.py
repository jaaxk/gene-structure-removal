"""Command-line configuration for the ESM Gene Structure Removal project.

Config style mirrors the sibling repo ``/home/jv2807/dms_contrastive``: a single
argparse parser is the source of truth, and each ``run/*.sbatch`` script sets a
block of shell variables that map onto these flags. Grouping + inline help keep
every knob discoverable; ``validate_args`` catches nonsensical combinations early
so runs fail loudly at startup rather than silently mid-training.
"""

from __future__ import annotations

import argparse

# Canonical amino-acid vocabulary used for single-substitution mutagenesis.
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

SCORER_CHOICES = ("masked_marginal", "wt_marginal", "pll")
POOLING_CHOICES = ("mean", "mutated_position", "concat")
LOSS_CHOICES = ("wt_anchored_bce", "contrastive_ce", "ntxent", "triplet")
DISTANCE_CHOICES = ("cosine", "euclidean")
BATCH_MODE_CHOICES = ("gene_diverse", "cross_gene")
ACTIVATION_CHOICES = ("relu", "gelu", "silu", "tanh")
NORM_CHOICES = ("none", "layernorm", "batchnorm")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gene_structure_removal",
        description="Train a projection head that removes ESM gene structure "
        "via a self-supervised contrastive objective labeled by frozen-ESM LL/PLL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Run / bookkeeping --------------------------------------------------
    g = p.add_argument_group("run")
    g.add_argument("--run_name", type=str, required=True,
                   help="Unique name; outputs go to <scratch>/runs/<run_name>/.")
    g.add_argument("--fasta_path", type=str,
                   default="/scratch/jv2807/gene_structure_removal/data/"
                           "human_uniref90.fasta",
                   help="Wild-type sequences (human UniRef90 by default).")
    g.add_argument("--limit_genes", type=int, default=0,
                   help="Cap number of genes (after length filter). 0 = all.")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--wandb_project", type=str, default="gene-structure-removal")
    g.add_argument("--wandb_mode", type=str, default="online",
                   choices=("online", "offline", "disabled"))
    g.add_argument("--device", type=str, default="auto",
                   choices=("auto", "cuda", "cpu"),
                   help="auto = cuda if a GPU is visible else cpu.")
    g.add_argument("--warm_only", action="store_true",
                   help="Fill the score + embedding + DMS caches, then exit "
                   "(a dedicated GPU pre-fill; training self-warms otherwise).")

    # --- Backbone (ESM) -----------------------------------------------------
    g = p.add_argument_group("backbone")
    g.add_argument("--esm_model", type=str, default="esmc_600m",
                   help="ESM-C model id / HF repo, loaded via transformers.")
    g.add_argument("--embedding_layer", type=int, default=-1,
                   help="Hidden layer to take residue embeddings from (-1 = last).")
    g.add_argument("--pooling", type=str, default="concat", choices=POOLING_CHOICES,
                   help="How a variant sequence becomes one vector for the head. "
                   "For mutated_position/concat the WT is pooled at the SAME "
                   "mutated residue as the variant.")
    g.add_argument("--max_seq_len", type=int, default=1024,
                   help="Length cap on wild-type sequences (PLL cost is O(L)).")

    # --- LoRA (off by default) ---------------------------------------------
    g = p.add_argument_group("lora")
    g.add_argument("--use_lora", action="store_true",
                   help="Finetune the backbone with LoRA (default: frozen).")
    g.add_argument("--lora_rank", type=int, default=8)
    g.add_argument("--lora_alpha", type=int, default=16)
    g.add_argument("--lora_dropout", type=float, default=0.05)
    g.add_argument("--lora_target_modules", type=str, nargs="+",
                   default=["query", "key", "value"])

    # --- Scoring (label source) --------------------------------------------
    g = p.add_argument_group("scoring")
    g.add_argument("--scorer", type=str, default="masked_marginal",
                   choices=SCORER_CHOICES,
                   help="Frozen-ESM likelihood used to label variants.")
    g.add_argument("--score_batch_size", type=int, default=8,
                   help="Sequences per forward batch during scoring/embedding.")

    # --- embedding read mode (frozen path) ----------------------------------
    g = p.add_argument_group("embeddings_io")
    g.add_argument("--embeddings_mode", type=str, default="auto",
                   choices=("auto", "ram", "stream"),
                   help="How training reads cached embeddings: ram (bulk-load "
                   "resident) / stream (worker-prefetch from H5) / auto (by size).")
    g.add_argument("--max_resident_gb", type=float, default=32.0,
                   help="auto mode uses ram below this pool size, else stream.")
    g.add_argument("--num_workers", type=int, default=8,
                   help="DataLoader workers for streaming reads.")
    g.add_argument("--prefetch_factor", type=int, default=4,
                   help="Batches prefetched per worker (streaming).")

    # --- Mutagenesis / dataset build ---------------------------------------
    g = p.add_argument_group("data_build")
    g.add_argument("--variants_per_gene", type=int, default=200,
                   help="Single-aa variants sampled + embedded per gene. Should be "
                   ">= batch_size so repeated-gene epochs can use fresh variants.")
    g.add_argument("--min_variants_per_gene", type=int, default=10,
                   help="Genes with fewer usable variants are dropped.")
    g.add_argument("--quartile_low", type=float, default=0.25,
                   help="Bottom fraction of |LL_mut-LL_wt| labeled 'same'.")
    g.add_argument("--quartile_high", type=float, default=0.25,
                   help="Top fraction of |LL_mut-LL_wt| labeled 'different'.")
    g.add_argument("--shard_size", type=int, default=64,
                   help="Genes per embedding shard (one h5 file).")

    # --- Projection head ----------------------------------------------------
    g = p.add_argument_group("head")
    g.add_argument("--head_hidden_dims", type=int, nargs="+", default=[1024, 512],
                   help="Hidden layer widths of the 3-layer MLP projection head.")
    g.add_argument("--head_out_dim", type=int, default=256)
    g.add_argument("--head_dropout", type=float, default=0.1)
    g.add_argument("--head_activation", type=str, default="gelu",
                   choices=ACTIVATION_CHOICES)
    g.add_argument("--head_norm", type=str, default="layernorm", choices=NORM_CHOICES)

    # --- Loss ---------------------------------------------------------------
    g = p.add_argument_group("loss")
    g.add_argument("--loss_type", type=str, default="wt_anchored_bce",
                   choices=LOSS_CHOICES,
                   help="wt_anchored_bce: pull each variant toward/away from ITS "
                   "WT by label. contrastive_ce: in-batch pairs between variants.")
    g.add_argument("--distance_metric", type=str, default="cosine",
                   choices=DISTANCE_CHOICES)
    g.add_argument("--use_learnable_scale", action="store_true", default=True,
                   help="Learnable alpha/beta on similarity logits (contrastive_ce).")
    g.add_argument("--no_learnable_scale", dest="use_learnable_scale",
                   action="store_false")
    g.add_argument("--ntxent_temperature", type=float, default=0.1)
    g.add_argument("--triplet_margin", type=float, default=1.0)

    # --- Batching / diversity ----------------------------------------------
    g = p.add_argument_group("batching")
    g.add_argument("--batch_mode", type=str, default="gene_diverse",
                   choices=BATCH_MODE_CHOICES,
                   help="gene_diverse: one gene per batch, every gene once per epoch. "
                   "cross_gene: mix genes_per_batch different genes (off by default).")
    g.add_argument("--batch_size", type=int, default=64)
    g.add_argument("--genes_per_batch", type=int, default=2,
                   help="Only used when batch_mode=cross_gene.")
    g.add_argument("--balance_labels", action="store_true", default=True,
                   help="Balance same/different within each batch.")
    g.add_argument("--no_balance_labels", dest="balance_labels",
                   action="store_false")

    # --- Optimization -------------------------------------------------------
    g = p.add_argument_group("optim")
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--esm_lr", type=float, default=1e-5,
                   help="Separate LR for LoRA params when --use_lora.")
    g.add_argument("--weight_decay", type=float, default=0.0)
    g.add_argument("--epochs", type=int, default=20)
    g.add_argument("--grad_clip", type=float, default=1.0)
    g.add_argument("--amp", action="store_true", default=True)
    g.add_argument("--no_amp", dest="amp", action="store_false")

    # --- Evaluation ---------------------------------------------------------
    g = p.add_argument_group("eval")
    g.add_argument("--eval_per_epoch", type=int, default=8,
                   help="Centroid evals per epoch (derives the step cadence). "
                   "LoRA runs auto-throttle to <=2/epoch (re-embedding DMS is "
                   "costly). 0 disables during-training eval.")
    g.add_argument("--eval_every_steps", type=int, default=0,
                   help="Explicit eval cadence override; 0 = derive from "
                   "eval_per_epoch.")
    g.add_argument("--centroid_subsample", type=int, default=2000,
                   help="Max held-out variants per selection type used to build "
                   "centroids during training (keeps eval steps fast). 0 = all.")
    g.add_argument("--eval_distance", type=str, default="cosine",
                   choices=DISTANCE_CHOICES,
                   help="Distance metric for centroid scoring.")
    g.add_argument("--held_out_gene_frac", type=float, default=0.2,
                   help="Per selection type, fraction of genes used as scored "
                   "queries; centroids are built from the remaining genes.")
    g.add_argument("--dms_selection_types", type=str, nargs="+",
                   default=["Activity", "Binding", "Expression", "Stability",
                            "OrganismalFitness"],
                   help="DMS selection types to evaluate (CSV file stems).")
    g.add_argument("--dms_max_per_assay", type=int, default=200,
                   help="Cap variants sampled per DMS assay (bounds embedding cost).")

    return p


def validate_args(args: argparse.Namespace) -> None:
    """Fail loudly on impossible combinations before any compute happens."""
    assert args.variants_per_gene >= args.min_variants_per_gene, (
        "variants_per_gene must be >= min_variants_per_gene"
    )
    if args.variants_per_gene < args.batch_size:
        print(
            f"[warn] variants_per_gene ({args.variants_per_gene}) < batch_size "
            f"({args.batch_size}): repeated-gene epochs cannot draw fresh variants."
        )
    assert 0.0 < args.quartile_low < 1.0 and 0.0 < args.quartile_high < 1.0, (
        "quartile fractions must be in (0, 1)"
    )
    assert args.quartile_low + args.quartile_high <= 1.0, (
        "quartile_low + quartile_high must be <= 1.0 (middle band is dropped)"
    )
    assert len(args.head_hidden_dims) >= 1, "head needs >= 1 hidden layer"
    assert 0.0 <= args.head_dropout < 1.0
    assert 0.0 < args.held_out_gene_frac < 1.0
    if args.batch_mode == "cross_gene":
        assert args.genes_per_batch >= 2, "cross_gene needs genes_per_batch >= 2"
        assert args.batch_size % args.genes_per_batch == 0, (
            "batch_size must be divisible by genes_per_batch in cross_gene mode"
        )
    if args.use_lora:
        assert args.lora_rank > 0 and args.lora_alpha > 0


def parse_args(argv=None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    validate_args(args)
    return args
