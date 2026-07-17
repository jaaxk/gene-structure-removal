# ESM Gene Structure Removal

Learn a projection of frozen-ESM protein embeddings that **removes per-gene
structure**, so that the small signal distinguishing single-amino-acid molecular
effects is no longer washed out by which protein a sequence belongs to.

## Why

We use ESM as a base model to predict *molecular* effects behind mutation
pathogenicity (e.g. trafficking-defective vs. functional PKD1 variants,
hypomorphic BRCA2 variants, selection-type-specific DMS scores). ESM's
log-likelihood ratio is a strong 1-D signal, but to predict *multi-dimensional*
effects we want informative **embeddings**. The obstacle: because ESM is trained
to model likely sequences, its embedding space is dominated by inter-protein
differences, and single-residue changes are tiny by comparison.

**Hypothesis:** the information needed to separate molecular effects already
exists inside ESM; we just need to remove the gene structure to access it.

**Approach:** train a projection head on top of a frozen ESM using a
*self-supervised* contrastive objective whose labels come from the frozen model
itself. For each wild-type (WT) protein we score its single-aa variants with
ESM's (pseudo-)log-likelihood. Variants whose likelihood is **close** to the WT
are pulled together; variants whose likelihood is **far** are pushed apart. No
external labels are used for training.

Success is measured by **selection-type-specific zero-shot DMS prediction**
(Spearman) improving on the projected embeddings vs. the raw ESM backbone.

## How it works

```
WT protein ──► sample single-aa variants ──► frozen ESM LL/PLL score
                                                     │
                     per-gene quartiles of |LL_mut − LL_wt|
                                                     │
                        label: "same" (close) / "different" (far)  [middle dropped]
                                                     │
frozen ESM embedding ──► projection head (3-layer MLP) ──► contrastive loss
                                                     │
             during/after training: zero-shot DMS centroid eval (Spearman)
```

- **Backbone:** ESM-C 600M (frozen by default; optional LoRA finetuning).
- **Labels:** frozen-ESM likelihood — `masked_marginal` (default), `wt_marginal`
  (cheapest), or full `pll` (most faithful, opt-in).
- **Loss:** in-batch cross-entropy contrastive (same-quartile pairs are positives;
  learnable scale/bias on the similarity logit). NT-Xent and triplet are
  swappable alternatives.
- **Batching:** by default each batch is drawn from a *single gene*, and every
  gene is seen once per epoch (maximum gene diversity); genes then repeat with
  fresh variants.

## Repository layout

```
src/gsr/            installable package
  args.py           all configuration (argparse) + validation
  paths.py          scratch/output path constants (single source of truth)
  data/             uniref loading, mutagenesis, labeling, dataset, sampler, dms
  backbone/         ESM-C loading, pooling, LoRA
  scoring/          LL/PLL scorers + parquet+h5 storage
  models/           projection head (configurable MLP)
  losses/           contrastive_ce (default), ntxent, triplet
  eval/             centroid Spearman, dim-reduction, regression, provider
  train/            training loop
  utils/            stats, hashing, seeding, logging, spearman
scripts/            thin entrypoints: build_dataset, train, eval_embeddings
run/                human-editable sbatch scripts (edit vars at the top)
tests/              CPU unit tests
```

## Storage

Code lives in `/home/jv2807/gene_structure` (**code only**). Everything heavy goes
to `/scratch/jv2807/gene_structure_removal/`:

| Dir           | Contents                                             |
|---------------|------------------------------------------------------|
| `data/`       | Downloaded human UniRef90, DMS CSVs, gene splits     |
| `scores/`     | Per-variant LL/PLL + metadata (Parquet)              |
| `embeddings/` | Per-variant embeddings (HDF5 shards + `manifest.json`)|
| `runs/`       | Checkpoints, resolved args, wandb, logs              |
| `eval/`       | Standalone evaluation figures + metric tables        |

## Quickstart (NYU Torch HPC)

Everything runs inside the shared Singularity overlay via SLURM — **never on the
login node**. Each step has a `run/*.sbatch` script whose top-of-file variables
are the only thing you edit (the single place to change any hyperparameter).

```bash
# 1. Build the training dataset (score + embed sampled variants -> store/<name>)
sbatch run/build_dataset.sbatch

# 2. Train the projection head on that store
sbatch run/train.sbatch

# 3. Evaluate any embeddings — leave CHECKPOINT empty for the raw-backbone
#    baseline, or point it at a run's best.pt to evaluate the projection.
sbatch run/eval_embeddings.sbatch
```

Interactive one-offs use `srun` inside the overlay — see `run/srun_interactive.md`.
Every run prints dataset stats up front as a sanity check.

## Key options

All configuration is argparse (`src/gsr/args.py`), surfaced as editable variables
in the `run/*.sbatch` scripts. The knobs most likely to matter:

| Option | Meaning | Default |
|--------|---------|---------|
| `--esm_model` | backbone (`esmc_600m`/`esmc_300m`/`esm2_650m`) | `esmc_600m` |
| `--pooling` | `mean` / `mutated_position` / `concat` | `mean` |
| `--scorer` | label source: `masked_marginal`/`wt_marginal`/`pll` | `masked_marginal` |
| `--variants_per_gene` | variants sampled + embedded per gene | 200 |
| `--loss_type` | `contrastive_ce` / `ntxent` / `triplet` | `contrastive_ce` |
| `--distance_metric` | `cosine` / `euclidean` | `cosine` |
| `--batch_mode` | `gene_diverse` (one gene/batch) / `cross_gene` | `gene_diverse` |
| `--use_lora` | LoRA-finetune the backbone | off |
| `--eval_every_steps` | during-training centroid eval cadence | 500 |
| `--centroid_subsample` | held-out variants/type for centroids at eval time | 2000 |

## Status

Functional end-to-end. Build order (see the plan in `.claude/plans/`): Phase 0
skeleton → Phase 1 data/store → Phase 2 scoring/labels → Phase 3 head/loss/trainer
→ Phase 4 evaluators → Phase 5 LoRA/scale/hardening. Phases 0–4 are implemented,
unit-tested, and validated on real data (95,627 human UniRef90 sequences; a
300-gene dev store; DMS centroid + regression + dim-reduction baselines).

**Note on pooling:** mean pooling washes out single-residue signal (one changed
residue in a ~200-long average), so raw-backbone zero-shot scores are weak — this
is the gene-structure problem the projection is meant to fix. `concat` pooling
(mean + mutated-position) is expected to carry much more per-variant signal.

This README is updated whenever a feature is added or changed.
