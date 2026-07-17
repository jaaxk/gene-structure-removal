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
- **Pooling (`concat` default):** each variant is `[mean-pool, mutated-residue
  embedding]`. The wild-type is pooled at the **same** mutated residue, so a
  variant is compared to its WT like-for-like at that position. `mean` and
  `mutated_position` are also available.
- **Labels:** frozen-ESM likelihood — `masked_marginal` (default), `wt_marginal`
  (cheapest), or full `pll` (most faithful, opt-in).
- **Loss (`wt_anchored_bce` default):** each mutant is pulled toward / pushed from
  **its own WT** by cross-entropy on the label (close vs far likelihood).
  `contrastive_ce` (in-batch pairs *between mutants*, mirroring `dms_contrastive`),
  `ntxent`, and `triplet` are swappable alternatives.
- **Batching:** by default each batch is drawn from a *single gene*, and every
  gene is seen once per epoch (maximum gene diversity); genes then repeat with
  fresh variants.
- **LoRA:** with `--use_lora` the backbone is finetuned and embeddings are computed
  live from sequences each step (frozen cached path otherwise).

## Repository layout

```
src/gsr/            installable package
  args.py           all configuration (argparse) + validation
  paths.py          scratch/output path constants (single source of truth)
  cache/            content-addressed score + embedding caches (shared across runs)
  data/             uniref loading, mutagenesis, labeling, training_data, datasets,
                    sampler, dms
  backbone/         ESM-C loading, pooling, LoRA
  scoring/          LL/PLL scorers + parquet+h5 store (utility)
  models/           projection head (configurable MLP)
  losses/           wt_anchored_bce (default), contrastive_ce, ntxent, triplet
  eval/             centroid Spearman, dim-reduction, regression, dms cache
  train/            training loop
  utils/            stats, hashing, seeding, spearman, wandb
scripts/            thin entrypoints: train, eval_embeddings
run/                human-editable sbatch scripts (edit vars at the top)
tests/              CPU unit tests
```

## Storage

Code lives in `/home/jv2807/gene_structure` (**code only**). Everything heavy goes
to `/scratch/jv2807/gene_structure_removal/`:

| Dir                   | Contents                                                  |
|-----------------------|-----------------------------------------------------------|
| `data/`               | Downloaded human UniRef90 FASTA                            |
| `cache/scores/`       | Per-gene LL/PLL for all variants (Parquet), by model+scorer|
| `cache/embeddings/`   | Per-variant mut+WT embeddings (HDF5), by model+layer+pooling|
| `eval/dms_cache/`     | Frozen ESM embeddings of DMS eval variants                |
| `runs/`               | Checkpoints (`best.pt`/`final.pt`), resolved args, wandb  |
| `eval/`               | Standalone evaluation figures + metric tables             |

## Quickstart (NYU Torch HPC)

Everything runs inside the shared Singularity overlay via SLURM — **never on the
login node**. Each step has a `run/*.sbatch` script whose top-of-file variables
are the only thing you edit (the single place to change any hyperparameter).

```bash
# 1. Train the projection head directly from the UniRef FASTA. Scores and
#    embeddings are computed lazily and CACHED on first use (content-addressed),
#    then reused by later runs — there is no separate build step. On a GPU node
#    this self-warms the caches then trains.
sbatch run/train.sbatch

# (optional) Pre-fill all caches on GPU without training — useful before running
# cheap CPU head-training / hyperparameter sweeps on a large set.
sbatch run/warm_cache.sbatch

# 2. Evaluate any embeddings — leave CHECKPOINT empty for the raw-backbone
#    baseline, or point it at a run's best.pt to evaluate the projection.
sbatch run/eval_embeddings.sbatch
```

**Device is auto-detected** (`--device auto`): a GPU node trains + self-warms the
cache; once embeddings are cached you can run cheap head-training / HP sweeps by
submitting the *same* job to a **CPU node** (no GPU → not subject to the low-util
canceller). A CPU run with un-warmed caches errors and tells you to warm on GPU
first. LoRA runs stay on GPU. **Embedding reads auto-select** `ram` (bulk-load
resident, small sets) vs `stream` (worker-prefetch from H5, bounded memory) by
`--max_resident_gb`, so training scales to arbitrarily large sets without holding
everything in RAM and without per-batch I/O stalls.

Interactive one-offs use `srun` inside the overlay — see `run/srun_interactive.md`.
Every run prints dataset stats up front as a sanity check.

**Caching & GPU utilization.** Per-gene LL/PLL scores are cached under
`cache/scores/<model>_<scorer>/` and per-variant embeddings under
`cache/embeddings/<model>_L<layer>_<pooling>/`, keyed by content so different runs
reuse the overlap and different pooling/scorer configs never collide. To keep GPU
utilization high, a run **bulk-loads** its embeddings once and holds them resident
(no per-batch H5 reads); cold embeddings are computed live on the first epoch
(which keeps the GPU busy) and then cached. A missing embedding is never fatal —
it is computed live and saved unless another writer holds the cache lock, in which
case it stays in memory for that run with a warning.

## Key options

All configuration is argparse (`src/gsr/args.py`), surfaced as editable variables
in the `run/*.sbatch` scripts. The knobs most likely to matter:

| Option | Meaning | Default |
|--------|---------|---------|
| `--esm_model` | backbone (`esmc_600m`/`esmc_300m`/`esm2_650m`) | `esmc_600m` |
| `--pooling` | `mean` / `mutated_position` / `concat` | `concat` |
| `--scorer` | label source: `masked_marginal`/`wt_marginal`/`pll` | `masked_marginal` |
| `--variants_per_gene` | variants sampled + embedded per gene | 200 |
| `--loss_type` | `wt_anchored_bce` / `contrastive_ce` / `ntxent` / `triplet` | `wt_anchored_bce` |
| `--distance_metric` | `cosine` / `euclidean` | `cosine` |
| `--batch_mode` | `gene_diverse` (one gene/batch) / `cross_gene` | `gene_diverse` |
| `--device` | `auto` / `cuda` / `cpu` (auto-detects the node) | `auto` |
| `--embeddings_mode` | `auto` / `ram` / `stream` embedding reads | `auto` |
| `--use_lora` | LoRA-finetune the backbone | off |
| `--eval_every_steps` | during-training centroid eval cadence | 500 |
| `--centroid_subsample` | held-out variants/type for centroids at eval time | 2000 |

## Status

Functional end-to-end. Build order (see the plan in `.claude/plans/`): Phase 0
skeleton → Phase 1 data/store → Phase 2 scoring/labels → Phase 3 head/loss/trainer
→ Phase 4 evaluators → Phase 5 LoRA/scale/hardening. Phases 0–4 are implemented,
unit-tested, and validated on real data (95,627 human UniRef90 sequences; a
300-gene dev store; DMS centroid + regression + dim-reduction baselines).

**Note on pooling:** `mean` pooling washes out single-residue signal (one changed
residue in a ~200-long average), giving near-zero raw-backbone zero-shot scores —
the gene-structure problem the projection targets. The default is now `concat`
(mean + mutated-residue), which retains the local signal; the WT is pooled at the
same mutated residue so the mutant/WT comparison is like-for-like.

This README is updated whenever a feature is added or changed.
