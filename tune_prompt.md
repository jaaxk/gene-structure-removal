# Hyperparameter-tuning session — setup + prompt

Kick off a **hypothesis-driven HP tune** of the projection head, optimizing
macro-averaged zero-shot centroid Spearman on DMS. Frozen runs go on the CPU
session node; LoRA runs go on a separate single-GPU (H200) allocation.

## 1. Launch the tuning session on a CPU node

```bash
srun --account=torch_pr_800_cds --partition=cpu_prem \
     --cpus-per-task=64 --mem=256G --time=48:00:00 --pty /bin/bash
# then inside the shell:
claude
```
(If `cpu_prem` is preemptible/time-capped on your setup, use `cs` or your usual
long CPU partition — the session must survive ~24–48 h.)

## 2. Separately request the GPU node for LoRA

```bash
salloc --account=torch_pr_800_cds --partition=h200_cds --gres=gpu:h200:1 \
       --cpus-per-task=8 --mem=64G --time=48:00:00
```
Note its **SLURM jobid** and give it to the session (it dispatches LoRA there via
`srun --jobid=<id> --overlap`).

## 3. Prompt to paste into the new session

```
You are running a hypothesis-driven hyperparameter tune for the ESM Gene Structure
Removal repo at /home/jv2807/gene_structure. First read README.md, CLAUDE.md, and
.claude/skills (code, environments). Then run continuously.

GOAL: maximize the macro-averaged zero-shot centroid Spearman on DMS — the
`centroid/spearman_mean` in each run's final_full_metrics.json (best model, full
non-subsampled centroids). Reference on the frozen 5000-gene concat setup: raw
backbone baseline = 0.288; first trained run (wt_anchored_bce, lr 1e-3, 20 epochs)
= 0.266 (did NOT beat baseline — helped OrganismalFitness -0.115->0.189, hurt Activity
0.358->0.070). Beating 0.288 macro without tanking the strong types is the bar. Always
report per-type spearman too.

DO NOT STOP until I explicitly tell you to. Keep the CPU node saturated with frozen
runs and (once I give you a GPU jobid) the GPU running one LoRA run at a time. When a
run finishes, record results and immediately launch the next hypothesis; when the
planned grid is exhausted, analyze results, form new hypotheses, and keep going.

ENVIRONMENT
- This session is on a CPU node. Run all FROZEN experiments directly here as
  concurrent background processes inside the overlay (no GPU). They reuse warm caches
  (750K concat embeddings + 5000 gene scores) — no embedding/scoring is recomputed.
- Run wrapper (CPU, no --nv):
  singularity exec --overlay /scratch/jv2807/dms_singularity/dms_contrastive.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash -c \
    "source /ext3/env.sh && export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1 && \
     cd /home/jv2807/gene_structure && PYTHONPATH=src python scripts/train.py <ARGS>"
- Concurrency: start ~8 concurrent frozen runs; watch `uptime` load vs 64 cores and
  scale to keep the node busy without thrashing. Use --num_workers 2 per run (the 13GB
  cache sits in page cache; reads are fast after warm-up). Each frozen run ~15-20 min.
- LoRA runs go on the GPU node I allocate. Ask me for its SLURM jobid, then dispatch
  each with: srun --jobid=<GPU_JOBID> --overlap --exact singularity exec --nv ...
  python scripts/train.py --use_lora ... (device auto->cuda). One at a time, <=2h each.

FIXED — DO NOT VARY (defines the comparison / keeps the cache valid):
  --fasta_path /scratch/jv2807/gene_structure_removal/data/human_uniref90.fasta
  --limit_genes 5000 --max_seq_len 400 --esm_model esmc_600m --embedding_layer -1
  --pooling concat --scorer masked_marginal --variants_per_gene 150
  --min_variants_per_gene 10 --quartile_low 0.25 --quartile_high 0.25
  --dms_selection_types Activity Binding Expression Stability OrganismalFitness
  --dms_max_per_assay 200 --eval_distance cosine --held_out_gene_frac 0.2 --seed 0
Changing any re-scores/re-embeds (hours) or changes the eval split → out of scope.
Always pass: --embeddings_mode stream --num_workers 2 --no_save_checkpoints
--device auto --wandb_project gene-structure-removal --wandb_mode online, a unique
--run_name tune_<hyp>_<idx>, and redirect stdout to
/scratch/jv2807/gene_structure_removal/tune/logs/<run_name>.log.

FREE TO SWEEP (frozen, reuse the warm cache):
  --loss_type {wt_anchored_bce, contrastive_ce, ntxent, triplet}
  --distance_metric {cosine, euclidean}; --use_learnable_scale / --no_learnable_scale
  --ntxent_temperature, --triplet_margin
  --head_hidden_dims ("512" | "1024 512" | "2048 1024 512"), --head_out_dim {128,256,512}
  --head_dropout {0,0.1,0.3}, --head_activation {relu,gelu,silu}, --head_norm
  {none,layernorm,batchnorm}
  --lr {1e-4..3e-3}, --weight_decay {0,1e-4,1e-2}, --epochs {3..10}
  --batch_size {32,64,128,256}, --batch_mode {gene_diverse,cross_gene}
  (cross_gene needs --genes_per_batch {2,4}, batch_size divisible by it)
  --eval_per_epoch 4, --centroid_subsample 2000 (during-training only; final_full,
  the ranking metric, always uses full centroids)

LoRA ARM (GPU, <=2h each, serial). LoRA eval IS backbone-aware now (it re-embeds the
DMS eval variants through the finetuned backbone; query rows capped by
--centroid_subsample during training, full at final_full), so LoRA and frozen numbers
are comparable. Reduce scope to fit 2h and keep eval cheap:
  --use_lora --limit_genes 500 --variants_per_gene 100 --epochs 5 --batch_size 32
  --eval_per_epoch 2 --centroid_subsample 400
  --esm_lr {1e-5,1e-4,5e-4} --lora_rank {4,8,16,32} --lora_alpha {16,32}
  --lora_dropout {0,0.05} --lora_target_modules {all-linear | query key value}
  (query/key/value auto-falls back to all-linear on ESM-C; scores for these 500 genes
  are already cached.) Keep the FIXED eval args (dms_*, held_out_gene_frac, seed). Also
  run a matched FROZEN control on the same --limit_genes 500 subset for a fair
  LoRA-vs-frozen comparison.

HYPOTHESES (roughly in order; adapt from results):
  1. Loss A/B: all 4 losses at a fixed head (1024 512 / out 256), lr 1e-3, 5 epochs.
  2. Preserve-backbone (training hurt strong types): lower lr {1e-4,3e-4}, fewer epochs
     {3}, shallower head ("512")/smaller out_dim {128}, higher dropout/weight_decay.
  3. Distance metric + learnable-scale on/off for the best loss.
  4. Head capacity/regularization grid for the best loss.
  5. Batching: cross_gene vs gene_diverse.
  6. LoRA arm + matched frozen control.

RESULTS (all to scratch; no model files):
  - Runs write only resolved_args.json + final_full_metrics.json + gene_splits.json to
    /scratch/jv2807/gene_structure_removal/runs/<run_name>/ (--no_save_checkpoints).
  - After each run, append one row to
    /scratch/jv2807/gene_structure_removal/tune/results.jsonl: run_name, all swept
    hyperparams, macro centroid/spearman_mean, per-type spearman_diff. Maintain a ranked
    /scratch/jv2807/gene_structure_removal/tune/SUMMARY.md (top configs, per-type
    tradeoff vs the 0.288 baseline, current best); update every ~20 runs.
  - Commit any code changes to git before running (repo code skill).

Each SUMMARY update, report the current best config and whether anything beats the
0.288 backbone baseline. Keep going until I say stop.
```

## Params kept off-limits (in FIXED)

- **Eval config** (`dms_*`, `held_out_gene_frac`, `eval_distance`, `seed`) — varying
  breaks cross-run comparability.
- **Cache-key / dataset params** (`esm_model`, `embedding_layer`, `pooling`, `scorer`,
  `limit_genes`, `max_seq_len`, `variants_per_gene`, quartiles) — varying forces hours
  of re-scoring/re-embedding (out of scope for this tune).

Everything else is fair game.
