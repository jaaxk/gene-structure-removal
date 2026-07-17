"""Build the training dataset: score + embed sampled variants -> VariantStore.

For each wild-type gene (human UniRef90 by default) this:
  1. samples ``--variants_per_gene`` single-aa variants,
  2. scores WT + variants with a frozen ESM likelihood (``--scorer``),
  3. embeds WT + variants (frozen ESM, chosen layer + pooling) -- ONLY the sampled
     variants, since per-variant embedding is the real compute cost,
  4. assigns per-gene quartile labels,
  5. writes one store shard (parquet scores + h5 embeddings + manifest).

SLURM array friendly: pass ``--num_shards N --shard_id i`` and each task processes
genes ``[i::N]`` and writes ``shard_<i>``. Heavy -- run via sbatch, not on login node.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from gsr import paths
from gsr.args import POOLING_CHOICES, SCORER_CHOICES
from gsr.backbone.esm import ESMBackbone
from gsr.data.labeling import assign_labels
from gsr.data.mutagenesis import sample_variants
from gsr.data.uniref import load_wildtypes
from gsr.scoring.scorer import score_gene
from gsr.scoring.store import VariantStore
from gsr.utils.hashing import seq_hash
from gsr.utils.seeding import seed_everything
from gsr.utils.stats import print_dataset_stats


def build_parser():
    p = argparse.ArgumentParser(description="Build variant score+embedding store.")
    p.add_argument("--dataset_name", required=True,
                   help="Store subdir under <scratch>/store/<dataset_name>/.")
    p.add_argument("--fasta_path", default=str(paths.DATA_DIR / "human_uniref90.fasta"))
    p.add_argument("--esm_model", default="esmc_600m")
    p.add_argument("--embedding_layer", type=int, default=-1)
    p.add_argument("--pooling", default="mean", choices=POOLING_CHOICES)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--scorer", default="masked_marginal", choices=SCORER_CHOICES)
    p.add_argument("--score_batch_size", type=int, default=8)
    p.add_argument("--variants_per_gene", type=int, default=200)
    p.add_argument("--min_variants_per_gene", type=int, default=10)
    p.add_argument("--quartile_low", type=float, default=0.25)
    p.add_argument("--quartile_high", type=float, default=0.25)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--limit_genes", type=int, default=0,
                   help="Cap total genes (after filtering) for smoke tests. 0=all.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p


def _gene_rows(gene_id, wt_seq, variants, scores, pooling, backbone, emb_layer, bs):
    """Assemble score rows + row-aligned embeddings for one gene (WT + variants)."""
    rows = [dict(gene_id=gene_id, variant_id=seq_hash(wt_seq), mutant="WT",
                 pos=0, wt_aa="", mut_aa="", seq_len=len(wt_seq), is_wt=True,
                 wt_score=0.0, mut_score=0.0, delta=0.0, abs_delta=0.0)]
    seqs = [wt_seq]
    positions = [None]  # WT positional-pooling component falls back to mean

    for v, (ws, ms, delta) in zip(variants, scores):
        rows.append(dict(gene_id=gene_id, variant_id=seq_hash(v.sequence),
                         mutant=v.mutant, pos=v.pos, wt_aa=v.wt_aa, mut_aa=v.mut_aa,
                         seq_len=len(v.sequence), is_wt=False,
                         wt_score=ws, mut_score=ms, delta=delta,
                         abs_delta=abs(delta)))
        seqs.append(v.sequence)
        positions.append(v.pos)

    embs = []
    for start in range(0, len(seqs), bs):
        e = backbone.embed(seqs[start:start + bs], layer=emb_layer, pooling=pooling,
                           positions=positions[start:start + bs])
        embs.append(e.float().cpu().numpy())
    return pd.DataFrame(rows), np.concatenate(embs, axis=0)


def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)

    records = load_wildtypes(args.fasta_path, max_seq_len=args.max_seq_len)
    records.sort(key=lambda r: r.gene_id)  # deterministic ordering for sharding
    if args.limit_genes:
        records = records[: args.limit_genes]
    shard_records = records[args.shard_id::args.num_shards]
    print(f"[build] shard {args.shard_id}/{args.num_shards}: "
          f"{len(shard_records)} of {len(records)} genes")

    device = args.device if torch.cuda.is_available() else "cpu"
    backbone = ESMBackbone(args.esm_model, device=device)
    print(f"[build] backbone {args.esm_model} dim={backbone.hidden_dim} "
          f"pooling={args.pooling} -> head_input_dim="
          f"{backbone.output_dim(args.pooling)}")

    rng = np.random.default_rng(args.seed + args.shard_id)
    gene_frames, gene_embs = [], []
    for rec in tqdm(shard_records, desc=f"shard {args.shard_id}"):
        variants = sample_variants(rec.gene_id, rec.sequence,
                                   n=args.variants_per_gene, rng=rng)
        if len(variants) < args.min_variants_per_gene:
            continue
        scores = score_gene(backbone, rec.sequence, variants,
                            scorer=args.scorer, batch_size=args.score_batch_size)
        df, emb = _gene_rows(rec.gene_id, rec.sequence, variants, scores,
                             args.pooling, backbone, args.embedding_layer,
                             args.score_batch_size)
        gene_frames.append(df)
        gene_embs.append(emb)

    if not gene_frames:
        print("[build] no genes produced; nothing to write.")
        return

    # Concatenate with a positional index that stays aligned with `embeddings`.
    df = pd.concat(gene_frames, ignore_index=True)
    embeddings = np.concatenate(gene_embs, axis=0)
    assert len(df) == len(embeddings)

    # assign_labels preserves row index and may drop whole genes; use the
    # surviving index to filter the embedding matrix in lockstep.
    labeled = assign_labels(df, quartile_low=args.quartile_low,
                            quartile_high=args.quartile_high,
                            min_variants=args.min_variants_per_gene)
    keep_idx = labeled.index.to_numpy()
    embeddings = embeddings[keep_idx]
    labeled = labeled.reset_index(drop=True)

    print_dataset_stats(labeled, title=f"build shard {args.shard_id}")

    store = VariantStore(paths.SCRATCH_ROOT / "store" / args.dataset_name)
    shard_name = f"shard_{args.shard_id:04d}"
    store.write_part(shard_name, labeled, embeddings)
    print(f"[build] wrote {shard_name}: {len(labeled)} rows, "
          f"emb {embeddings.shape} -> {store.base}")


if __name__ == "__main__":
    main()
