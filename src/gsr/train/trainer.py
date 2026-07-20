"""Training loop for the projection head.

Two paths share this trainer:

- **Frozen (default):** the backbone is frozen, so the dataset serves precomputed
  (mut, wt) embeddings and the loop is head-only and cheap.
- **LoRA live:** the backbone is LoRA-finetuned, so embeddings cannot be cached;
  the dataset serves sequences and the loop embeds mutant + WT (at the mutated
  position) on the fly through the LoRA backbone each step. The optimizer trains
  the head + loss params at ``--lr`` and the LoRA adapters at ``--esm_lr``.

Both project mut and wt through the same head and call ``loss(z_mut, z_wt, y)``.
The evaluator (if attached) runs on a cadence and at the end of training; the best
checkpoint is kept by the evaluator's primary metric.

Note: during-training centroid eval currently projects the *frozen* DMS embedding
cache through the head. For LoRA runs this reflects the head but not the LoRA
backbone's effect on DMS embeddings -- a LoRA-aware eval is a follow-up.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gsr import paths
from gsr.data.sampler import GeneBatchSampler
from gsr.data.seq_dataset import collate_sequences
from gsr.data.stream_dataset import StreamingVariantDataset, stream_worker_init
from gsr.losses.registry import build_loss
from gsr.models.projection_head import ProjectionHead
from gsr.utils.device import resolve_device
from gsr.utils.wandb_logger import WandbLogger


class Trainer:
    def __init__(self, args, dataset, evaluator=None, backbone=None):
        self.args = args
        self.dataset = dataset
        self.evaluator = evaluator
        self.use_lora = args.use_lora
        self.device = resolve_device(args.device)

        # LoRA live path embeds through this (finetuned) backbone each step; the
        # frozen path holds resident embeddings and needs no backbone in the loop.
        self.backbone = backbone
        if self.use_lora:
            assert self.backbone is not None, "LoRA trainer requires a backbone"
            input_dim = self.backbone.output_dim(args.pooling)
        else:
            input_dim = dataset.input_dim

        self.head = ProjectionHead(
            input_dim=input_dim, hidden_dims=args.head_hidden_dims,
            out_dim=args.head_out_dim, dropout=args.head_dropout,
            activation=args.head_activation, norm=args.head_norm).to(self.device)
        self.loss_fn = build_loss(args).to(self.device)

        param_groups = [{"params": list(self.head.parameters()) +
                         list(self.loss_fn.parameters()), "lr": args.lr}]
        if self.use_lora:
            lora_params = [p for p in self.backbone.model.parameters()
                           if p.requires_grad]
            param_groups.append({"params": lora_params, "lr": args.esm_lr})
        self.optim = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

        self.amp_enabled = args.amp and self.device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        self.sampler = GeneBatchSampler(
            dataset, batch_size=args.batch_size, batch_mode=args.batch_mode,
            genes_per_batch=args.genes_per_batch,
            balance_labels=args.balance_labels, seed=args.seed)
        if isinstance(dataset, StreamingVariantDataset):
            # Worker-prefetch streaming: reads overlap compute, bounded memory.
            self.loader = DataLoader(
                dataset, batch_sampler=self.sampler,
                num_workers=max(1, args.num_workers),
                prefetch_factor=args.prefetch_factor,
                pin_memory=(self.device == "cuda"),
                worker_init_fn=stream_worker_init, persistent_workers=True)
        else:
            collate = collate_sequences if self.use_lora else None
            self.loader = DataLoader(dataset, batch_sampler=self.sampler,
                                     num_workers=0, collate_fn=collate)

        # Eval cadence: derive from eval_per_epoch unless an explicit override.
        steps_per_epoch = max(1, len(self.sampler))
        if args.eval_every_steps > 0:
            self.eval_cadence = args.eval_every_steps
        elif args.eval_per_epoch > 0:
            per_epoch = args.eval_per_epoch
            if self.use_lora:
                per_epoch = min(per_epoch, 2)  # re-embedding DMS is costly
            self.eval_cadence = max(1, steps_per_epoch // per_epoch)
        else:
            self.eval_cadence = 0
        print(f"[train] steps/epoch={steps_per_epoch} eval every "
              f"{self.eval_cadence} steps")

        self.run_dir = paths.run_dir(args.run_name)
        paths.ensure_dirs(self.run_dir)
        if self.evaluator is not None and hasattr(self.evaluator, "save_splits"):
            self.evaluator.save_splits(self.run_dir / "gene_splits.json")
        self.wandb = WandbLogger(args.wandb_project, args.run_name, vars(args),
                                 mode=args.wandb_mode, dir=str(self.run_dir))
        self.best_metric = -float("inf")
        self.global_step = 0
        self.save_ckpt = not getattr(args, "no_save_checkpoints", False)
        self._best_head_state = None  # RAM copy of the best head (for final eval)
        self._best_lora_state = None  # RAM copy of the best LoRA adapters

    def _trainable_params(self):
        params = list(self.head.parameters()) + list(self.loss_fn.parameters())
        if self.use_lora:
            params += [p for p in self.backbone.model.parameters()
                       if p.requires_grad]
        return params

    def _embed_batch(self, batch):
        """Return (mut_emb, wt_emb, y) on device for either dataset type."""
        a = self.args
        if self.use_lora:
            pos = batch["positions"]
            mut = self.backbone.embed(batch["mut_seqs"], layer=a.embedding_layer,
                                      pooling=a.pooling, positions=pos)
            wt = self.backbone.embed(batch["wt_seqs"], layer=a.embedding_layer,
                                     pooling=a.pooling, positions=pos)
            assert mut.shape == wt.shape, (
                f"mut/wt embedding shape mismatch: {tuple(mut.shape)} vs "
                f"{tuple(wt.shape)}")
            return mut, wt, batch["labels"].to(self.device)
        mut_emb, wt_emb, y, _gene = batch
        assert mut_emb.shape == wt_emb.shape, (
            f"mut/wt embedding shape mismatch: {tuple(mut_emb.shape)} vs "
            f"{tuple(wt_emb.shape)}")
        return (mut_emb.to(self.device), wt_emb.to(self.device), y.to(self.device))

    # --- projection callback for the evaluator --------------------------
    def _project_fn(self):
        head, device = self.head, self.device

        def fn(emb: np.ndarray) -> np.ndarray:
            head.eval()
            with torch.no_grad():
                x = torch.from_numpy(np.asarray(emb, dtype=np.float32)).to(device)
                z = head(x).cpu().numpy()
            head.train()
            return z
        return fn

    def _eval_kwargs(self) -> dict:
        """For LoRA, re-embed DMS variants through the live backbone so the metric
        reflects the finetuned backbone (not the frozen cache)."""
        if self.use_lora:
            return dict(backbone=self.backbone, embed_args=dict(
                layer=self.args.embedding_layer, pooling=self.args.pooling,
                batch_size=self.args.score_batch_size))
        return {}

    def _lora_state(self) -> dict:
        return {k: v.detach().cpu().clone()
                for k, v in self.backbone.model.state_dict().items()
                if "lora" in k.lower()}

    def _run_eval(self, tag: str) -> None:
        if self.evaluator is None:
            return
        metrics = self.evaluator.evaluate(self._project_fn(), **self._eval_kwargs())
        self.wandb.log({f"eval/{k}": v for k, v in metrics.items()},
                       step=self.global_step)
        primary = metrics.get(self.evaluator.primary_metric, float("nan"))
        print(f"[eval:{tag}] {self.evaluator.primary_metric}={primary:.4f}")
        if primary == primary and primary > self.best_metric:
            self.best_metric = primary
            # Keep the best head (and LoRA adapters) in RAM for the final eval;
            # write to disk only when checkpointing is enabled.
            self._best_head_state = {k: v.detach().cpu().clone()
                                     for k, v in self.head.state_dict().items()}
            if self.use_lora:
                self._best_lora_state = self._lora_state()
            if self.save_ckpt:
                self._save_checkpoint("best.pt")
            print(f"[eval:{tag}] new best ({primary:.4f})")

    def _save_checkpoint(self, name: str) -> None:
        ckpt = {
            "head_state": self.head.state_dict(),
            "loss_state": self.loss_fn.state_dict(),
            "config": vars(self.args),
            "input_dim": self.head.input_dim,
            "out_dim": self.args.head_out_dim,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
        }
        if self.use_lora:
            ckpt["lora_state"] = {k: v.cpu() for k, v in
                                  self.backbone.model.state_dict().items()
                                  if "lora" in k.lower()}
        torch.save(ckpt, self.run_dir / name)

    def fit(self) -> None:
        args = self.args
        with open(self.run_dir / "resolved_args.json", "w") as fh:
            json.dump(vars(args), fh, indent=2)

        for epoch in range(args.epochs):
            self.sampler.set_epoch(epoch)
            self.head.train()
            if self.use_lora:
                self.backbone.model.train()
            pbar = tqdm(self.loader, desc=f"epoch {epoch}")
            for batch in pbar:
                self.optim.zero_grad()
                with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                    mut_emb, wt_emb, y = self._embed_batch(batch)
                    z_mut = self.head(mut_emb)
                    z_wt = self.head(wt_emb)
                    loss, metrics = self.loss_fn(z_mut, z_wt, y)
                self.scaler.scale(loss).backward()
                if args.grad_clip:
                    self.scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self._trainable_params(),
                                                   args.grad_clip)
                self.scaler.step(self.optim)
                self.scaler.update()

                self.global_step += 1
                pbar.set_postfix(loss=metrics.get("loss", 0.0))
                self.wandb.log({f"train/{k}": v for k, v in metrics.items()},
                               step=self.global_step)

                if self.eval_cadence and \
                        self.global_step % self.eval_cadence == 0:
                    self._run_eval(tag=f"step{self.global_step}")

        self._run_eval(tag="final")
        if self.save_ckpt:
            self._save_checkpoint("final.pt")
        self._final_full_eval()
        print(f"[train] done. outputs in {self.run_dir}")
        self.wandb.finish()

    def _final_full_eval(self) -> None:
        """Evaluate the BEST checkpoint on FULL (non-subsampled) centroids."""
        if self.evaluator is None:
            return
        # Use the best head kept in RAM (works with or without disk checkpoints).
        head = self.head
        if self._best_head_state is not None:
            head = ProjectionHead(
                input_dim=self.head.input_dim,
                hidden_dims=self.args.head_hidden_dims,
                out_dim=self.args.head_out_dim, dropout=self.args.head_dropout,
                activation=self.args.head_activation, norm=self.args.head_norm)
            head.load_state_dict(self._best_head_state)
            head.eval().to(self.device)

        def project(emb):
            with torch.no_grad():
                x = torch.from_numpy(np.asarray(emb, dtype=np.float32)).to(self.device)
                return head(x).cpu().numpy()

        # For LoRA, restore the best adapters so the final eval re-embeds DMS
        # through the best backbone (not just the best head).
        if self.use_lora and self._best_lora_state is not None:
            self.backbone.model.load_state_dict(self._best_lora_state, strict=False)

        self.evaluator.reprepare(0)  # full centroids
        metrics = self.evaluator.evaluate(project, **self._eval_kwargs())
        self.wandb.log({f"final_full/{k}": v for k, v in metrics.items()},
                       step=self.global_step)
        print(f"[eval:final_full] best-ckpt primary metric "
              f"({self.evaluator.primary_metric})="
              f"{metrics.get(self.evaluator.primary_metric, float('nan')):.4f}")

        # LLR-vs-projection-effect-score figure, from the best checkpoint.
        llr_eval = getattr(self.evaluator, "evaluators", {}).get("llr_projection")
        if llr_eval is not None:
            from gsr.eval.llr_figure import make_llr_projection_figure
            table = llr_eval.effect_table(project, **self._eval_kwargs())
            rho = metrics.get("llr_projection/spearman", float("nan"))
            fig_path = make_llr_projection_figure(
                table, rho, self.run_dir / "llr_projection_figure.png")
            self.wandb.log_figure("final_full/llr_projection_figure", str(fig_path),
                                  step=self.global_step)

        # LoRA also finetunes the backbone itself, so the raw (post-LoRA,
        # pre-head) embeddings are a second candidate representation --
        # compute both and keep whichever scores higher.
        if self.use_lora:
            identity_metrics = self.evaluator.evaluate(
                lambda emb: np.asarray(emb, dtype=np.float32), **self._eval_kwargs())
            self.wandb.log(
                {f"final_full_backbone/{k}": v for k, v in identity_metrics.items()},
                step=self.global_step)
            head_score = metrics.get(self.evaluator.primary_metric, float("nan"))
            backbone_score = identity_metrics.get(
                self.evaluator.primary_metric, float("nan"))
            best_source = "backbone" if backbone_score > head_score else "head"
            print(f"[eval:final_full] backbone-only macro Spearman="
                  f"{backbone_score:.4f} (best_source={best_source})")
            metrics = {
                **{f"head/{k}": v for k, v in metrics.items()},
                **{f"backbone/{k}": v for k, v in identity_metrics.items()},
                "best_source": best_source,
                "best_spearman_mean": max(head_score, backbone_score),
            }

        import json
        with open(self.run_dir / "final_full_metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)
