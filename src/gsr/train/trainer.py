"""Training loop for the projection head (frozen-embedding path).

Consumes precomputed embeddings from a VariantStore, trains the projection head
with the selected contrastive loss, logs to wandb, runs an optional evaluator on
a cadence and again at the end of training, and checkpoints the best model by the
evaluator's primary metric (or the final model when no evaluator is attached).

The optional ``evaluator`` must expose ``.primary_metric`` (str) and
``.evaluate(project_fn) -> dict[str, float]`` where ``project_fn`` maps a numpy
embedding matrix to its projected version using the current head.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gsr import paths
from gsr.data.sampler import GeneBatchSampler
from gsr.losses.registry import build_loss
from gsr.models.projection_head import ProjectionHead
from gsr.utils.wandb_logger import WandbLogger


class Trainer:
    def __init__(self, args, dataset, evaluator=None):
        self.args = args
        self.dataset = dataset
        self.evaluator = evaluator
        self.device = args.device if torch.cuda.is_available() else "cpu"

        self.head = ProjectionHead(
            input_dim=dataset.input_dim,
            hidden_dims=args.head_hidden_dims,
            out_dim=args.head_out_dim,
            dropout=args.head_dropout,
            activation=args.head_activation,
            norm=args.head_norm,
        ).to(self.device)
        self.loss_fn = build_loss(args).to(self.device)

        params = list(self.head.parameters()) + list(self.loss_fn.parameters())
        self.optim = torch.optim.AdamW(params, lr=args.lr,
                                       weight_decay=args.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.amp and
                                                self.device == "cuda")

        self.sampler = GeneBatchSampler(
            dataset, batch_size=args.batch_size, batch_mode=args.batch_mode,
            genes_per_batch=args.genes_per_batch,
            balance_labels=args.balance_labels, seed=args.seed)
        self.loader = DataLoader(dataset, batch_sampler=self.sampler, num_workers=0)

        self.run_dir = paths.run_dir(args.run_name)
        paths.ensure_dirs(self.run_dir)
        self.wandb = WandbLogger(args.wandb_project, args.run_name, vars(args),
                                 mode=args.wandb_mode, dir=str(self.run_dir))
        self.best_metric = -float("inf")
        self.global_step = 0

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

    def _run_eval(self, tag: str) -> None:
        if self.evaluator is None:
            return
        metrics = self.evaluator.evaluate(self._project_fn())
        logged = {f"eval/{k}": v for k, v in metrics.items()}
        self.wandb.log(logged, step=self.global_step)
        primary = metrics.get(self.evaluator.primary_metric, float("nan"))
        print(f"[eval:{tag}] {self.evaluator.primary_metric}={primary:.4f}")
        if primary == primary and primary > self.best_metric:  # not nan and better
            self.best_metric = primary
            self._save_checkpoint("best.pt")
            print(f"[eval:{tag}] new best -> best.pt")

    # --- checkpoint -----------------------------------------------------
    def _save_checkpoint(self, name: str) -> None:
        ckpt = {
            "head_state": self.head.state_dict(),
            "loss_state": self.loss_fn.state_dict(),
            "config": vars(self.args),
            "input_dim": self.dataset.input_dim,
            "out_dim": self.args.head_out_dim,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
        }
        torch.save(ckpt, self.run_dir / name)

    # --- main loop ------------------------------------------------------
    def fit(self) -> None:
        args = self.args
        with open(self.run_dir / "resolved_args.json", "w") as fh:
            json.dump(vars(args), fh, indent=2)

        for epoch in range(args.epochs):
            self.sampler.set_epoch(epoch)
            self.head.train()
            pbar = tqdm(self.loader, desc=f"epoch {epoch}")
            for emb, y, _gene in pbar:
                emb = emb.to(self.device)
                y = y.to(self.device)
                self.optim.zero_grad()
                with torch.cuda.amp.autocast(enabled=args.amp and
                                             self.device == "cuda"):
                    z = self.head(emb)
                    loss, metrics = self.loss_fn(z, y)
                self.scaler.scale(loss).backward()
                if args.grad_clip:
                    self.scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.head.parameters()) +
                        list(self.loss_fn.parameters()), args.grad_clip)
                self.scaler.step(self.optim)
                self.scaler.update()

                self.global_step += 1
                pbar.set_postfix(loss=metrics.get("loss", 0.0))
                self.wandb.log({f"train/{k}": v for k, v in metrics.items()},
                               step=self.global_step)

                if args.eval_every_steps and \
                        self.global_step % args.eval_every_steps == 0:
                    self._run_eval(tag=f"step{self.global_step}")

        self._run_eval(tag="final")
        self._save_checkpoint("final.pt")
        print(f"[train] done. checkpoints in {self.run_dir}")
        self.wandb.finish()
