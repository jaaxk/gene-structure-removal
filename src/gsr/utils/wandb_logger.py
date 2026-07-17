"""Thin wandb wrapper with online/offline/disabled modes.

Kept minimal so training/eval code just calls ``log`` / ``log_figure`` without
caring whether wandb is active. When disabled it is a no-op.
"""

from __future__ import annotations

from typing import Dict


class WandbLogger:
    def __init__(self, project: str, run_name: str, config: dict,
                 mode: str = "online", dir: str | None = None):
        self.mode = mode
        self.enabled = mode != "disabled"
        self._wandb = None
        if self.enabled:
            import wandb

            self._wandb = wandb
            wandb.init(project=project, name=run_name, config=config,
                       mode=mode, dir=dir)

    def log(self, metrics: Dict[str, float], step: int | None = None) -> None:
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def log_figure(self, name: str, fig, step: int | None = None) -> None:
        if self._wandb is not None:
            self._wandb.log({name: self._wandb.Image(fig)}, step=step)

    def finish(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
