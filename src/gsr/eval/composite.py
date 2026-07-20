"""Wrap several DMS evaluators into one object Trainer can treat as a single
evaluator, so which metric drives best-checkpoint selection is just a CLI
flag (``--primary_metric``) rather than a hardcoded attribute on one class.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np


class CompositeEvaluator:
    def __init__(self, evaluators: Dict[str, object], primary_metric: str):
        self.evaluators = evaluators
        self.primary_metric = primary_metric

    def evaluate(self, project_fn: Callable[[np.ndarray], np.ndarray],
                 **kwargs) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for evaluator in self.evaluators.values():
            out.update(evaluator.evaluate(project_fn, **kwargs))
        return out

    def reprepare(self, subsample: int) -> None:
        for evaluator in self.evaluators.values():
            if hasattr(evaluator, "reprepare"):
                evaluator.reprepare(subsample)

    def save_splits(self, path) -> None:
        for evaluator in self.evaluators.values():
            if hasattr(evaluator, "save_splits"):
                evaluator.save_splits(path)
