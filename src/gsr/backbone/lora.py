"""LoRA wrapping for the ESM backbone (off by default).

Kept isolated so the default frozen path never imports peft. When
``--use_lora`` is set, the backbone is wrapped with a PEFT LoRA adapter on the
attention projection modules (query/key/value by default).
"""

from __future__ import annotations

from typing import Dict


def apply_lora(model, cfg: Dict):
    """Wrap ``model`` with a PEFT LoRA adapter and return the wrapped model.

    cfg keys: rank, alpha, dropout, target_modules. ESM-2 uses query/key/value;
    ESM-C/ESM++ names its attention projections differently, so if the requested
    target modules are not found we fall back to ``all-linear`` (with a warning)
    rather than crashing.
    """
    from peft import LoraConfig, get_peft_model

    targets = cfg.get("target_modules", ["query", "key", "value"])
    if isinstance(targets, (list, tuple)) and len(targets) == 1 and \
            targets[0] == "all-linear":
        targets = "all-linear"

    def _wrap(tm):
        lc = LoraConfig(r=cfg.get("rank", 8), lora_alpha=cfg.get("alpha", 16),
                        lora_dropout=cfg.get("dropout", 0.05),
                        target_modules=tm, bias="none")
        return get_peft_model(model, lc)

    try:
        wrapped = _wrap(targets)
    except ValueError as e:
        print(f"[lora] target modules {targets} not found ({e}); "
              f"falling back to 'all-linear'.")
        wrapped = _wrap("all-linear")

    trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    total = sum(p.numel() for p in wrapped.parameters())
    print(f"[lora] trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.3f}%)")
    return wrapped
