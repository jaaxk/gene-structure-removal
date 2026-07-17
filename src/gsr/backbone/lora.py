"""LoRA wrapping for the ESM backbone (off by default).

Kept isolated so the default frozen path never imports peft. When
``--use_lora`` is set, the backbone is wrapped with a PEFT LoRA adapter on the
attention projection modules (query/key/value by default).
"""

from __future__ import annotations

from typing import Dict


def apply_lora(model, cfg: Dict):
    """Wrap ``model`` with a PEFT LoRA adapter and return the wrapped model.

    cfg keys: rank, alpha, dropout, target_modules.
    """
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=cfg.get("rank", 8),
        lora_alpha=cfg.get("alpha", 16),
        lora_dropout=cfg.get("dropout", 0.05),
        target_modules=cfg.get("target_modules", ["query", "key", "value"]),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.3f}%)")
    return model
