"""ESM backbone: embeddings + masked-LM logits.

Loads an ESM protein language model via HuggingFace transformers and exposes a
uniform interface for (a) per-residue / pooled embeddings from a chosen hidden
layer and (b) masked-LM logits used by the LL/PLL scorers.

Default is ESM-C 600M, served by the community HF port ``Synthyra/ESMplusplus_large``
(hidden dim 1152) -- the same model the sibling repo uses, so it is already in the
overlay's model cache. ESM-C 300M and ESM-2 650M are also selectable.

Tokenization note: these tokenizers prepend one special token (BOS/CLS), so a
1-indexed residue position ``p`` maps to token index ``p`` (residue p-1 shifted by
the leading special token).
"""

from __future__ import annotations

from typing import List, Optional

import torch

from gsr.args import AA_ALPHABET

# model alias -> (hf_repo, revision or None, hidden_dim, trust_remote_code)
_MODEL_REGISTRY = {
    "esmc_600m": ("Synthyra/ESMplusplus_large",
                  "097dfabfdd13d25847dcac278b5f49a1f133e885", 1152, True),
    "esmc_300m": ("Synthyra/ESMplusplus_small", None, 960, True),
    "esm2_650m": ("facebook/esm2_t33_650M_UR50D", None, 1280, False),
}


class ESMBackbone:
    def __init__(
        self,
        model_name: str = "esmc_600m",
        device: str = "cuda",
        use_lora: bool = False,
        lora_cfg: Optional[dict] = None,
        model_cache: Optional[str] = None,
    ):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        if model_name not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown esm_model {model_name!r}; choices: {list(_MODEL_REGISTRY)}"
            )
        repo, revision, hidden_dim, trust = _MODEL_REGISTRY[model_name]
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.device = device

        kwargs = dict(trust_remote_code=trust, cache_dir=model_cache)
        if revision:
            kwargs["revision"] = revision
        self.model = AutoModelForMaskedLM.from_pretrained(repo, **kwargs)
        # ESM++ ports attach the tokenizer to the model; ESM-2 does not.
        self.tokenizer = getattr(self.model, "tokenizer", None)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                repo, trust_remote_code=trust, cache_dir=model_cache
            )

        self.use_lora = use_lora
        if use_lora:
            from gsr.backbone.lora import apply_lora
            self.model = apply_lora(self.model, lora_cfg or {})
        else:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
        self.model.to(device)

        self.mask_token_id = self.tokenizer.mask_token_id
        # Token ids for the 20 canonical amino acids (for restricting scoring).
        self.aa_token_ids = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(list(AA_ALPHABET)),
            device=device, dtype=torch.long,
        )
        self.aa_to_col = {aa: i for i, aa in enumerate(AA_ALPHABET)}

    # --- low level ------------------------------------------------------
    def _tokenize(self, sequences: List[str]):
        tok = self.tokenizer(sequences, padding=True, return_tensors="pt")
        return tok["input_ids"].to(self.device), tok["attention_mask"].to(self.device)

    def tokenize(self, sequences: List[str]):
        """Public tokenizer: returns (input_ids, attention_mask) on device.

        Token index convention: one leading special token, so 1-indexed residue
        position ``p`` is at token index ``p``.
        """
        return self._tokenize(sequences)

    @torch.no_grad()
    def forward_logits(self, input_ids, attention_mask):
        """Masked-LM logits (B, L, vocab) for the given token batch (frozen)."""
        logits, _ = self._forward(input_ids, attention_mask,
                                  need_hidden=False, layer=-1)
        return logits

    def _forward(self, input_ids, attention_mask, need_hidden: bool, layer: int):
        want_hidden_states = need_hidden and layer != -1
        ctx = torch.enable_grad() if self.use_lora else torch.no_grad()
        with ctx:
            out = self.model(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=want_hidden_states,
            )
        logits = out["logits"] if "logits" in out else getattr(out, "logits", None)
        if need_hidden:
            if want_hidden_states:
                hidden = out["hidden_states"][layer]
            else:
                hidden = out["last_hidden_state"] if "last_hidden_state" in out \
                    else out.hidden_states[-1]
        else:
            hidden = None
        return logits, hidden

    # --- embeddings -----------------------------------------------------
    @torch.no_grad()
    def embed(
        self,
        sequences: List[str],
        layer: int = -1,
        pooling: str = "mean",
        positions: Optional[List[Optional[int]]] = None,
    ) -> torch.Tensor:
        """Return pooled embeddings for a batch of sequences.

        Args:
            layer: hidden layer index (-1 = last).
            pooling: 'mean' | 'mutated_position' | 'concat'.
            positions: 1-indexed mutated residue positions (one per sequence);
                ``None`` for a sequence (e.g. WT) means use the mean vector for the
                positional component.
        """
        input_ids, attn = self._tokenize(sequences)
        _, hidden = self._forward(input_ids, attn, need_hidden=True, layer=layer)
        mask = attn.unsqueeze(-1).to(hidden.dtype)          # (B,L,1)
        mean_pool = (hidden * mask).sum(1) / mask.sum(1)    # (B,D)

        if pooling == "mean":
            return mean_pool

        # positional component: embedding at token index == 1-indexed residue pos
        B = hidden.shape[0]
        pos_vec = mean_pool.clone()
        if positions is None:
            positions = [None] * B
        for i, p in enumerate(positions):
            if p is not None:
                tok_idx = p  # BOS offset: residue p (1-indexed) -> token p
                if tok_idx < hidden.shape[1]:
                    pos_vec[i] = hidden[i, tok_idx]
        if pooling == "mutated_position":
            return pos_vec
        if pooling == "concat":
            return torch.cat([mean_pool, pos_vec], dim=-1)
        raise ValueError(f"Unknown pooling {pooling!r}")

    def output_dim(self, pooling: str) -> int:
        return self.hidden_dim * (2 if pooling == "concat" else 1)
