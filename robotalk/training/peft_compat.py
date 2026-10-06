"""Narrow compatibility helpers for the installed PEFT/Transformers pair."""

from __future__ import annotations


def ensure_peft_tensor_parallel_import_compatibility() -> None:
    """Supply PEFT's newer TP import on older non-TP Transformers installs.

    PEFT imports ``EmbeddingParallel`` unconditionally while loading any LoRA
    adapter, before it checks whether the model is actually tensor parallel.
    The Transformers build in the RoboCasa environment predates that symbol.
    Our training and live-sim models are ordinary DDP/single-device models, so
    the embedding branch is never used; the alias only lets PEFT complete its
    import and reach that check.
    """

    from transformers.integrations import tensor_parallel

    if not hasattr(tensor_parallel, "EmbeddingParallel"):
        tensor_parallel.EmbeddingParallel = tensor_parallel.ColwiseParallel
