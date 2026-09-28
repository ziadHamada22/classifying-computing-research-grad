"""Agent 1 — discipline classification (CS / IS / IT / SE / CE / DS)."""
from .features import format_chunk, format_chunk_obj, build_text_column

#: The Agent 1 model the system ships (a directory under ``models/``). Every
#: default -- CLI, pipeline, web UI, evaluations of "the deployed system" -- reads
#: this one name. Since 2026-09 it is SciBERT trained with the ambiguous papers
#: as soft (vote-share) targets: same accuracy, ~3x better calibrated on 2025+
#: papers, higher Computer Science recall (docs/AGENT1_COMPLETION_RESULTS.md).
#: The previous model, ``scibert``, is kept and reproducible by passing it
#: explicitly.
DEPLOYED_MODEL = "scibert-soft"

__all__ = ["format_chunk", "format_chunk_obj", "build_text_column", "DEPLOYED_MODEL"]
