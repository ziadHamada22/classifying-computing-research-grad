"""The proposal's Agent 3: "SciBERT + RAG few-shot" -- a local LLM names the design.

The proposal (slides 7 and 9) specified Agent 3 as retrieval-augmented few-shot
prompting: retrieve the k most similar labelled papers, give them to an LLM as
examples, and ensemble its answer with the fine-tuned model's. This module is that
component, built on the same machinery Agent 1's second opinion uses
(`agents/discipline/llm_review.py`):

* a local GGUF instruction model (Qwen2.5-3B-Instruct, llama-cpp, 6 GB GPU) --
  nothing leaves the machine;
* **constrained label scoring** over the nine design names, so the answer is always
  one of the nine and comes with a distribution that can be ensembled;
* the rubric is the taxonomy's own design descriptions and boundary rules
  (`taxonomy/methodology.py`), the single source of truth;
* examples come from the reference store (`retrieval.py`): the most similar
  training papers *that had evidence*, each with the design the label model gave
  it, the most similar placed last, next to the paper being classified.

Whether it helps is measured in `eval/evaluate_agent3_llm.py`; it is not in the
deployed path unless that measurement says so.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from crc.agents.methodology.retrieval import ReferenceStore
from crc.taxonomy.methodology import DESIGN_NAMES, scope_block

DESIGN_SYSTEM_PROMPT = """You classify the research design of computing research papers into exactly one of nine designs:

{scopes}

Many papers both build something and evaluate it. Choose the single design that describes how the paper's main claim is established, applying the boundary rules. Reply with the design name only."""


def design_rubric() -> str:
    return DESIGN_SYSTEM_PROMPT.format(scopes=scope_block())


def design_reviewer(model_path: str | Path | None = None, n_ctx: int = 4096):
    """A `LocalLLMReviewer` that scores the nine design names."""
    from crc.agents.discipline.llm_review import LocalLLMReviewer

    return LocalLLMReviewer(model_path=model_path, n_ctx=n_ctx,
                            labels=list(DESIGN_NAMES), system=design_rubric())


def retrieved_examples(store: ReferenceStore, query: np.ndarray,
                       texts: dict[str, str], k: int = 4,
                       exclude: str | None = None) -> list[tuple[str, str]]:
    """The k most similar reference papers with evidence, as (abstract, design).

    Ordered least to most similar, so the closest example sits next to the paper.
    ``exclude`` drops the query paper itself if it is in the store.
    """
    idx, _ = store.search(query, k * 4)
    out = []
    for i in idx:
        pid = store.paper_ids[i]
        if not store.has_evidence[i] or pid == exclude or pid not in texts:
            continue
        out.append((texts[pid], str(store.designs[i])))
        if len(out) == k:
            break
    return out[::-1]


__all__ = ["DESIGN_SYSTEM_PROMPT", "design_reviewer", "design_rubric",
           "retrieved_examples"]
