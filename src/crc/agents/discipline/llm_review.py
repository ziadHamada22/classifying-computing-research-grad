"""Local-LLM second opinion for contested discipline predictions.

The final system uses no online model. When Agent 1's conformal prediction set
says a paper is genuinely contested, a locally downloaded instruction model can
give a second opinion. It runs on the same 6 GB RTX 3060 as the classifiers, via
llama-cpp-python (GGUF, full GPU offload), and never touches the network.

**Constrained label scoring, not free-form generation.** The model is asked, in
its own chat format, which discipline fits; instead of letting it write an
answer, each of the six discipline names is scored by the log-probability the
model assigns to it as the reply. That cannot produce an unparseable or
out-of-vocabulary answer, and it returns a real distribution that can be blended
with the classifier's rather than merely overriding it.

**One prompt pass per paper.** The prompt (rubric + paper, ~1,000 tokens) is
evaluated once; each label's two or three tokens are then scored by rewinding the
KV cache to the end of the prompt, so the six labels cost six short continuations
rather than six full prompts.

**Optional retrieved examples** (``examples=[(text, label), ...]``) are given as
prior chat turns -- the retrieval-augmented few-shot variant the proposal named.
Retrieval itself lives in ``retrieval.py``; this module only formats and scores.

The heavy dependency is imported lazily, so the rest of the system -- training,
evaluation, ingestion, the CLI -- runs with no llama-cpp installed.
``is_available()`` reports whether the second opinion can run.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from crc.taxonomy import DISCIPLINES, scope_block

PROJECT = Path(__file__).resolve().parents[4]
#: Searched in order. Bulk models live outside OneDrive (gp_data), like the corpora.
MODEL_DIRS = [
    Path(os.environ["CRC_LLM_DIR"]) if os.environ.get("CRC_LLM_DIR") else None,
    Path(r"C:\Users\ziada\gp_data\llm"),
    PROJECT / "models" / "llm",
]
MODEL_DIRS = [d for d in MODEL_DIRS if d is not None]
#: Kept for callers that referenced the old single location.
MODELS_DIR = MODEL_DIRS[-1]

# Candidate local models, in preference order for a 6 GB card (matched
# case-insensitively). All are instruction-tuned and available as GGUF.
DEFAULT_MODEL_CANDIDATES = [
    "qwen2.5-3b-instruct-q4_k_m.gguf",     # ~2.0 GB -- fits beside the classifiers
    "phi-3.5-mini-instruct-q4_k_m.gguf",   # ~2.4 GB
    "qwen2.5-7b-instruct-q4_k_m.gguf",     # ~4.7 GB -- needs the GPU mostly to itself
]

SYSTEM_PROMPT = """You classify computing research papers into exactly one of six disciplines defined by the ACM/IEEE Computing Curricula 2020:

{scopes}

Many papers touch several areas. Choose the single discipline that describes what the paper actually contributes, applying the boundary rules. Reply with the discipline name only."""


def _ensure_cuda_dlls() -> None:
    """Let llama-cpp's CUDA build find cudart/cublas on Windows.

    The prebuilt cu124 wheel links against the CUDA 12 runtime but does not ship
    it. PyTorch's CUDA build already carries exactly those DLLs, so its ``lib``
    folder is added to the DLL search path -- no system-wide CUDA install needed.
    """
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return
    try:
        import torch
        lib = Path(torch.__file__).parent / "lib"
        if lib.exists():
            os.add_dll_directory(str(lib))
    except Exception:
        pass


def _resolve_model(model_path: str | Path | None) -> Path | None:
    if model_path is not None:
        p = Path(model_path)
        return p if p.exists() else None
    for d in MODEL_DIRS:
        if not d.exists():
            continue
        present = {p.name.lower(): p for p in d.glob("*.gguf")}
        for name in DEFAULT_MODEL_CANDIDATES:
            if name in present:
                return present[name]
        if present:
            return sorted(present.values())[0]
    return None


def is_available(model_path: str | Path | None = None) -> tuple[bool, str]:
    """Whether the local second opinion can run: llama-cpp installed + a model."""
    _ensure_cuda_dlls()
    try:
        import llama_cpp  # noqa: F401
    except Exception as e:     # ImportError, or the DLL failing to load
        return False, f"llama-cpp-python unavailable: {e}"
    resolved = _resolve_model(model_path)
    if resolved is None:
        return False, (f"no GGUF model found in {[str(d) for d in MODEL_DIRS]}; "
                       f"download {DEFAULT_MODEL_CANDIDATES[0]}")
    return True, f"ready: {resolved.name}"


def build_prompt(paper: str, examples: list[tuple[str, str]] | None = None,
                 system: str | None = None) -> str:
    """ChatML prompt (Qwen's native format), ending where the reply begins.

    ``system`` replaces the discipline rubric, so the same scorer serves any label
    set (Agent 3 passes its research-design rubric).
    """
    system = system or SYSTEM_PROMPT.format(scopes=scope_block())
    parts = [f"<|im_start|>system\n{system}<|im_end|>\n"]
    for text, label in examples or []:
        parts.append(f"<|im_start|>user\nPaper:\n{text}<|im_end|>\n"
                     f"<|im_start|>assistant\n{label}<|im_end|>\n")
    parts.append(f"<|im_start|>user\nPaper:\n{paper}<|im_end|>\n<|im_start|>assistant\n")
    return "".join(parts)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


@dataclass
class LLMReview:
    """Result of one local second opinion."""

    label: str
    confidence: float
    probs: dict[str, float]
    scores: dict[str, float]                 # total log-prob of each label string
    model: str
    runner_up: str = ""
    gap: float = 0.0
    ranked: list[tuple[str, float]] = field(default_factory=list)
    n_examples: int = 0
    prompt_tokens: int = 0
    latency_ms: float = 0.0


class LocalLLMReviewer:
    """Wraps a local GGUF instruction model for constrained label scoring.

    By default the labels are the six disciplines and the rubric is Agent 1's;
    ``labels`` and ``system`` score any other closed label set the same way.
    """

    labels: list[str] = list(DISCIPLINES)
    system: str | None = None

    def __init__(self, model_path: str | Path | None = None,
                 n_gpu_layers: int = -1, n_ctx: int = 4096, verbose: bool = False,
                 labels: list[str] | None = None, system: str | None = None):
        self.labels = list(labels) if labels else list(DISCIPLINES)
        self.system = system
        _ensure_cuda_dlls()
        from llama_cpp import Llama

        resolved = _resolve_model(model_path)
        if resolved is None:
            raise FileNotFoundError(
                f"No GGUF model found (looked in {[str(d) for d in MODEL_DIRS]}). "
                f"See is_available() for guidance.")
        self.model_name = resolved.name
        self.n_ctx = n_ctx
        # n_gpu_layers=-1 offloads every layer; llama.cpp keeps what does not fit
        # on the CPU, so a larger model still runs on 6 GB, only slower.
        self.llm = Llama(model_path=str(resolved), n_gpu_layers=n_gpu_layers,
                         n_ctx=n_ctx, logits_all=False, verbose=verbose)
        self._n_vocab = self.llm.n_vocab()

    def _last_logits(self) -> np.ndarray:
        """Logits after the most recent ``eval``, read from the context.

        llama-cpp-python >= 0.3 no longer copies them into ``Llama.scores``
        unless ``logits_all=True``, which would allocate n_ctx x vocabulary
        floats (~2.5 GB for Qwen's 152k vocabulary). Only the last position is
        needed here, and llama.cpp always computes that one.
        """
        ptr = self.llm._ctx.get_logits()
        return np.ctypeslib.as_array(ptr, shape=(self._n_vocab,)).copy()

    def _label_tokens(self, prompt: str, prompt_tokens: list[int]) -> dict[str, list[int]]:
        """Each label's tokens *in context*, so merges at the boundary are exact."""
        out = {}
        for d in self.labels:
            full = self.llm.tokenize((prompt + d).encode("utf-8"),
                                     add_bos=False, special=True)
            if full[:len(prompt_tokens)] == prompt_tokens:
                out[d] = full[len(prompt_tokens):]
            else:   # a merge crossed the boundary; fall back to standalone tokens
                out[d] = self.llm.tokenize(d.encode("utf-8"), add_bos=False,
                                           special=False)
        return out

    def review(self, paper_text: str, examples: list[tuple[str, str]] | None = None,
               max_paper_chars: int = 3500) -> LLMReview:
        t0 = time.perf_counter()
        paper = paper_text.strip()[:max_paper_chars]
        examples = [(t.strip()[:1200], lab) for t, lab in (examples or [])]
        prompt = build_prompt(paper, examples, self.system)
        tokens = self.llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        if len(tokens) + 8 > self.n_ctx:          # keep the paper, drop examples
            prompt = build_prompt(paper, None, self.system)
            examples = []
            tokens = self.llm.tokenize(prompt.encode("utf-8"), add_bos=False,
                                       special=True)
        label_tokens = self._label_tokens(prompt, tokens)

        self.llm.reset()
        self.llm.eval(tokens)
        n_prompt = self.llm.n_tokens
        first_logp = _log_softmax(self._last_logits())

        scores: dict[str, float] = {}
        for d in self.labels:
            toks = label_tokens[d]
            total = float(first_logp[toks[0]])
            for i in range(1, len(toks)):
                self.llm.eval([toks[i - 1]])
                total += float(_log_softmax(self._last_logits())[toks[i]])
            scores[d] = total
            # Rewind to the end of the prompt; llama-cpp drops the KV entries past
            # n_tokens on the next eval, so the prompt is never re-processed.
            self.llm.n_tokens = n_prompt

        labels = self.labels
        arr = np.array([scores[d] for d in labels], dtype=np.float64)
        probs = np.exp(arr - arr.max())
        probs = probs / probs.sum()
        order = np.argsort(-probs)
        return LLMReview(
            label=labels[order[0]],
            confidence=float(probs[order[0]]),
            probs={d: float(p) for d, p in zip(labels, probs)},
            scores=scores,
            model=self.model_name,
            runner_up=labels[order[1]],
            gap=float(probs[order[0]] - probs[order[1]]),
            ranked=[(labels[i], float(probs[i])) for i in order],
            n_examples=len(examples),
            prompt_tokens=len(tokens),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )


def blend_with_classifier(classifier_probs: dict[str, float],
                          review: LLMReview,
                          llm_weight: float = 0.5) -> dict[str, float]:
    """Convex combination of the classifier and the local LLM distributions.

    Blending is more robust than a hard override: the LLM nudges the decision on
    genuine boundary cases without throwing away the classifier's evidence.
    """
    out = {}
    for d in DISCIPLINES:
        out[d] = ((1 - llm_weight) * classifier_probs.get(d, 0.0)
                  + llm_weight * review.probs.get(d, 0.0))
    total = sum(out.values()) or 1.0
    return {d: v / total for d, v in out.items()}


__all__ = [
    "LocalLLMReviewer",
    "LLMReview",
    "is_available",
    "build_prompt",
    "blend_with_classifier",
    "DEFAULT_MODEL_CANDIDATES",
    "MODEL_DIRS",
    "MODELS_DIR",
]
