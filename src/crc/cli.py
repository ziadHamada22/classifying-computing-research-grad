"""Command-line entry point.

    python -m crc.cli classify paper.pdf
    python -m crc.cli classify "We propose a new transformer architecture..."
    python -m crc.cli classify paper.pdf --json --show-chunks
    python -m crc.cli inspect paper.pdf          # ingestion only, no model

`inspect` exists because most surprises in this system come from parsing, not
from the classifier: it shows the detected document type, the recovered section
skeleton and the chunk plan without loading a model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
from crc.agents.discipline import DEPLOYED_MODEL  # noqa: E402

DEFAULT_MODEL = PROJECT / "models" / DEPLOYED_MODEL


def cmd_inspect(args: argparse.Namespace) -> int:
    from crc.ingest import ingest

    t0 = time.perf_counter()
    doc = ingest(args.source)
    elapsed = time.perf_counter() - t0

    print(f"source      : {doc.source}")
    print(f"parser      : {doc.parser}")
    print(f"doc_type    : {doc.doc_type.value} "
          f"(confidence {doc.meta.get('doctype_confidence', 0):.2f})")
    for r in doc.meta.get("doctype_reasons", []):
        print(f"              - {r}")
    print(f"title       : {doc.title}")
    print(f"words       : {doc.n_words:,}")
    print(f"parse time  : {elapsed*1000:.0f} ms")
    print(f"\nsections ({len(doc.sections)}):")
    for s in doc.sections:
        print(f"  {s.section.value:18s} {len(s.text.split()):>6,}w  "
              f"<- {s.heading_raw[:50]!r}")
    print(f"\nchunks ({len(doc.chunks)}): {doc.section_summary()}")
    if args.show_chunks:
        for c in doc.chunks:
            print(f"  [{c.index:3d}] {c.section.value:14s} {c.n_words:>4d}w  "
                  f"{c.text[:90]!r}")
    return 0


def cmd_classify(args: argparse.Namespace) -> int:
    from crc.agents.discipline.predict import DisciplineClassifier
    from crc.ingest import ingest

    model_dir = Path(args.model)
    if not model_dir.exists():
        print(f"error: model not found at {model_dir}\n"
              f"Train one first:\n"
              f"  python -m crc.agents.discipline.train --tag scibert",
              file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    doc = ingest(args.source)
    t_parse = time.perf_counter() - t0

    t1 = time.perf_counter()
    clf = DisciplineClassifier.load(model_dir, strategy=args.strategy)
    t_load = time.perf_counter() - t1

    t2 = time.perf_counter()
    res = clf.classify_document(doc)
    t_infer = time.perf_counter() - t2

    if args.json:
        payload = json.loads(res.to_json())
        payload["timing_ms"] = {
            "parse": round(t_parse * 1000, 1),
            "model_load": round(t_load * 1000, 1),
            "inference": round(t_infer * 1000, 1),
        }
        if not args.show_chunks:
            payload.pop("chunks", None)
        print(json.dumps(payload, indent=2))
        return 0

    print(f"\n  {res.label}   {res.confidence:.1%}")
    print(f"  runner-up: {res.runner_up} (gap {res.gap:.3f})")
    print(f"\n  document type : {res.doc_type}  ({res.n_chunks} chunks, "
          f"parser={res.parser})")
    print(f"  aggregation   : {res.strategy}")
    if res.borderline:
        print(f"  ** BORDERLINE ** {res.borderline_reason}")
        print("     -> would be sent to the local-LLM second opinion")
    print("\n  full distribution:")
    for name, p in sorted(res.probs.items(), key=lambda kv: -kv[1]):
        bar = "#" * int(round(p * 40))
        print(f"    {name:24s} {p:6.1%} {bar}")
    if res.section_mass:
        print("\n  evidence weight by section:")
        for sec, share in sorted(res.section_mass.items(), key=lambda kv: -kv[1]):
            print(f"    {sec:18s} {share:6.1%}")
    if args.show_chunks:
        print("\n  per-chunk:")
        for c in res.chunks:
            print(f"    [{c.index:3d}] {c.section:14s} w={c.weight:5.2f} "
                  f"{c.label:22s} {c.confidence:.2f}  {c.text_preview[:60]!r}")
    print(f"\n  timing: parse {t_parse*1000:.0f} ms | load {t_load*1000:.0f} ms | "
          f"inference {t_infer*1000:.0f} ms")
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Agent 1 -> Agent 2 on one document."""
    from crc.pipeline import Pipeline

    pipe = Pipeline.load(ensemble=args.ensemble)
    if pipe.field_clf is None:
        print("note: Agent 2 not trained yet — showing discipline only.\n",
              file=sys.stderr)

    res = pipe.run(args.source)
    if args.json:
        print(res.to_json())
        return 0

    for w in res.warnings:
        print(f"\n  WARNING: {w}")
    if res.title:
        print(f"\n  {res.title}")
    print(f"\n  {res.discipline}   {res.discipline_confidence:.1%}")
    if res.field:
        print(f"    └─ {res.field}   {res.field_confidence:.1%}")
    print(f"\n  {res.doc_type} · {res.n_chunks} chunks")

    if res.design:
        print(f"\n  research design: {res.design}   "
              f"{res.design_confidence:.1%}")
        if res.facets_present:
            print(f"    facets observed: {', '.join(res.facets_present)}")
        else:
            print("    facets observed: none above threshold")
        print(f"    worldview {res.worldview} · method "
              f"{res.research_method} (derived from the design, not predicted)")
        if res.derived_rule:
            label = ("from similar papers" if res.design_source == "similar_papers"
                     else "rule")
            print(f"    {label}: {res.derived_rule}")
        if res.design_read_from and res.design_read_from != "abstract":
            print(f"    read from '{res.design_read_from}'")
        if res.similar_papers:
            print("    most similar reference papers:")
            for s in res.similar_papers:
                print(f"      {s['similarity']:.2f}  {s['title'][:70]}  "
                      f"[{s['paper_id']}] -> {s['design']}")

    if res.alternatives:
        print(f"\n  ALTERNATIVE READINGS (discipline was contested):")
        for a in res.alternatives:
            print(f"    {a['discipline']} ({a['discipline_confidence']:.1%})"
                  f"  └─ {a['field']} ({a['field_confidence']:.1%})")

    if res.alternative:
        a = res.alternative
        print(f"\n  ALTERNATIVE READING (discipline was borderline):")
        print(f"    {a['discipline']} ({a['discipline_confidence']:.1%})"
              f"  └─ {a['field']} ({a['field_confidence']:.1%})")

    for n in res.notes:
        print(f"\n  ! {n}")

    print("\n  discipline distribution:")
    for name, p in sorted(res.discipline_probs.items(), key=lambda kv: -kv[1]):
        print(f"    {name:26s} {p:6.1%} {'#' * int(round(p * 34))}")
    if res.field_probs:
        print(f"\n  field distribution (within {res.discipline}):")
        for name, p in sorted(res.field_probs.items(), key=lambda kv: -kv[1]):
            print(f"    {name:46s} {p:6.1%} {'#' * int(round(p * 26))}")

    t = res.timing_ms
    print(f"\n  timing: ingest {t['ingest']:.0f} ms | discipline "
          f"{t['discipline']:.0f} ms | field {t['field']:.0f} ms | "
          f"total {t['total']:.0f} ms")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="crc", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_i = sub.add_parser("inspect", help="Parse and chunk without classifying.")
    p_i.add_argument("source", help="File path or raw text.")
    p_i.add_argument("--show-chunks", action="store_true")
    p_i.set_defaults(func=cmd_inspect)

    p_c = sub.add_parser("classify", help="Predict the discipline.")
    p_c.add_argument("source", help="File path or raw text.")
    p_c.add_argument("--model", default=str(DEFAULT_MODEL))
    p_c.add_argument("--strategy", default="weighted_mean",
                     choices=["weighted_mean", "weighted_geometric", "mean", "max"])
    p_c.add_argument("--json", action="store_true")
    p_c.add_argument("--show-chunks", action="store_true")
    p_c.set_defaults(func=cmd_classify)

    p_p = sub.add_parser("pipeline",
                         help="Discipline then field, with uncertainty carried.")
    p_p.add_argument("source", help="File path or raw text.")
    p_p.add_argument("--ensemble", action="store_true",
                     help="Use the discipline ensemble rather than one model.")
    p_p.add_argument("--json", action="store_true")
    p_p.set_defaults(func=cmd_pipeline)
    return ap


def main(argv: list[str] | None = None) -> int:
    from crc.ingest import IngestError

    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, IngestError) as exc:
        # An input the system cannot read is the user's to fix, not a crash.
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
