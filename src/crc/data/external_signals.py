"""Methodology evidence from arXiv metadata the labellers and the model never read.

Agent 3 has no gold labels, and every labelling function reads the paper's own
text with one shared cue vocabulary -- so a vocabulary that is wrong about a facet
biases every labeller the same way and the label model cannot see it. The arXiv
``comments`` and ``journal-ref`` fields are written by the authors about *where and
how* the work appeared, and neither the labellers nor the model ever see them:

==================  ==============  ================================================
signal              facet           what it is
==================  ==============  ================================================
code_release        builds          a code link or "code is available"
tool_demo           builds          demo / system-demonstration / tool track
theory_venue        proves          STOC, FOCS, SODA, ICALP, LICS, COLT, J. ACM, ...
hci_venue           human_data      CHI, CSCW, UIST, IUI, SIGCSE, ...
industry_track      field_context   industry / SEIP / applied-data-science track
survey_venue        secondary       ACM Computing Surveys, "review paper", ...
simulation_venue    simulates       Winter Simulation Conference, TOMACS, ...
position            (no facet)      position / vision / perspective papers
==================  ==============  ================================================

These are *weak, one-sided* signals: a theory-venue paper is very likely to prove
something, but most proofs appear elsewhere. So they are used only as an
independent check -- does the facet fire more often on papers with the signal than
without (lift) -- never as training labels, which keeps them independent.

Coverage on the 20,343-paper methodology pool: 56% have comments, 11% a
journal-ref; the signals fire on 0.03-1.8% of papers each.

Build:
    python -m crc.data.external_signals
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import pandas as pd

WORK = Path(r"C:\Users\ziada\gp_data")
CORPUS = WORK / "corpus"
SNAPSHOT = (WORK / "hf" / "hub" / "datasets--librarian-bots--arxiv-metadata-snapshot"
            / "snapshots" / "*" / "data" / "*.parquet")
OUT = CORPUS / "facets_external.parquet"

#: (regex, case-insensitive?) per signal. Venue acronyms are matched
#: case-sensitively so "chi" in prose or "focs" in a word cannot fire.
SIGNALS: dict[str, tuple[str, bool]] = {
    "code_release": (
        r"github\.com|gitlab\.|bitbucket\.org"
        r"|\b(our |the |source )?code (is |will be )?(publicly |made )?(available|released)"
        r"|source code|code and (data|models?) (are|is|will be) (available|released)|codebase",
        True),
    "tool_demo": (
        r"\bdemo(nstration)? (track|paper|session)|system demonstrations?"
        r"|tool (paper|demo|track)|tools? track|\bdemos? track"
        r"|artifact (available|evaluated|functional|reusable)",
        True),
    "theory_venue": (
        r"\b(STOC|FOCS|SODA|ICALP|LICS|CCC|COLT|ITCS|STACS|MFCS|PODC|SoCG|CONCUR|FSTTCS"
        r"|ISAAC|APPROX|RANDOM|WADS|SWAT|CSL|FSCD|CADE|IJCAR)\b"
        r"|Theoretical Computer Science|Journal of the ACM|SIAM J(ournal)?\.? ?(on )?Comput"
        r"|Logical Methods in Computer Science|Information and Computation|Algorithmica"
        r"|Combinatorica|Discrete Mathematics|Mathematics of Operations Research"
        r"|Annals of (Statistics|Probability|Applied Probability)|Bernoulli"
        r"|Electronic Journal of (Probability|Statistics)",
        False),
    "hci_venue": (
        r"\b(CHI|CSCW|UIST|IUI|TOCHI|IMWUT|UbiComp|HRI|SIGCSE|ITiCSE|ICER|L@S|CSCL"
        r"|NordiCHI|MobileHCI|DIS 20\d\d|ACM DIS)\b|Human Factors in Computing"
        r"|Human-Computer Interaction|Computers in Human Behavior"
        r"|Int(ernational)?\.? J(ournal)?\.? (of )?Human[- ]Computer",
        False),
    "industry_track": (
        r"industr(y|ial) (track|paper|session)|\bSEIP\b|software engineering in practice"
        r"|applied data science track|\bADS track|deployed (track|paper)|case study track",
        True),
    "survey_venue": (
        r"ACM Computing Surveys|Computing Surveys|Artificial Intelligence Review"
        r"|Foundations and Trends|Annual Review|\b(survey|review) (paper|article)\b"
        r"|systematic (literature )?review|\bsurvey\b(?! (data|respondents))"
        r"|literature review|mapping study",
        True),
    "simulation_venue": (
        r"Winter Simulation|\bWSC\b|SIMULTECH|Simulation Modelling Practice|SIGSIM|\bPADS\b"
        r"|Modeling and Computer Simulation|TOMACS|Simulation Conference",
        False),
    "position": (
        r"position paper|vision paper|\bperspective (paper|article)\b|keynote|invited talk"
        r"|opinion|viewpoint|essay|extended abstract of a talk|panel",
        True),
}

#: The facet each signal speaks for (``position`` speaks for none).
SIGNAL_FACET: dict[str, str] = {
    "code_release": "builds", "tool_demo": "builds", "theory_venue": "proves",
    "hci_venue": "human_data", "industry_track": "field_context",
    "survey_venue": "secondary", "simulation_venue": "simulates",
}

_COMPILED = {k: re.compile(p, re.I if ci else 0) for k, (p, ci) in SIGNALS.items()}


def detect(comments: str | None, journal_ref: str | None) -> dict[str, bool]:
    """Every signal, from one paper's comments and journal-ref."""
    text = " ".join(f"{comments or ''} || {journal_ref or ''}".split())
    return {k: bool(rx.search(text)) for k, rx in _COMPILED.items()}


def build(paper_ids: set[str], out: Path = OUT) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    ids = pa.array(sorted(paper_ids))
    frames = []
    for f in sorted(glob.glob(str(SNAPSHOT))):
        t = pq.read_table(f, columns=["id", "comments", "journal-ref"])
        frames.append(t.filter(pc.is_in(t["id"], value_set=ids)).to_pandas())
    meta = (pd.concat(frames, ignore_index=True)
              .rename(columns={"id": "paper_id", "journal-ref": "journal_ref"})
              .drop_duplicates("paper_id"))
    sig = pd.DataFrame([detect(c, j) for c, j in zip(meta["comments"], meta["journal_ref"])])
    df = pd.concat([meta.reset_index(drop=True), sig], axis=1)
    df.to_parquet(out, index=False)
    return df


def load_signals(path: Path = OUT) -> pd.DataFrame:
    """Boolean signal columns indexed by paper_id."""
    df = pd.read_parquet(path)
    return df.set_index("paper_id")[list(SIGNALS)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(CORPUS / "methodology_pool.parquet"))
    args = ap.parse_args()
    ids = set(pd.read_parquet(args.pool, columns=["paper_id"])["paper_id"].astype(str))
    df = build(ids)
    print(f"{len(df):,} of {len(ids):,} papers matched in the arXiv snapshot; "
          f"comments {df['comments'].notna().mean():.1%}, "
          f"journal-ref {df['journal_ref'].notna().mean():.1%}")
    for k in SIGNALS:
        print(f"  {k:17s} {int(df[k].sum()):>5,}  ({df[k].mean():.2%})")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()


__all__ = ["SIGNALS", "SIGNAL_FACET", "detect", "build", "load_signals"]
