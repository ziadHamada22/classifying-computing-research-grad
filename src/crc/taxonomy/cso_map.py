"""Map Computer Science Ontology (CSO) topics onto our six CC2020 disciplines.

The CSO Classifier is unsupervised and returns fine-grained CSO topics (of ~14k)
for a paper; it has no notion of the CC2020 practitioner disciplines. As with
OpenAlex, we map its high-level branch topics onto our disciplines with a
deliberate, conservative vote, and leave papers whose topics touch no mapped
branch UNMAPPED (coverage loss, never a wrong answer).

CSO is a *topic* tagger rooted at "computer science", so the root itself and
purely generic topics carry no discipline signal and are excluded. The map below
covers the high-level branches that actually discriminate between disciplines.
"""
from __future__ import annotations

from collections import Counter

from crc.taxonomy.disciplines import (
    COMPUTER_ENGINEERING as CE,
    COMPUTER_SCIENCE as CS,
    DATA_SCIENCE as DS,
    INFORMATION_SYSTEMS as IS,
    INFORMATION_TECHNOLOGY as IT,
    SOFTWARE_ENGINEERING as SE,
)

#: high-level CSO topic (lowercase) -> our discipline. Conservative: only topics
#: with a defensible one-to-one analogue. Generic roots ("computer science",
#: "computer systems") are deliberately absent.
CSO_TOPIC_TO_DISCIPLINE: dict[str, str] = {
    # Computer Science: method/theory-first
    "artificial intelligence": CS,
    "computer vision": CS,
    "natural language processing": CS,
    "theoretical computer science": CS,
    "computational complexity": CS,
    "computational geometry": CS,
    "computer graphics": CS,
    "programming languages": CS,
    "automata theory": CS,
    # Software Engineering
    "software engineering": SE,
    "software design": SE,
    "software architecture": SE,
    "software quality": SE,
    "software testing": SE,
    "program debugging": SE,
    # Information Systems: information + organisation + people
    "information retrieval": IS,
    "database systems": IS,
    "digital libraries": IS,
    "human computer interaction": IS,
    "semantic web": IS,
    "world wide web": IS,
    "social networks": IS,
    "recommender systems": IS,
    "ontology": IS,
    # Information Technology: infrastructure / operations / security
    "computer networks": IT,
    "network security": IT,
    "computer security": IT,
    "cryptography": IT,
    "distributed systems": IT,
    "distributed computer systems": IT,
    "operating systems": IT,
    "cloud computing": IT,
    "internet of things": IT,
    "wireless telecommunication systems": IT,
    # Computer Engineering: hardware / device / signal / control
    "computer hardware": CE,
    "integrated circuits": CE,
    "embedded systems": CE,
    "field programmable gate arrays": CE,
    "signal processing": CE,
    "image processing": CE,
    "control systems": CE,
    "robotics": CE,
    "sensors": CE,
    # Data Science: statistics-first / mining
    "data mining": DS,
    "big data": DS,
    "machine learning": DS,          # judgement call: CSO nests ML under AI; we
                                     # send it to DS to match our CC2020 split
    "bioinformatics": DS,
    "statistics": DS,
}

#: excluded generic / root topics that would otherwise vote spuriously
CSO_IGNORE = {"computer science", "computer systems", "engineering", "mathematics"}


def map_topics(topics) -> str | None:
    """Vote a paper's CSO topics into one discipline (argmax), or None if no
    topic maps or the top two disciplines tie."""
    votes: Counter[str] = Counter()
    for t in topics or []:
        key = str(t).lower().strip()
        if key in CSO_IGNORE:
            continue
        d = CSO_TOPIC_TO_DISCIPLINE.get(key)
        if d:
            votes[d] += 1
    if not votes:
        return None
    top = votes.most_common(2)
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None                  # tie -> abstain (conservative)
    return top[0][0]


__all__ = ["CSO_TOPIC_TO_DISCIPLINE", "CSO_IGNORE", "map_topics"]
