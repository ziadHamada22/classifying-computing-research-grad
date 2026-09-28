"""Computing Research Classifier — the final system.

Three coordinated agents over an ingested document:

  Agent 1  discipline    (6-way: CS / IS / IT / SE / CE / DS)
  Agent 2  field         (conditioned on the predicted discipline)
  Agent 3  methodology   (design / method / worldview)

Superseded by nothing; the earlier `prototype/` tree is kept only as the
archived artefact referenced by the preliminary report.
"""

__version__ = "2.0.0.dev0"
