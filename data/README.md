# Data

The final labelled datasets. Files are the originals used for training and
evaluation, stored with Git LFS. Each `*_stats.json` summarises its dataset.

| File | Rows | Used by | What it is |
|---|---:|---|---|
| `corpus_v2.parquet` | 74,160 | Agent 1 | arXiv computing papers from 2015 to 2024, balanced across the six disciplines. `split`: 50,400 train, 10,800 validation, 12,960 test. |
| `corpus_v2_temporal.parquet` | 9,000 | Agent 1 | Papers from 2025 onward (1,500 per discipline), the temporal test set. |
| `fields_corpus_st.parquet` | 126,674 | Agent 2 | Papers with a field label. `split`: 93,090 train (78,333 rule-labelled plus 14,757 added by self-training), 16,780 validation, 16,804 test. |
| `facets_pool_v2.parquet` | 20,343 | Agent 3 | Facet labels for papers from 2022 and 2023: the Dawid-Skene probability of each of the 8 facets, the derived design and the rule that produced it. `split`: 14,238 train, 3,050 validation, 3,055 test. |
| `wos_pool.parquet` | 9,462 | Agents 1 and 2 | Web of Science abstracts mapped to the six disciplines, used only as an external test set. |

## How the labels were made

- **Discipline (`discipline`, `ambiguous`, `margin`).** Each of 46 arXiv categories is
  mapped to a discipline with a strength between 0 and 1. A paper's categories vote,
  with the primary category counted twice; a paper whose winner leads by less than 25%
  of the vote is marked ambiguous. Ambiguous papers are kept out of train and
  validation but kept in the test sets. The map is `src/crc/taxonomy/arxiv_map.py`.
- **Field (`field`, `field_evidence`, `field_runner_up`).** Within the paper's
  discipline, each field scores 3 points for a matching primary category, 2 per
  keyword phrase in the title and 1 per phrase in the abstract; the winner is kept
  only if it leads by at least 20% of the points. `field_evidence` records whether
  the label came from the category, keywords, or both. See
  `src/crc/data/label_fields.py` and `src/crc/data/self_train_fields.py`.
- **Facets (`p_*` columns).** Five labelling functions, one per region of the paper
  (title, abstract, introduction, methods, results and conclusion), vote from cue
  phrases, and a Dawid-Skene model combines them without gold labels. See
  `src/crc/data/label_facets.py` and `src/crc/taxonomy/facets.py`.

## Sources

- Titles, abstracts and categories: the arXiv metadata snapshot
  `librarian-bots/arxiv-metadata-snapshot` on Hugging Face.
- Agent 3's full texts: `neuralwork/arxiver` on Hugging Face, joined on `paper_id`
  (the arXiv id). The full texts are not redistributed here; only the labels are.
- Web of Science: WoS-46985, Kowsari et al., "HDLTex: Hierarchical deep learning for
  text classification", ICMLA 2017.
- The 9,136 ambiguous papers that Agent 1's deployed model also trains on (as soft
  vote-share targets) are listed by arXiv id in
  `models/scibert-soft/added_ambiguous_ids.json`; their text comes from the same
  arXiv metadata snapshot.
