"""Discipline taxonomy: CC2020 scope definitions and the arXiv category map."""
from .arxiv_map import (
    AMBIGUITY_MARGIN,
    ARXIV_CATEGORIES,
    BY_CATEGORY,
    CATEGORY_MAPPINGS,
    EXCLUDED,
    CategoryMapping,
    PaperLabel,
    coverage_summary,
    label_paper,
    v1_to_v2_changes,
)
from .disciplines import (
    DISCIPLINE_ABBR,
    DISCIPLINES,
    ID2LABEL,
    LABEL2ID,
    SCOPES,
    DisciplineScope,
    scope_block,
)
from .fields import (
    FIELD_NAMES_BY_DISCIPLINE,
    FIELDS,
    FIELDS_BY_DISCIPLINE,
    Field,
    n_fields,
)
from .fields import label2id as field_label2id
from .fields import id2label as field_id2label
from .fields import scope_block as field_scope_block
from .fields import summary as field_summary

__all__ = [
    "DISCIPLINES",
    "DISCIPLINE_ABBR",
    "LABEL2ID",
    "ID2LABEL",
    "SCOPES",
    "DisciplineScope",
    "scope_block",
    "CategoryMapping",
    "CATEGORY_MAPPINGS",
    "BY_CATEGORY",
    "ARXIV_CATEGORIES",
    "EXCLUDED",
    "PaperLabel",
    "label_paper",
    "v1_to_v2_changes",
    "coverage_summary",
    "AMBIGUITY_MARGIN",
    # field level (Agent 2)
    "Field",
    "FIELDS",
    "FIELDS_BY_DISCIPLINE",
    "FIELD_NAMES_BY_DISCIPLINE",
    "n_fields",
    "field_label2id",
    "field_id2label",
    "field_scope_block",
    "field_summary",
]
