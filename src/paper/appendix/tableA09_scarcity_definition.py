from __future__ import annotations

from .._table_common import build_named_table
from ..config import BuildContext


def build(context: BuildContext):
    return build_named_table("tableA09_scarcity_definition", context.outputs / "appendix", context, ['src.reinforcement.search_metrics'], sample="Documented appendix subsample", supports="appendix")
