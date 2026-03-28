from __future__ import annotations
from .._table_common import build_named_table
from ..config import BuildContext


def build(context: BuildContext):
    return build_named_table("tableA14_lower_tail_robustness", context.outputs / "appendix", context, ['src.next_round.lower_tail_repair'], sample="Documented appendix subsample", supports="appendix")
