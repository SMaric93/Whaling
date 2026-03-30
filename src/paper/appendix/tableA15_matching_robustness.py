from __future__ import annotations

from .._table_common import build_named_table
from ..config import BuildContext


def build(context: BuildContext):
    return build_named_table("tableA15_matching_robustness", context.outputs / "appendix", context, ['src.next_round.risk_matching'], sample="Documented appendix subsample", supports="appendix")
