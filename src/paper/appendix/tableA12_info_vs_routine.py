from __future__ import annotations
from .._table_common import build_named_table
from ..config import BuildContext


def build(context: BuildContext):
    return build_named_table("tableA12_info_vs_routine", context.outputs / "appendix", context, ['src.next_round.info_vs_routine'], sample="Documented appendix subsample", supports="appendix")
