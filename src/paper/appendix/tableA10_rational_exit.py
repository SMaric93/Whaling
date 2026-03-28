from __future__ import annotations
from .._table_common import build_named_table
from ..config import BuildContext


def build(context: BuildContext):
    return build_named_table("tableA10_rational_exit", context.outputs / "appendix", context, ['src.next_round.rational_exit_tests'], sample="Documented appendix subsample", supports="appendix")
