from __future__ import annotations
from .._table_common import build_named_table
from ..config import BuildContext


def build(context: BuildContext):
    return build_named_table("tableA05_mover_switcher_balance", context.outputs / "appendix", context, ['src.analyses.switch_propensity'], sample="Documented appendix subsample", supports="appendix")
