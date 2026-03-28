from __future__ import annotations
from .._table_common import build_named_table
from ..config import BuildContext


def build(context: BuildContext):
    return build_named_table("tableA07_same_vessel_same_captain", context.outputs / "appendix", context, ['src.analyses.vessel_mover_analysis'], sample="Documented appendix subsample", supports="appendix")
