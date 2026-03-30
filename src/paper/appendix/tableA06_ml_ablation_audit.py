from __future__ import annotations

from .._table_common import build_named_table
from ..config import BuildContext


def build(context: BuildContext):
    return build_named_table("tableA06_ml_ablation_audit", context.outputs / "appendix", context, ['src.next_round.repairs.ablation_feature_audit'], sample="Documented appendix subsample", supports="appendix")
