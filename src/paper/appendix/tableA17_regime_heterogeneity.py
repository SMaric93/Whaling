from __future__ import annotations
from .._table_common import build_named_table
from ..config import BuildContext


def build(context: BuildContext):
    return build_named_table("tableA17_regime_heterogeneity", context.outputs / "appendix", context, ['src.compass.regimes', 'src.analyses.ml_prediction'], sample="Documented appendix subsample", supports="appendix")
