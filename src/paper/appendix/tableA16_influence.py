from __future__ import annotations
from .._table_common import build_named_table
from ..config import BuildContext


def build(context: BuildContext):
    return build_named_table("tableA16_influence", context.outputs / "appendix", context, ['src.ml.interpret'], sample="Documented appendix subsample", supports="appendix")
