from __future__ import annotations

from .._table_common import build_named_table
from ..config import BuildContext


def build(context: BuildContext):
    return build_named_table("tableA04_type_robustness", context.outputs / "appendix", context, ['src.reinforcement.type_estimation'], sample="Documented appendix subsample", supports="appendix")
