from __future__ import annotations
from .._table_common import build_named_table
from ..config import BuildContext


def build(context: BuildContext):
    return build_named_table("tableA01_included_excluded", context.outputs / "appendix", context, ['src.analyses.data_loader'], sample="Documented appendix subsample", supports="appendix")
