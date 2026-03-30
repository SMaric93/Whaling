from __future__ import annotations

from .._table_common import build_named_table
from ..config import BuildContext


def build(context: BuildContext):
    return build_named_table("tableA03_connected_set", context.outputs / "appendix", context, ['src.analyses.connected_set'], sample="Documented appendix subsample", supports="appendix")
