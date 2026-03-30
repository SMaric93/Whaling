from __future__ import annotations

from .._table_common import build_named_table
from ..config import BuildContext


def build(context: BuildContext):
    return build_named_table("tableA18_archival_mechanisms", context.outputs / "appendix", context, ['src.next_round.archival_mechanisms'], sample="Documented appendix subsample", supports="appendix")
