from __future__ import annotations
from .._table_common import build_named_table
from ..config import BuildContext


def build(context: BuildContext):
    return build_named_table("tableA08_patch_definition", context.outputs / "appendix", context, ['src.reinforcement.patch_spells'], sample="Documented appendix subsample", supports="appendix")
