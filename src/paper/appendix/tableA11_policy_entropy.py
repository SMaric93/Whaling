from __future__ import annotations
from .._table_common import build_named_table
from ..config import BuildContext


def build(context: BuildContext):
    return build_named_table("tableA11_policy_entropy", context.outputs / "appendix", context, ['src.next_round.policy_entropy'], sample="Documented appendix subsample", supports="appendix")
