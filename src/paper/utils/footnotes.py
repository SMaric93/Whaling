from __future__ import annotations


def standard_footnote(sample: str, unit: str, types_note: str, fe: str, cluster: str, controls: str, interpretation: str, caution: str) -> str:
    return (
        f"Sample: {sample}. Unit: {unit}. Types: {types_note}. FE: {fe}. "
        f"Clustering: {cluster}. Controls: {controls}. Interpretation: {interpretation}. Caution: {caution}."
    )
