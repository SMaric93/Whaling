import pandas as pd
import numpy as np
from pathlib import Path

def audit_regimes(parquet_path):
    if not Path(parquet_path).exists():
        print(f"File not found: {parquet_path}")
        return
    df = pd.read_parquet(parquet_path)
    if "regime_label" not in df.columns:
        print("No regime_label column")
        return
    
    print("Regime Counts:")
    print(df["regime_label"].value_counts())
    
    print("\nMean stats per regime:")
    stats = df.groupby("regime_label")[["speed_mps", "turning_angle_rad"]].mean()
    print(stats)
    
if __name__ == "__main__":
    audit_regimes("data/staging/steps.parquet")
