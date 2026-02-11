import os, sys
import pandas as pd

# make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

RAW = "data/raw/ds_mujer_individual_full.csv"
OUT = "data/processed/encuesta_fecundidad_mujeres.csv"

# 1) load raw
df = pd.read_csv(RAW)

# 2) preprocess
df = preprocess_data(df, target_col="nr_children")

# 3) ensure target is 0/1 only if still object
if "nr_children" in df.columns and df["nr_children"].dtype == "object":
    df["nr_children"] = df["nr_children"].str.strip().map({"No": 0, "Yes": 1}).astype("Int64")

# sanity checks
assert df["nr_children"].isna().sum() == 0, "nr_children has NaNs after preprocess"
assert set(df["nr_children"].unique()) <= {0, 1}, "nr_children not 0/1 after preprocess"

# 4) features
df_processed = build_features(df, target_col="nr_children")

# 5) save
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df_processed.to_csv(OUT, index=False)
print(f"✅ Processed dataset saved to {OUT} | Shape: {df_processed.shape}")
