"""Load the BRFSS 2015 diabetes file and make a stratified train/validation/test split."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "diabetes_binary_health_indicators_BRFSS2015.csv"
TARGET = "Diabetes_binary"
SEED = 42

AGE_GROUPS = {  # BRFSS 13-level age code -> broader band
    **{c: "18-34" for c in (1, 2, 3)},
    **{c: "35-49" for c in (4, 5, 6)},
    **{c: "50-64" for c in (7, 8, 9)},
    **{c: "65+" for c in (10, 11, 12, 13)},
}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the CSV; the row number becomes an explicit `id` index (the file has none)."""
    df = pd.read_csv(path)
    df[TARGET] = df[TARGET].astype(int)
    df.index.name = "id"
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != TARGET]


def split_data(df: pd.DataFrame, seed: int = SEED):
    """60/20/20 split, stratified on the target. Returns (train, val, test) keeping the id index."""
    train, rest = train_test_split(df, test_size=0.4, stratify=df[TARGET], random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest[TARGET], random_state=seed)
    return train, val, test


def describe_balance(df: pd.DataFrame) -> str:
    n_pos = int(df[TARGET].sum())
    return f"{len(df):,} rows, {n_pos:,} positive (diabetes or prediabetes) ({n_pos / len(df):.1%}); ratio about 1:{(len(df) - n_pos) / n_pos:.1f}"
