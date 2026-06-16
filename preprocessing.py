"""
backend/data_cleaning.py
Data cleaning operations with before/after snapshots.
"""
import pandas as pd
import numpy as np


def snapshot(df: pd.DataFrame) -> dict:
    """Capture a summary snapshot of the dataframe for before/after comparison."""
    return {
        "shape": df.shape,
        "missing_total": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_per_col": df.isnull().sum().to_dict(),
    }


def drop_duplicates(df: pd.DataFrame):
    before = snapshot(df)
    df_clean = df.drop_duplicates()
    after = snapshot(df_clean)
    removed = before["shape"][0] - after["shape"][0]
    return df_clean, before, after, f"Removed {removed} duplicate row(s)."


def drop_missing_rows(df: pd.DataFrame):
    before = snapshot(df)
    df_clean = df.dropna()
    after = snapshot(df_clean)
    removed = before["shape"][0] - after["shape"][0]
    return df_clean, before, after, f"Removed {removed} row(s) with missing values."


def fill_missing_mean(df: pd.DataFrame):
    before = snapshot(df)
    df_clean = df.copy()
    num_cols = df_clean.select_dtypes(include="number").columns
    df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())
    after = snapshot(df_clean)
    return df_clean, before, after, "Filled missing numeric values with column mean."


def fill_missing_median(df: pd.DataFrame):
    before = snapshot(df)
    df_clean = df.copy()
    num_cols = df_clean.select_dtypes(include="number").columns
    df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
    after = snapshot(df_clean)
    return df_clean, before, after, "Filled missing numeric values with column median."


def fill_missing_mode(df: pd.DataFrame):
    before = snapshot(df)
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            mode = df_clean[col].mode()
            if len(mode) > 0:
                df_clean[col].fillna(mode.iloc[0], inplace=True)
    after = snapshot(df_clean)
    return df_clean, before, after, "Filled all missing values with column mode."


def drop_column(df: pd.DataFrame, col: str):
    before = snapshot(df)
    df_clean = df.drop(columns=[col])
    after = snapshot(df_clean)
    return df_clean, before, after, f"Dropped column '{col}'."


def convert_dtype(df: pd.DataFrame, col: str, target_type: str):
    before = snapshot(df)
    df_clean = df.copy()
    try:
        df_clean[col] = df_clean[col].astype(target_type)
        msg = f"Converted '{col}' to {target_type}."
    except Exception as e:
        msg = f"Failed to convert '{col}': {e}"
    after = snapshot(df_clean)
    return df_clean, before, after, msg
