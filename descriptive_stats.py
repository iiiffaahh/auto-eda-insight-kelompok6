"""
backend/descriptive_stats.py
Advanced descriptive statistics for numeric and categorical columns.
"""
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats


def numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with advanced stats for each numeric column.
    """
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) == 0:
        return pd.DataFrame()

    rows = []
    for col in num_cols:
        s = df[col].dropna()
        missing_count = df[col].isnull().sum()
        missing_pct = round(missing_count / len(df) * 100, 2)

        # Normality test (Shapiro if n<=5000, else skip)
        if len(s) >= 3:
            try:
                _, p_val = scipy_stats.shapiro(s[:5000])
                normal = "Normal" if p_val > 0.05 else "Not Normal"
            except Exception:
                normal = "N/A"
        else:
            normal = "N/A"

        # Outliers via IQR
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())

        rows.append({
            "Column": col,
            "Count": len(s),
            "Mean": round(s.mean(), 4),
            "Median": round(s.median(), 4),
            "Min": round(s.min(), 4),
            "Max": round(s.max(), 4),
            "Std Dev": round(s.std(), 4),
            "Variance": round(s.var(), 4),
            "Mode": round(s.mode().iloc[0], 4) if len(s.mode()) > 0 else "N/A",
            "Skewness": round(s.skew(), 4),
            "Kurtosis": round(s.kurtosis(), 4),
            "Missing Count": missing_count,
            "Missing %": f"{missing_pct}%",
            "Normality": normal,
            "# Outliers": outliers,
        })
    return pd.DataFrame(rows)


def categorical_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with stats for each categorical column.
    """
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    if len(cat_cols) == 0:
        return pd.DataFrame()

    rows = []
    for col in cat_cols:
        s = df[col].dropna()
        missing_count = df[col].isnull().sum()
        missing_pct = round(missing_count / len(df) * 100, 2)
        mode_val = s.mode().iloc[0] if len(s.mode()) > 0 else "N/A"
        mode_freq = int((s == mode_val).sum()) if mode_val != "N/A" else 0
        mode_pct = round(mode_freq / len(df) * 100, 2) if len(df) > 0 else 0

        rows.append({
            "Column": col,
            "Count": len(s),
            "Unique Categories": s.nunique(),
            "Mode": mode_val,
            "Mode Frequency": mode_freq,
            "Mode %": f"{mode_pct}%",
            "Missing Count": missing_count,
            "Missing %": f"{missing_pct}%",
        })
    return pd.DataFrame(rows)
