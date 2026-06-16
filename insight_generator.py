"""
backend/insight_generator.py
Automated initial intelligent insight generation from a DataFrame.
"""
import pandas as pd
import numpy as np


def generate_insights(df: pd.DataFrame) -> list[str]:
    """Return safe automated insight strings for the active dataset."""
    insights: list[str] = []

    if df is None or df.empty:
        return ["Dataset belum tersedia. Silakan upload dataset terlebih dahulu."]

    total_rows, total_cols = df.shape
    num_df = df.select_dtypes(include=[np.number])
    cat_df = df.select_dtypes(include=["object", "category", "bool"])

    missing_total = int(df.isna().sum().sum())
    duplicate_total = int(df.duplicated().sum())
    total_cells = max(total_rows * total_cols, 1)

    insights.append(
        f"📌 Dataset memiliki **{total_rows:,} baris** dan **{total_cols:,} kolom**."
    )
    insights.append(
        f"🔢 Komposisi data: **{num_df.shape[1]} kolom numerik** dan **{cat_df.shape[1]} kolom kategorikal**."
    )

    if missing_total > 0:
        missing_pct = missing_total / total_cells * 100
        top_missing = df.isna().sum().sort_values(ascending=False)
        top_missing = top_missing[top_missing > 0].head(3)
        detail = ", ".join([f"{col} ({int(val):,})" for col, val in top_missing.items()])
        insights.append(
            f"⚠️ Terdapat **{missing_total:,} missing values** ({missing_pct:.2f}% dari total sel). Kolom paling terdampak: {detail}."
        )
    else:
        insights.append("✅ Tidak ditemukan missing value pada dataset.")

    if duplicate_total > 0:
        insights.append(
            f"♻️ Ditemukan **{duplicate_total:,} baris duplikat**. Disarankan menjalankan fitur Data Cleaning."
        )
    else:
        insights.append("✅ Tidak ditemukan baris duplikat.")

    if not num_df.empty:
        means = num_df.mean(numeric_only=True).dropna()
        if not means.empty:
            top_mean_col = means.idxmax()
            insights.append(
                f"📈 Kolom numerik dengan rata-rata tertinggi adalah **{top_mean_col}** ({means.loc[top_mean_col]:,.2f})."
            )

        stds = num_df.std(numeric_only=True).dropna()
        if not stds.empty:
            top_std_col = stds.idxmax()
            insights.append(
                f"📊 Kolom dengan variasi terbesar berdasarkan standar deviasi adalah **{top_std_col}** ({stds.loc[top_std_col]:,.4f})."
            )

        outlier_counts = {}
        for col in num_df.columns:
            s = pd.to_numeric(num_df[col], errors="coerce").dropna()
            if len(s) < 4:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            outlier_counts[col] = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        if outlier_counts:
            top_out = max(outlier_counts, key=outlier_counts.get)
            insights.append(
                f"🔴 Potensi outlier terbanyak terdapat pada kolom **{top_out}** sebanyak **{outlier_counts[top_out]:,}** data."
            )

        # Safe correlation: tidak memakai np.fill_diagonal(corr.values) agar tidak error read-only.
        if num_df.shape[1] >= 2:
            corr = num_df.corr(numeric_only=True).abs().copy(deep=True)
            for col in corr.columns:
                corr.loc[col, col] = np.nan
            stacked = corr.stack().dropna()
            if not stacked.empty:
                idx = stacked.idxmax()
                insights.append(
                    f"🔗 Korelasi numerik terkuat: **{idx[0]}** dan **{idx[1]}** (r = {stacked.loc[idx]:.3f})."
                )
            else:
                insights.append("🔗 Korelasi numerik belum dapat dihitung karena data tidak cukup bervariasi.")
    else:
        insights.append("🔢 Dataset belum memiliki kolom numerik untuk statistik dan korelasi otomatis.")

    if not cat_df.empty:
        for col in cat_df.columns[:2]:
            vc = df[col].value_counts(dropna=True)
            if not vc.empty:
                top_val = vc.index[0]
                top_pct = vc.iloc[0] / max(len(df), 1) * 100
                insights.append(
                    f"🏷️ Kategori dominan pada **{col}** adalah **{top_val}** ({top_pct:.1f}% dari baris data)."
                )

    insights.append(
        "💡 Initial intelligent insight generation berhasil membaca struktur dataset, kualitas data, missing value, duplikasi, outlier, kategori dominan, dan korelasi awal."
    )
    return insights
