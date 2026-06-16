"""Robust data loader for Auto EDA Insight."""
import json
import io
import pandas as pd


def _reset(f):
    try:
        f.seek(0)
    except Exception:
        pass


def _try_read_csv(uploaded_file):
    attempts = [
        {"sep": None, "engine": "python"},
        {"sep": ","},
        {"sep": ";"},
        {"sep": "\t"},
        {"sep": "|"},
    ]
    last_err = None
    for kwargs in attempts:
        try:
            _reset(uploaded_file)
            df = pd.read_csv(uploaded_file, **kwargs)
            if df is not None and df.shape[1] >= 1:
                return df
        except Exception as e:
            last_err = e
    raise last_err or ValueError("File tidak dapat dibaca sebagai CSV/TXT tabular.")


def load_file(uploaded_file):
    """Returns (df, text_content, error_message). Tries hard not to crash for unknown tabular data."""
    name = getattr(uploaded_file, "name", "uploaded_file")
    ext = name.split(".")[-1].lower() if "." in name else "csv"
    df = None
    text_content = None
    error = None
    try:
        if ext in ("csv", "txt", "tsv"):
            try:
                df = _try_read_csv(uploaded_file)
            except Exception:
                _reset(uploaded_file)
                raw = uploaded_file.read()
                if isinstance(raw, bytes):
                    text_content = raw.decode("utf-8", errors="replace")
                else:
                    text_content = str(raw)
                # convert plain text into a simple dataframe so dashboard can still run
                lines = [line for line in text_content.splitlines() if line.strip()]
                df = pd.DataFrame({"text": lines}) if lines else pd.DataFrame({"text": [text_content]})
        elif ext in ("xlsx", "xls"):
            _reset(uploaded_file)
            df = pd.read_excel(uploaded_file)
        elif ext == "json":
            _reset(uploaded_file)
            try:
                df = pd.read_json(uploaded_file)
            except Exception:
                _reset(uploaded_file)
                raw = json.load(uploaded_file)
                df = pd.json_normalize(raw)
        else:
            # Last-resort read as delimited text
            try:
                df = _try_read_csv(uploaded_file)
            except Exception as e:
                error = f"Format .{ext} belum didukung penuh dan gagal dibaca: {e}"
        if df is not None:
            df.columns = [str(c).strip() or f"column_{i+1}" for i, c in enumerate(df.columns)]
    except Exception as e:
        error = str(e)
    return df, text_content, error
