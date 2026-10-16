"""
PrepPilot - core engine.

Every cleaning or transformation step is written as a few lines of pandas code.
The engine runs that code to change the data and keeps it, so the whole pipeline
can be exported as a Python script that reproduces the result exactly.

This file has no Streamlit code, so it can be tested and reused on its own.
"""

from __future__ import annotations

import base64
import csv
import html
import io
import json
import re
import textwrap
import warnings
from dataclasses import dataclass
from datetime import datetime
from pprint import pformat

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers shared by the engine and by the exported script
# ---------------------------------------------------------------------------

HELPERS = '''
def clean_text(value):
    """Trim spaces and collapse repeated spaces inside a text value."""
    if isinstance(value, str):
        return re.sub(r"\\s+", " ", value).strip()
    return value


def to_number(series):
    """Turn text like '$1,200' or '15%' into numbers. Bad values become NaN."""
    text = series.astype(str).str.replace(r"[$€£¥,%\\s]", "", regex=True)
    return pd.to_numeric(text, errors="coerce")


def to_date(series, dayfirst=False):
    """Parse dates. ISO dates (2024-12-31) are read as ISO; others use dayfirst."""
    raw = series.astype(str).str.strip()
    iso = raw.str.match(r"^\\d{4}-\\d{1,2}-\\d{1,2}")
    iso_dates = pd.to_datetime(raw.where(iso), errors="coerce", format="ISO8601")
    other_dates = pd.to_datetime(raw.where(~iso), errors="coerce", format="mixed", dayfirst=dayfirst)
    return iso_dates.fillna(other_dates)
'''

_HELPER_NS: dict = {"pd": pd, "np": np, "re": re}
exec(HELPERS, _HELPER_NS)  # noqa: S102 - our own trusted helper code
clean_text = _HELPER_NS["clean_text"]
to_number = _HELPER_NS["to_number"]
to_date = _HELPER_NS["to_date"]

NULL_TOKENS = ["", "-", "?", "NA", "N/A", "n/a", "na", "null", "NULL", "None", "none", "nan", "NaN"]
ID_NAME = re.compile(r"(^|[_\s])(id|code|zip|postal|phone|tel)($|[_\s])", re.IGNORECASE)
TIME_PART = re.compile(r"_(year|quarter|month|day)$", re.IGNORECASE)


class NothingToDo(ValueError):
    """Raised when a step would not change anything (for a friendly message)."""


@dataclass
class Step:
    title: str
    description: str
    code: str
    rows_before: int = 0
    rows_after: int = 0
    cols_before: int = 0
    cols_after: int = 0
    missing_before: int = 0
    missing_after: int = 0

    @property
    def changes(self) -> str:
        parts = []
        if self.rows_before != self.rows_after:
            parts.append(f"rows {self.rows_before:,} → {self.rows_after:,}")
        if self.cols_before != self.cols_after:
            parts.append(f"columns {self.cols_before} → {self.cols_after}")
        if self.missing_before != self.missing_after:
            parts.append(f"missing cells {self.missing_before:,} → {self.missing_after:,}")
        return ", ".join(parts) if parts else "values updated"


# ---------------------------------------------------------------------------
# Column type helpers
# ---------------------------------------------------------------------------

def is_category(s: pd.Series) -> bool:
    return isinstance(s.dtype, pd.CategoricalDtype)


def is_text(s: pd.Series) -> bool:
    if is_category(s) or pd.api.types.is_bool_dtype(s):
        return False
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def is_number(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def is_date(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)


def is_id_like(col) -> bool:
    return bool(ID_NAME.search(str(col)))


def text_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if is_text(df[c])]


def numeric_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if is_number(df[c])]


def date_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if is_date(df[c])]


def categorical_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if is_text(df[c]) or is_category(df[c]) or pd.api.types.is_bool_dtype(df[c])]


def analysis_columns(df: pd.DataFrame) -> list:
    """Number columns worth analysing: not IDs, and not date parts such as order_date_year."""
    return [c for c in numeric_columns(df) if not is_id_like(c) and not TIME_PART.search(str(c))]


def friendly_type(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "Yes/No"
    if is_date(s):
        return "Date"
    if is_category(s):
        return "Category"
    if pd.api.types.is_integer_dtype(s):
        return "Whole number"
    if is_number(s):
        return "Number"
    return "Text"


def _n(count: int, word: str, plural: str | None = None) -> str:
    """'1 value', '3 values'."""
    return f"{count:,} {word if count == 1 else (plural or word + 's')}"


def _style(text: str) -> str:
    if text.isupper():
        return "upper"
    if text.islower():
        return "lower"
    return "title" if text == text.title() else "mixed"


def _names(cols) -> str:
    return ", ".join(map(str, cols))


def _cols(cols) -> str:
    return pformat(list(cols), width=80, compact=True)


def _need(cols, what="column"):
    if not cols:
        raise ValueError(f"Choose at least one {what}.")


# ---------------------------------------------------------------------------
# Loading: CSV / TSV / TXT, Excel (any sheet), JSON and JSON Lines
# ---------------------------------------------------------------------------

def file_kind(name: str) -> str:
    lower = name.lower()
    if lower.endswith((".xlsx", ".xlsm", ".xls")):
        return "excel"
    if lower.endswith((".json", ".jsonl", ".ndjson")):
        return "json"
    return "csv"


def excel_sheets(data: bytes) -> list[str]:
    with pd.ExcelFile(io.BytesIO(data)) as book:
        return [str(s) for s in book.sheet_names]


def _encoding(data: bytes) -> str:
    chunk = data[:1_000_000]
    try:
        chunk.decode("utf-8")
    except UnicodeDecodeError as exc:
        if exc.start < len(chunk) - 3:  # a real error, not a character cut off at the end
            return "latin-1"
    return "utf-8-sig"


def _json_layout(text: str) -> dict:
    """Find where the table is inside a JSON file."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        first = next((line for line in text.splitlines() if line.strip()), "")
        json.loads(first)  # raises a clear error if this is not JSON Lines either
        return {"lines": True}
    if isinstance(obj, list):
        if obj and all(isinstance(item, dict) for item in obj[:50]):
            return {}
        raise ValueError("This JSON file is a list, but not a list of records.")
    if isinstance(obj, dict):
        tables = {k: v for k, v in obj.items()
                  if isinstance(v, list) and v and all(isinstance(i, dict) for i in v[:50])}
        if tables:
            return {"record_path": max(tables, key=lambda k: len(tables[k]))}
        if obj and all(isinstance(v, (list, dict)) for v in obj.values()):
            return {"orient": "columns"}
        return {"orient": "record"}
    raise ValueError("This JSON file does not contain a table.")


def read_options(name: str, data: bytes, sheet: str | None = None) -> dict:
    """Work out how to read a file: format, sheet, encoding, separator and number style."""
    kind = file_kind(name)
    if kind == "excel":
        return {"format": "excel", "sheet_name": sheet if sheet is not None else excel_sheets(data)[0]}
    encoding = _encoding(data)
    if kind == "json":
        return {"format": "json", "encoding": encoding, **_json_layout(data.decode(encoding))}
    sample = data[:65_536].decode(encoding, errors="ignore")
    if len(data) > 65_536 and "\n" in sample:
        sample = sample[: sample.rfind("\n")]  # do not sniff a half line
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = "\t" if name.lower().endswith(".tsv") else ","
    options = {"format": "csv", "sep": sep, "encoding": encoding}
    if sep == ";":  # European CSV saved by Excel: 1.234,56
        options.update(decimal=",", thousands=".")
    return options


def describe_format(options: dict) -> str:
    fmt = options.get("format")
    if fmt == "excel":
        return f"Excel · sheet '{options.get('sheet_name')}'"
    if fmt == "json":
        if options.get("lines"):
            return "JSON Lines"
        if options.get("record_path"):
            return f"JSON · records under '{options['record_path']}'"
        return "JSON"
    sep = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}.get(options.get("sep"), "comma")
    extra = " · European numbers (1.234,56)" if options.get("decimal") == "," else ""
    return f"CSV · {sep}-separated{extra}"


def _flatten_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Nested lists and objects become JSON text, so every cell can be compared and counted."""
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, (list, dict))).any():
            df[col] = df[col].map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
    return df


def _read_json(text: str, options: dict) -> pd.DataFrame:
    if options.get("lines"):
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        obj = json.loads(text)
        if options.get("record_path"):
            records = obj[options["record_path"]]
        elif options.get("orient") == "columns":
            return _flatten_cells(pd.DataFrame(obj))
        elif options.get("orient") == "record":
            records = [obj]
        else:
            records = obj
    return _flatten_cells(pd.json_normalize(records))


def load_file(name: str, data: bytes, options: dict | None = None) -> pd.DataFrame:
    """Read a CSV, TSV, Excel or JSON file from raw bytes."""
    options = read_options(name, data) if options is None else options
    fmt = options.get("format", file_kind(name))
    try:
        if fmt == "excel":
            return pd.read_excel(io.BytesIO(data), sheet_name=options.get("sheet_name", 0))
        if fmt == "json":
            return _read_json(data.decode(options.get("encoding", "utf-8-sig")), options)
        kwargs = {k: v for k, v in options.items() if k != "format"}
        return pd.read_csv(io.BytesIO(data), **kwargs)
    except (UnicodeDecodeError, pd.errors.ParserError, json.JSONDecodeError, KeyError, ValueError) as exc:
        raise ValueError(f"Could not read the file: {exc}") from None


_JSON_FLATTEN = [
    "for col in df.columns:  # store nested lists and objects as JSON text",
    "    if df[col].map(lambda v: isinstance(v, (list, dict))).any():",
    "        df[col] = df[col].map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)",
]


def _reader_code(name: str, options: dict) -> list[str]:
    """The lines of the exported script that read the original file."""
    fmt = options.get("format", file_kind(name))
    if fmt == "excel":
        return [f"df = pd.read_excel({name!r}, sheet_name={options.get('sheet_name', 0)!r})"]
    if fmt == "json":
        encoding = options.get("encoding", "utf-8-sig")
        if options.get("lines"):
            lines = [f"with open({name!r}, encoding={encoding!r}) as f:",
                     "    df = pd.json_normalize([json.loads(line) for line in f if line.strip()])"]
        else:
            lines = [f"with open({name!r}, encoding={encoding!r}) as f:", "    data = json.load(f)"]
            if options.get("record_path"):
                lines.append(f"df = pd.json_normalize(data[{options['record_path']!r}])")
            elif options.get("orient") == "columns":
                lines.append("df = pd.DataFrame(data)")
            elif options.get("orient") == "record":
                lines.append("df = pd.json_normalize([data])")
            else:
                lines.append("df = pd.json_normalize(data)")
        return lines + _JSON_FLATTEN
    kwargs = "".join(f", {k}={v!r}" for k, v in options.items() if k != "format")
    return [f"df = pd.read_csv({name!r}{kwargs})"]


# ---------------------------------------------------------------------------
# Running steps
# ---------------------------------------------------------------------------

def run_code(df: pd.DataFrame, code: str) -> pd.DataFrame:
    namespace = dict(_HELPER_NS)
    namespace["df"] = df.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(code, namespace)  # noqa: S102 - code is generated by the planners below
    result = namespace["df"]
    if not isinstance(result, pd.DataFrame):
        raise ValueError("The step did not produce a table.")
    return result


def apply_step(df: pd.DataFrame, plan: tuple[str, str, str]) -> tuple[pd.DataFrame, Step]:
    title, description, code = plan
    new_df = run_code(df, code)
    step = Step(
        title=title,
        description=description,
        code=code,
        rows_before=len(df),
        rows_after=len(new_df),
        cols_before=df.shape[1],
        cols_after=new_df.shape[1],
        missing_before=int(df.isna().sum().sum()),
        missing_after=int(new_df.isna().sum().sum()),
    )
    return new_df, step


def run_plans(df: pd.DataFrame, planners) -> list[tuple[pd.DataFrame, Step]]:
    """Run planners one after another. Each one looks at the data left by the previous one."""
    results, current = [], df
    for planner in planners:
        try:
            plan = planner(current)
        except NothingToDo:
            continue
        current, step = apply_step(current, plan)
        results.append((current, step))
    return results


# ---------------------------------------------------------------------------
# Detecting problems
# ---------------------------------------------------------------------------

def column_name_changes(df: pd.DataFrame) -> dict:
    """Map each messy column name to a tidy, unique lowercase_with_underscores name."""
    taken, mapping = set(), {}
    for i, col in enumerate(df.columns):
        new = re.sub(r"\W+", "_", str(col).strip().lower()).strip("_") or f"column_{i + 1}"
        base, n = new, 2
        while new in taken:
            new, n = f"{base}_{n}", n + 1
        taken.add(new)
        if new != col:
            mapping[col] = new
    return mapping


def _strings(s: pd.Series) -> pd.Series:
    values = s.dropna()
    return values[values.map(lambda v: isinstance(v, str))].astype(str)


def messy_text_counts(df: pd.DataFrame) -> tuple[int, int]:
    """(values with extra spaces, placeholder values like 'N/A' that should be missing)."""
    tokens = set(NULL_TOKENS)
    spaces = placeholders = 0
    for col in text_columns(df):
        values = _strings(df[col])
        if values.empty:
            continue
        is_token = values.str.strip().isin(tokens)
        placeholders += int(is_token.sum())
        spaces += int((values.map(clean_text) != values)[~is_token].sum())
    return spaces, placeholders


def detect_types(df: pd.DataFrame) -> dict:
    """Find text columns that are really numbers or dates."""
    found: dict = {}
    for col in text_columns(df):
        values = df[col].dropna()
        values = values[~values.astype(str).str.strip().isin(set(NULL_TOKENS))]
        if len(values) == 0 or is_id_like(col):
            continue
        strings = values.astype(str)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if to_number(strings).notna().mean() >= 0.9:
                found[col] = "number"
                continue
            has_digits = strings.str.contains(r"\d").mean() >= 0.9
            looks_like_date = strings.str.contains(r"\d{1,4}[-/.]\d{1,2}|[A-Za-z]{3,}\s+\d{1,2}").mean() >= 0.9
            if has_digits and looks_like_date and to_date(strings).notna().mean() >= 0.9:
                found[col] = "date"
    return found


def guess_dayfirst(series: pd.Series) -> bool:
    """Look at dates like 05/03/2024: if the first number is ever above 12, it is day-first."""
    parts = series.dropna().astype(str).str.extract(r"^\s*(\d{1,2})[/.-](\d{1,2})[/.-]\d{2,4}")
    parts = parts.dropna().astype(int)
    if parts.empty:
        return False
    return bool((parts[0] > 12).any() and not (parts[1] > 12).any())


def spelling_variants(df: pd.DataFrame) -> dict:
    """Find values written in different ways, e.g. 'north', 'NORTH ' and 'North'.

    Returns {column: {variant: most common spelling}}.
    """
    found: dict = {}
    for col in text_columns(df):
        if is_id_like(col):
            continue
        values = _strings(df[col])
        if values.empty:
            continue
        norm = values.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
        if norm.nunique() == values.nunique():
            continue
        styles = values.map(clean_text).map(_style).value_counts()
        usual = styles.index[0] if len(styles) else "title"
        counts = pd.DataFrame({"norm": norm, "value": values}).value_counts()
        distinct = counts.groupby(level="norm").size()
        counts = counts[counts.index.get_level_values("norm").isin(distinct[distinct > 1].index)]
        mapping = {}
        for _, group in counts.groupby(level="norm"):
            variants = group.droplevel("norm")
            cleaned: dict = {}
            for value, n in variants.items():
                cleaned[clean_text(value)] = cleaned.get(clean_text(value), 0) + int(n)
            # use the column's usual style (e.g. Title Case), then the most common spelling
            best = max(cleaned, key=lambda c: (_style(c) == usual, cleaned[c], _style(c) == "title"))
            mapping.update({value: best for value in variants.index if value != best})
        if mapping:
            found[col] = mapping
    return found


def constant_columns(df: pd.DataFrame) -> list:
    if len(df) < 2:
        return []
    return [c for c in df.columns if df[c].notna().all() and df[c].nunique() == 1]


def sparse_columns(df: pd.DataFrame, threshold_pct: float) -> list:
    pct = df.isna().mean() * 100
    return [c for c in df.columns if pct[c] > threshold_pct]


OUTLIER_RULES = {
    "Extreme only (3 × IQR)": ("IQR", 3.0),
    "Standard (1.5 × IQR)": ("IQR", 1.5),
    "Z-score (3 standard deviations)": ("Z-score", 3.0),
}
OUTLIER_RULE_TEXT = {
    "Extreme only (3 × IQR)": "beyond 3 × IQR",
    "Standard (1.5 × IQR)": "outside 1.5 × IQR",
    "Z-score (3 standard deviations)": "more than 3 standard deviations from the mean",
}


def outlier_counts(df: pd.DataFrame, method: str = "IQR", k: float = 1.5) -> dict:
    counts = {}
    for col in analysis_columns(df):
        s = df[col].dropna()
        if len(s) < 20 or s.nunique() < 5:  # too little data for an outlier rule to mean much
            continue
        if method == "IQR":
            q1, q3 = s.quantile([0.25, 0.75])
            spread = q3 - q1
            low, high = q1 - k * spread, q3 + k * spread
        else:
            spread = s.std()
            low, high = s.mean() - k * spread, s.mean() + k * spread
        if not spread > 0:
            continue
        n = int(((s < low) | (s > high)).sum())
        if n:
            counts[col] = n
    return counts


# ---------------------------------------------------------------------------
# Cleaning planners: each returns (title, description, code)
# ---------------------------------------------------------------------------

def plan_standardize_columns(df: pd.DataFrame):
    mapping = column_name_changes(df)
    if not mapping:
        raise NothingToDo("Column names are already tidy.")
    old, new = next(iter(mapping.items()))
    code = f"df = df.rename(columns={pformat(mapping, width=80, sort_dicts=False)})"
    desc = (f"Renamed {len(mapping)} columns to lowercase_with_underscores "
            f"(e.g. '{str(old).strip()}' → {new}).")
    return "Tidy column names", desc, code


def plan_trim_text(df: pd.DataFrame):
    cols = text_columns(df)
    spaces, placeholders = messy_text_counts(df)
    if not cols or (spaces == 0 and placeholders == 0):
        raise NothingToDo("Text values are already tidy.")
    code = (
        f"null_tokens = {_cols(NULL_TOKENS)}\n"
        f"text_cols = {_cols(cols)}\n"
        "for col in text_cols:\n"
        "    df[col] = df[col].map(clean_text).replace(null_tokens, np.nan)"
    )
    parts = []
    if spaces:
        parts.append(f"trimmed extra spaces in {_n(spaces, 'value')}")
    if placeholders:
        parts.append(f"marked {_n(placeholders, 'placeholder')} like 'N/A' or '-' as missing")
    desc = " and ".join(parts)
    desc = desc[0].upper() + desc[1:] + "."
    return "Trim text and mark placeholders as missing", desc, code


def plan_drop_duplicates(df: pd.DataFrame, subset=None):
    subset = list(subset) if subset else None
    count = int(df.duplicated(subset=subset).sum())
    if count == 0:
        raise NothingToDo("No duplicate rows found.")
    code = f"df = df.drop_duplicates(subset={subset!r}, keep='first').reset_index(drop=True)"
    scope = "all columns" if subset is None else _names(subset)
    return "Remove duplicate rows", f"Removed {_n(count, 'duplicate row')} (compared on {scope}).", code


def plan_convert_types(df: pd.DataFrame, conversions: dict, dayfirst: bool | None = None):
    """conversions maps column -> 'number' | 'date' | 'text' | 'category'."""
    if not conversions:
        raise NothingToDo("All columns already have a sensible type.")
    lines, names = [], []
    for col, kind in conversions.items():
        if kind == "number":
            lines.append(f"df[{col!r}] = to_number(df[{col!r}])")
        elif kind == "date":
            first = guess_dayfirst(df[col]) if dayfirst is None else dayfirst
            lines.append(f"df[{col!r}] = to_date(df[{col!r}], dayfirst={first})")
        elif kind == "text":
            lines.append(f"df[{col!r}] = df[{col!r}].astype(str).where(df[{col!r}].notna())")
        elif kind == "category":
            lines.append(f"df[{col!r}] = df[{col!r}].astype('category')")
        else:
            raise ValueError(f"Unknown type: {kind}")
        names.append(f"{col} → {kind}")
    return "Fix column types", "Converted " + "; ".join(names) + ".", "\n".join(lines)


def plan_text_case(df: pd.DataFrame, cols, case: str):
    _need(cols)
    method = {"Title Case": "title", "lowercase": "lower", "UPPERCASE": "upper"}[case]
    code = (f"for col in {_cols(cols)}:\n"
            f"    df[col] = df[col].map(lambda v: v.{method}() if isinstance(v, str) else v)")
    return "Change capital letters", f"Changed {_names(cols)} to {case}.", code


def plan_unify_variants(df: pd.DataFrame, cols=None):
    variants = spelling_variants(df)
    if cols is not None:
        variants = {c: m for c, m in variants.items() if c in cols}
    if not variants:
        raise NothingToDo("No values written in different ways.")
    lines = [f"df[{col!r}] = df[{col!r}].replace({pformat(m, width=80, sort_dicts=False)})"
             for col, m in variants.items()]
    total = sum(len(m) for m in variants.values())
    variant, best = next(iter(next(iter(variants.values())).items()))
    desc = (f"Merged {_n(total, 'spelling variant')} in {_names(variants)} into one spelling "
            f"(e.g. '{variant}' → '{best}').")
    return "Unify spelling variants", desc, "\n".join(lines)


def plan_drop_constant(df: pd.DataFrame):
    cols = constant_columns(df)
    if not cols:
        raise NothingToDo("No column has the same value in every row.")
    return ("Drop single-value columns", f"Removed {_names(cols)}, which had the same value in every row.",
            f"df = df.drop(columns={_cols(cols)})")


def plan_drop_sparse(df: pd.DataFrame, threshold_pct: float = 60.0):
    cols = sparse_columns(df, threshold_pct)
    if not cols:
        raise NothingToDo(f"No column is more than {threshold_pct:g}% empty.")
    pct = df[cols].isna().mean() * 100
    desc = "Removed " + ", ".join(f"{c} ({pct[c]:.0f}% empty)" for c in cols) + "."
    return "Drop mostly empty columns", desc, f"df = df.drop(columns={_cols(cols)})"


def plan_drop_columns(df: pd.DataFrame, cols):
    _need(cols)
    return "Drop columns", f"Removed {_names(cols)}.", f"df = df.drop(columns={_cols(cols)})"


def plan_missing(df: pd.DataFrame, cols, strategy: str, value=None):
    """Handle missing values in chosen columns."""
    _need(cols)
    cols = list(cols)
    missing = int(df[cols].isna().sum().sum())
    if missing == 0:
        raise NothingToDo("Those columns have no missing values.")
    non_numeric = [c for c in cols if not is_number(df[c])]
    if strategy in ("mean", "median") and non_numeric:
        raise ValueError(f"The {strategy} only works for number columns. Remove: {_names(non_numeric)}.")
    if strategy == "drop rows":
        code = f"df = df.dropna(subset={_cols(cols)}).reset_index(drop=True)"
        desc = f"Removed rows with missing values in {_names(cols)}."
    elif strategy in ("mean", "median"):
        code = f"cols = {_cols(cols)}\ndf[cols] = df[cols].fillna(df[cols].{strategy}())"
        desc = f"Filled {missing:,} missing values with the column {strategy}."
    elif strategy == "most common value":
        code = (f"for col in {_cols(cols)}:\n"
                "    mode = df[col].mode()\n"
                "    if not mode.empty:\n"
                "        df[col] = df[col].fillna(mode.iloc[0])")
        desc = f"Filled {missing:,} missing values with the most common value."
    elif strategy == "fixed value":
        if value is None or str(value) == "":
            raise ValueError("Type the value to fill in.")
        lines = []
        for c in cols:
            v = value
            if is_number(df[c]):
                try:
                    v = float(value)
                except ValueError:
                    raise ValueError(f"'{value}' is not a number, but {c} is a number column.") from None
            lines.append(f"df[{c!r}] = df[{c!r}].fillna({v!r})")
        code = "\n".join(lines)
        desc = f"Filled {missing:,} missing values with {value!r}."
    elif strategy == "previous value":
        code = f"cols = {_cols(cols)}\ndf[cols] = df[cols].ffill()"
        desc = f"Filled {missing:,} missing values with the value from the row above."
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return "Handle missing values", desc, code


MISSING_CHOICES = {
    "numbers": ["median", "mean", "zero", "drop rows", "leave as is"],
    "text": ["Unknown", "most common value", "drop rows", "leave as is"],
    "dates": ["leave as is", "previous value", "drop rows"],
}


def missing_by_type(df: pd.DataFrame) -> dict:
    """{'numbers': [...], 'text': [...], 'dates': [...]} columns that have missing values."""
    has = [c for c in df.columns if df[c].isna().any()]
    return {
        "numbers": [c for c in has if is_number(df[c])],
        "text": [c for c in has if is_text(df[c]) or is_category(df[c])],
        "dates": [c for c in has if is_date(df[c])],
    }


def plan_missing_by_type(df: pd.DataFrame, kind: str, strategy: str):
    """Handle missing values in every column of one kind (numbers, text or dates)."""
    if strategy == "leave as is":
        raise NothingToDo("Missing values are left as they are.")
    cols = missing_by_type(df)[kind]
    if not cols:
        raise NothingToDo(f"No missing {kind}.")
    count = int(df[cols].isna().sum().sum())
    names = _names(cols)
    if strategy == "drop rows":
        return (f"Remove rows with missing {kind}", f"Removed rows with missing values in {names}.",
                f"df = df.dropna(subset={_cols(cols)}).reset_index(drop=True)")
    if strategy in ("median", "mean", "zero"):
        fill = "0" if strategy == "zero" else f"df[cols].{strategy}()"
        code = f"cols = {_cols(cols)}\ndf[cols] = df[cols].fillna({fill})"
        how = "0" if strategy == "zero" else f"each column's {strategy}"
    elif strategy == "Unknown":
        lines = []
        plain = [c for c in cols if not is_category(df[c])]
        if plain:
            lines.append(f"cols = {_cols(plain)}\ndf[cols] = df[cols].fillna('Unknown')")
        for c in cols:
            if is_category(df[c]):
                lines.append(f"if 'Unknown' not in df[{c!r}].cat.categories:\n"
                             f"    df[{c!r}] = df[{c!r}].cat.add_categories(['Unknown'])\n"
                             f"df[{c!r}] = df[{c!r}].fillna('Unknown')")
        code = "\n".join(lines)
        how = "'Unknown'"
    elif strategy == "most common value":
        code = (f"for col in {_cols(cols)}:\n"
                "    mode = df[col].mode()\n"
                "    if not mode.empty:\n"
                "        df[col] = df[col].fillna(mode.iloc[0])")
        how = "the most common value"
    elif strategy == "previous value":
        code = f"cols = {_cols(cols)}\ndf[cols] = df[cols].ffill()"
        how = "the value from the row above"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return f"Fill missing {kind}", f"Filled {_n(count, 'missing value')} in {names} with {how}.", code


def plan_outliers(df: pd.DataFrame, cols, method: str = "IQR", action: str = "cap", k: float | None = None):
    _need(cols)
    bad = [c for c in cols if not is_number(df[c])]
    if bad:
        raise ValueError(f"Outliers only apply to number columns. Remove: {_names(bad)}.")
    k = (1.5 if method == "IQR" else 3.0) if k is None else k
    if method == "IQR":
        bounds = ("    q1, q3 = df[col].quantile([0.25, 0.75])\n"
                  "    spread = q3 - q1\n"
                  f"    low, high = q1 - {k:g} * spread, q3 + {k:g} * spread\n")
        rule = f"outside {k:g} × IQR"
    else:
        bounds = ("    spread = df[col].std()\n"
                  f"    low, high = df[col].mean() - {k:g} * spread, df[col].mean() + {k:g} * spread\n")
        rule = f"more than {k:g} standard deviations from the mean"
    guard = ("    if not spread > 0:  # (almost) all values are the same, so the rule does not apply\n"
             "        continue\n")
    if action == "remove":
        body = "    df = df[df[col].isna() | df[col].between(low, high)]\n"
        tail = "df = df.reset_index(drop=True)"
        verb = "Removed rows with values"
    else:
        body = ("    if pd.api.types.is_integer_dtype(df[col]):  # keep whole numbers whole\n"
                "        low, high = int(np.floor(low)), int(np.ceil(high))\n"
                "    df[col] = df[col].clip(low, high)\n")
        tail = ""
        verb = "Capped values"
    code = f"for col in {_cols(cols)}:\n{bounds}{guard}{body}{tail}".rstrip()
    return "Handle outliers", f"{verb} {rule} in {_names(cols)}.", code


def plan_outliers_auto(df: pd.DataFrame, rule: str = "Extreme only (3 × IQR)", action: str = "cap"):
    method, k = OUTLIER_RULES[rule]
    counts = outlier_counts(df, method, k)
    if not counts:
        raise NothingToDo("No outliers found with this rule.")
    title, _, code = plan_outliers(df, list(counts), method, action, k)
    detail = ", ".join(f"{c} ({n})" for c, n in counts.items())
    verb = "Capped" if action == "cap" else "Removed rows with"
    total = _n(sum(counts.values()), "value")
    return title, f"{verb} {total} {OUTLIER_RULE_TEXT[rule]} in {detail}.", code


@dataclass
class CleanOptions:
    """Which cleaning steps to run, and how. The defaults are what 'Do everything' uses."""
    column_names: bool = True
    trim_text: bool = True
    drop_empty: bool = True
    empty_threshold: float = 60.0
    drop_constant: bool = True
    fix_types: bool = True
    dayfirst: bool | None = None
    unify_spelling: bool = True
    duplicates: bool = True
    outliers: bool = True
    outlier_rule: str = "Extreme only (3 × IQR)"
    outlier_action: str = "cap"
    missing: bool = True
    missing_numbers: str = "median"
    missing_text: str = "Unknown"
    missing_dates: str = "leave as is"


def run_cleaning(df: pd.DataFrame, options: CleanOptions | None = None) -> list[tuple[pd.DataFrame, Step]]:
    """Run the chosen cleaning steps in a sensible order."""
    o = options or CleanOptions()
    planners = []
    if o.column_names:
        planners.append(plan_standardize_columns)
    if o.trim_text:
        planners.append(plan_trim_text)
    if o.drop_empty:
        planners.append(lambda d: plan_drop_sparse(d, o.empty_threshold))
    if o.drop_constant:
        planners.append(plan_drop_constant)
    if o.fix_types:
        planners.append(lambda d: plan_convert_types(d, detect_types(d), dayfirst=o.dayfirst))
    if o.unify_spelling:
        planners.append(plan_unify_variants)
    if o.duplicates:  # after trimming and spelling fixes, so near-duplicates are caught too
        planners.append(plan_drop_duplicates)
    if o.outliers:  # before filling gaps, so the fill values do not hide real outliers
        planners.append(lambda d: plan_outliers_auto(d, o.outlier_rule, o.outlier_action))
    if o.missing:
        planners.append(lambda d: plan_missing_by_type(d, "numbers", o.missing_numbers))
        planners.append(lambda d: plan_missing_by_type(d, "text", o.missing_text))
        planners.append(lambda d: plan_missing_by_type(d, "dates", o.missing_dates))
    return run_plans(df, planners)


# ---------------------------------------------------------------------------
# Transformation planners
# ---------------------------------------------------------------------------

FILTER_OPS = ["is equal to", "is not equal to", "is greater than", "is at least",
              "is less than", "is at most", "is one of", "contains", "is missing", "is not missing"]


def _literal(series: pd.Series, value):
    if is_number(series):
        try:
            return repr(float(value))
        except (TypeError, ValueError):
            raise ValueError(f"'{value}' is not a number.") from None
    if is_date(series):
        try:
            return f"pd.Timestamp({str(pd.Timestamp(value))!r})"
        except (TypeError, ValueError):
            raise ValueError(f"'{value}' is not a date. Try 2024-12-31.") from None
    return repr(str(value))


def plan_filter(df: pd.DataFrame, col, op: str, value=None):
    c = repr(col)
    if op == "is missing":
        cond = f"df[{c}].isna()"
    elif op == "is not missing":
        cond = f"df[{c}].notna()"
    elif op == "is one of":
        if not value:
            raise ValueError("Choose at least one value.")
        cond = f"df[{c}].isin({list(value)!r})"
    elif op == "contains":
        if value in (None, ""):
            raise ValueError("Type the text to look for.")
        cond = f"df[{c}].astype(str).str.contains({str(value)!r}, case=False, regex=False, na=False)"
    else:
        if value in (None, ""):
            raise ValueError("Type a value to compare with.")
        symbol = {"is equal to": "==", "is not equal to": "!=", "is greater than": ">",
                  "is at least": ">=", "is less than": "<", "is at most": "<="}[op]
        cond = f"df[{c}] {symbol} {_literal(df[col], value)}"
    shown = _names(value) if isinstance(value, (list, tuple)) else value
    desc = f"Kept rows where {col} {op}" + (f" {shown}" if shown not in (None, "") else "") + "."
    return "Filter rows", desc, f"df = df[{cond}].reset_index(drop=True)"


def plan_sort(df: pd.DataFrame, cols, ascending: bool = True):
    _need(cols)
    order = "smallest first" if ascending else "largest first"
    return ("Sort rows", f"Sorted by {_names(cols)} ({order}).",
            f"df = df.sort_values({_cols(cols)}, ascending={ascending}).reset_index(drop=True)")


def plan_new_column(df: pd.DataFrame, name: str, expression: str):
    name, expression = name.strip(), expression.strip()
    if not name or not expression:
        raise ValueError("Give the new column a name and a formula.")
    code = f"df[{name!r}] = df.eval({expression!r})"
    try:
        run_code(df.head(50), code)
    except Exception as exc:  # show the formula problem in plain words
        raise ValueError(f"The formula did not work: {exc}") from None
    return "Add calculated column", f"Created {name} = {expression}.", code


DATE_PARTS = {
    "year": "dt.year", "quarter": "dt.quarter", "month": "dt.month",
    "month name": "dt.month_name()", "weekday": "dt.day_name()", "day": "dt.day",
}


def plan_date_parts(df: pd.DataFrame, col, parts):
    _need(parts, "part")
    if not is_date(df[col]):
        raise ValueError(f"{col} is not a date column yet. Convert it in the Clean step first.")
    lines = [f"df[{(str(col) + '_' + p.replace(' ', '_'))!r}] = df[{col!r}].{DATE_PARTS[p]}" for p in parts]
    return "Extract date parts", f"Added {', '.join(parts)} from {col}.", "\n".join(lines)


AGG_FUNCS = ["sum", "mean", "median", "min", "max", "count", "nunique"]


def plan_group(df: pd.DataFrame, by, aggregations: list[tuple[str, str]]):
    _need(by, "column to group by")
    if not aggregations:
        raise ValueError("Add at least one calculation (for example: sales → sum).")
    for col, func in aggregations:
        if func in ("sum", "mean", "median") and not is_number(df[col]):
            raise ValueError(f"'{func}' needs a number column, but {col} is not a number.")
    named = ",\n".join(f"    {(str(col) + '_' + func)!r}: ({col!r}, {func!r})" for col, func in aggregations)
    code = f"df = df.groupby({_cols(by)}, as_index=False, observed=True).agg(**{{\n{named}\n}})"
    calcs = ", ".join(f"{func} of {col}" for col, func in aggregations)
    return "Group and summarize", f"Grouped by {_names(by)} and calculated {calcs}.", code


def plan_rename(df: pd.DataFrame, old, new: str):
    new = new.strip()
    if not new:
        raise ValueError("Type the new name.")
    if new in df.columns:
        raise ValueError(f"A column called {new} already exists.")
    return "Rename column", f"Renamed {old} to {new}.", f"df = df.rename(columns={{{old!r}: {new!r}}})"


AUTO_DATE_PARTS = ["year", "month", "weekday"]


def plan_auto_date_parts(df: pd.DataFrame):
    """Add year, month and weekday columns for up to two date columns."""
    cols = [c for c in date_columns(df) if df[c].notna().any()][:2]
    lines = []
    for col in cols:
        for part in AUTO_DATE_PARTS:
            name = f"{col}_{part.replace(' ', '_')}"
            if name not in df.columns:
                lines.append(f"df[{name!r}] = df[{col!r}].{DATE_PARTS[part]}")
    if not lines:
        raise NothingToDo("Date parts are already there.")
    return "Extract date parts", f"Added year, month and weekday columns from {_names(cols)}.", "\n".join(lines)


def plan_auto_sort(df: pd.DataFrame):
    cols = date_columns(df)
    if not cols or df[cols[0]].is_monotonic_increasing:
        raise NothingToDo("Already in date order.")
    title, _, code = plan_sort(df, [cols[0]], True)
    return title, f"Sorted rows by {cols[0]}, oldest first.", code


_QTY_NAME = re.compile(r"(^|_)(qty|quantity|quantities|units|unit_count|pieces)($|_)", re.IGNORECASE)
_PRICE_NAME = re.compile(r"(^|_)(unit_)?price($|_)", re.IGNORECASE)
_HAS_AMOUNT = re.compile(r"revenue|sales|amount|turnover|total", re.IGNORECASE)


def plan_auto_revenue(df: pd.DataFrame):
    """Add revenue = quantity × price (minus a discount %) when the data has no money total yet."""
    cols = analysis_columns(df)
    if "revenue" in df.columns or any(_HAS_AMOUNT.search(str(c)) for c in cols):
        raise NothingToDo("The data already has a revenue or amount column.")
    qty = next((c for c in cols if _QTY_NAME.search(str(c))), None)
    price = next((c for c in cols if _PRICE_NAME.search(str(c))), None)
    if qty is None or price is None:
        raise NothingToDo("No quantity and price columns to multiply.")
    expr = f"`{qty}` * `{price}`"
    discount = next((c for c in cols if "discount" in str(c).lower()), None)
    if discount is not None:
        d = df[discount].dropna()
        if len(d) and d.between(0, 100).all() and d.max() > 1:
            expr += f" * (1 - `{discount}` / 100)"
        elif len(d) and d.between(0, 1).all():
            expr += f" * (1 - `{discount}`)"
    code = f"df['revenue'] = df.eval({expr!r})"
    return ("Add revenue column", f"Created revenue = {expr.replace('`', '')}, so there is an amount to analyse.", code)


def run_auto_transform(df: pd.DataFrame) -> list[tuple[pd.DataFrame, Step]]:
    return run_plans(df, [plan_auto_revenue, plan_auto_date_parts, plan_auto_sort])


def run_everything(df: pd.DataFrame) -> list[tuple[pd.DataFrame, Step]]:
    """'Do everything': all recommended cleaning, then useful automatic transformations."""
    results = run_cleaning(df, CleanOptions())
    last = results[-1][0] if results else df
    return results + run_auto_transform(last)


# ---------------------------------------------------------------------------
# Exploratory data analysis
# ---------------------------------------------------------------------------

def overview(df: pd.DataFrame) -> dict:
    cells = df.shape[0] * df.shape[1]
    missing = int(df.isna().sum().sum())
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing": missing,
        "missing_pct": (missing / cells * 100) if cells else 0.0,
        "duplicates": int(df.duplicated().sum()) if len(df) else 0,
        "numeric": len(numeric_columns(df)),
        "text": len(categorical_columns(df)),
        "dates": len(date_columns(df)),
    }


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (float, np.floating)):
        return f"{v:,.0f}" if float(v).is_integer() and abs(v) >= 1000 else f"{v:,.2f}"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return str(v)


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        row = {
            "Column": str(col),
            "Type": friendly_type(s),
            "Filled": int(s.notna().sum()),
            "Missing": int(s.isna().sum()),
            "Missing %": round(float(s.isna().mean() * 100), 1) if len(s) else 0.0,
            "Unique": int(s.nunique(dropna=True)),
            "Most common / mean": "",
            "Min": "",
            "Max": "",
        }
        if is_number(s) and len(non_null):
            if "year" in str(col).lower():  # show 2025, not 2,025
                row["Min"], row["Max"] = str(int(non_null.min())), str(int(non_null.max()))
            else:
                row["Most common / mean"] = f"mean {_fmt(float(non_null.mean()))}"
                row["Min"], row["Max"] = _fmt(non_null.min()), _fmt(non_null.max())
        elif is_date(s) and len(non_null):
            row["Min"], row["Max"] = _fmt(non_null.min()), _fmt(non_null.max())
        elif len(non_null):
            top = non_null.astype(str).value_counts()
            row["Most common / mean"] = f"{top.index[0]} ({top.iloc[0]:,})"
        rows.append(row)
    return pd.DataFrame(rows)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = numeric_columns(df)
    if not cols or df.empty:
        return pd.DataFrame()
    desc = df[cols].describe().T
    desc["skew"] = df[cols].skew()
    desc = desc.rename(columns={"50%": "median", "25%": "Q1", "75%": "Q3"})
    desc = desc[["count", "mean", "std", "min", "Q1", "median", "Q3", "max", "skew"]].round(2)
    desc["count"] = desc["count"].astype(int)
    return desc


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in analysis_columns(df) if df[c].nunique() > 1]
    if len(cols) < 2:
        return pd.DataFrame()
    return df[cols].corr(numeric_only=True).round(2)


def findings(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Plain-language findings. Each item is ('issue', text) or ('note', text)."""
    if df.empty:
        return [("issue", "The dataset has no rows.")]
    out: list[tuple[str, str]] = []

    def issue(text):
        out.append(("issue", text))

    def note(text):
        out.append(("note", text))

    pct = (df.isna().mean() * 100).sort_values(ascending=False)
    pct = pct[pct > 0]
    if len(pct):
        top = ", ".join(f"{c} ({p:.0f}%)" for c, p in pct.head(3).items())
        verb = "column has" if len(pct) == 1 else "columns have"
        issue(f"{len(pct)} of {df.shape[1]} {verb} missing values. Most affected: {top}.")
    dups = int(df.duplicated().sum())
    if dups:
        verb = "row is an exact duplicate" if dups == 1 else "rows are exact duplicates"
        issue(f"{dups:,} {verb} ({dups / len(df) * 100:.1f}% of the data).")
    spaces, placeholders = messy_text_counts(df)
    parts = []
    if spaces:
        parts.append(f"{_n(spaces, 'text value')} {'has' if spaces == 1 else 'have'} extra spaces")
    if placeholders:
        parts.append(f"{_n(placeholders, 'value')} {'is a placeholder' if placeholders == 1 else 'are placeholders'} "
                     "like 'N/A' or '-' instead of real missing values")
    if parts:
        issue(" and ".join(parts) + ".")
    for col in constant_columns(df):
        issue(f"{col} has the same value in every row, so it adds no information.")
    detected = detect_types(df)
    for col, kind in detected.items():
        issue(f"{col} is stored as text but looks like a {kind} column.")
    for col, mapping in spelling_variants(df).items():
        sample = ", ".join(f"'{v}' → '{c}'" for v, c in list(mapping.items())[:2])
        issue(f"{col} has the same values written in different ways ({sample}).")
    extreme = outlier_counts(df, "IQR", 3.0)
    for col, n in extreme.items():
        issue(f"{col} has {_n(n, 'extreme value')} (beyond 3 × IQR) that may be errors.")
    for col, n in outlier_counts(df, "IQR", 1.5).items():
        if col not in extreme:
            note(f"{col} has {_n(n, 'unusually high or low value')} (outside 1.5 × IQR); they may be real.")
    for col in text_columns(df):
        values = df[col].dropna()
        if len(values) > 20 and col not in detected and values.nunique() / len(values) > 0.95:
            note(f"{col} is unique in almost every row, so it is probably an ID or free text.")
    for col in analysis_columns(df):
        s = df[col].dropna()
        if len(s) > 10 and s.nunique() > 5:
            skew = float(s.skew())
            if abs(skew) > 1:
                side = "right (a few very large values)" if skew > 0 else "left (a few very small values)"
                note(f"{col} is skewed to the {side}; the median describes it better than the mean.")
    corr = correlations(df)
    if not corr.empty:
        cols = list(corr.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                r = corr.loc[a, b]
                if abs(r) >= 0.7:
                    direction = "rises" if r > 0 else "falls"
                    note(f"Strong link between {a} and {b} (r = {r:.2f}): when one goes up, the other {direction}.")
    return out


def insights(df: pd.DataFrame) -> list[str]:
    return [text for _, text in findings(df)]


def rows_over_time(series: pd.Series) -> tuple[pd.Series, str]:
    """Count rows per day, week or month, depending on how long the dates span."""
    s = series.dropna()
    span = (s.max() - s.min()).days if len(s) else 0
    freq, label = ("D", "day") if span <= 62 else (("W", "week") if span <= 400 else ("M", "month"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        counts = s.dt.to_period(freq).dt.to_timestamp().value_counts().sort_index()
    return counts, label


# ---------------------------------------------------------------------------
# Finance
# ---------------------------------------------------------------------------

_MONEY_STRONG = ("amount", "revenue", "sales", "price", "cost", "profit", "income", "expense", "payment",
                 "salary", "turnover", "budget", "spend", "earning")
_MONEY_WEAK = {"net", "gross", "total", "value", "fee", "fees", "tax", "margin", "balance", "gmv", "paid"}
_AMOUNT_PRIORITY = ("revenue", "sales", "turnover", "amount", "income", "total", "profit", "payment", "value",
                    "spend", "expense", "cost", "price")
_CATEGORY_PRIORITY = ("product", "category", "segment", "region", "channel", "department", "customer",
                      "vendor", "supplier", "account", "type", "country", "city", "store")
CURRENCIES = ["", "$", "€", "£", "¥"]


def is_money_name(col) -> bool:
    name = str(col).lower()
    if re.search(r"_(count|nunique)$", name):
        return False
    tokens = set(re.findall(r"[a-z0-9]+", name))
    return any(word in name for word in _MONEY_STRONG) or bool(tokens & _MONEY_WEAK)


def money_columns(df: pd.DataFrame) -> list:
    return [c for c in analysis_columns(df) if is_money_name(c)]


def best_amount_column(df: pd.DataFrame):
    cols = money_columns(df)
    for word in _AMOUNT_PRIORITY:
        for c in cols:
            if word in str(c).lower():
                return c
    return cols[0] if cols else None


def best_category_column(df: pd.DataFrame):
    cats = [c for c in categorical_columns(df) if 1 < df[c].nunique() <= 50 and not is_id_like(c)]
    for word in _CATEGORY_PRIORITY:
        for c in cats:
            if word in str(c).lower():
                return c
    return min(cats, key=lambda c: df[c].nunique()) if cats else None


def detect_currency(df: pd.DataFrame) -> str:
    """The currency symbol used most in the text of the (original) data, or '' if none."""
    counts = dict.fromkeys(CURRENCIES[1:], 0)
    for col in text_columns(df):
        values = _strings(df[col])
        if values.empty:
            continue
        for symbol in counts:
            counts[symbol] += int(values.str.contains(symbol, regex=False).sum())
    best = max(counts, key=counts.get)
    return best if counts[best] else ""


def money(value, symbol: str = "") -> str:
    if value is None or pd.isna(value):
        return "–"
    sign = "-" if value < 0 else ""
    return f"{sign}{symbol}{abs(value):,.2f}"


def money_short(value, symbol: str = "") -> str:
    if value is None or pd.isna(value):
        return "–"
    sign, size = ("-" if value < 0 else ""), abs(value)
    for divisor, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if size >= divisor:
            return f"{sign}{symbol}{size / divisor:,.1f}{suffix}"
    return f"{sign}{symbol}{size:,.2f}"


def _month_span(months: set) -> str:
    names = [pd.Timestamp(2000, m, 1).strftime("%B") for m in sorted(months)]
    if len(months) == 1:
        return names[0]
    if sorted(months) == list(range(min(months), max(months) + 1)):
        return f"{names[0]}–{names[-1]}"
    return f"the {len(months)} months"


def finance_summary(df: pd.DataFrame, amount, date=None, category=None, symbol: str = "") -> dict:
    """Totals, monthly trend and growth, and the breakdown by category, with plain-language insights."""
    if amount not in df.columns or not is_number(df[amount]):
        raise ValueError("Choose a number column for the amount.")
    s = df[amount].dropna()
    if s.empty:
        raise ValueError(f"{amount} has no values.")
    label = str(amount)

    def m(value):
        return money(value, symbol)

    out = {"amount": amount, "date": date, "category": category, "symbol": symbol,
           "total": float(s.sum()), "mean": float(s.mean()), "median": float(s.median()), "count": int(len(s)),
           "negatives": int((s < 0).sum()), "negative_total": float(s[s < 0].sum()),
           "monthly": None, "by_category": None, "insights": []}
    notes = out["insights"]
    notes.append(f"Total {label}: {m(out['total'])} across {_n(out['count'], 'row')}. "
                 f"The average row is {m(out['mean'])} and the median is {m(out['median'])}.")
    if out["median"] > 0 and out["mean"] > 1.5 * out["median"]:
        notes.append("The average is much higher than the median, so a few large rows make up a big part of the total.")
    if out["negatives"]:
        verb = "has a negative value" if out["negatives"] == 1 else "have negative values"
        notes.append(f"{_n(out['negatives'], 'row')} {verb} in {label} (refunds or corrections), "
                     f"adding up to {m(out['negative_total'])}.")

    if date is not None and date in df.columns and is_date(df[date]):
        sub = df[[date, amount]].dropna()
        if not sub.empty:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                monthly = sub.groupby(sub[date].dt.to_period("M"))[amount].sum()
            monthly.index = monthly.index.to_timestamp()
            months_with_rows = len(monthly)
            monthly = monthly.asfreq("MS", fill_value=0)
            empty_months = len(monthly) - months_with_rows
            table = pd.DataFrame({"month": monthly.index, "total": monthly.values.astype(float)})
            previous = table["total"].shift(1)
            change = ((table["total"] - previous) / previous.abs() * 100).where(previous != 0)
            table["change %"] = change.round(1)
            out["monthly"] = table
            if empty_months:
                notes.append(f"{_n(empty_months, 'month')} between the first and last date "
                             f"{'has' if empty_months == 1 else 'have'} no rows at all, so "
                             f"{'it counts' if empty_months == 1 else 'they count'} as 0.")
            if len(table) >= 2:
                best = table.loc[table["total"].idxmax()]
                worst = table.loc[table["total"].idxmin()]
                notes.append(f"Best month: {best['month']:%B %Y} ({m(best['total'])}). "
                             f"Weakest month: {worst['month']:%B %Y} ({m(worst['total'])}).")
                last, before = table.iloc[-1], table.iloc[-2]
                if pd.notna(last["change %"]):
                    direction = "up" if last["change %"] >= 0 else "down"
                    notes.append(f"The latest month, {last['month']:%B %Y}, is {direction} "
                                 f"{abs(last['change %']):.1f}% compared with {before['month']:%B %Y}.")
                last_day = sub[date].max()
                month_end = (last_day + pd.offsets.MonthEnd(0)).normalize()
                if last_day.normalize() < month_end - pd.Timedelta(days=2):
                    notes.append(f"The data ends on {last_day:%d %B %Y}, so {last_day:%B} may be incomplete. "
                                 "Keep that in mind when comparing it with other months.")
                year = int(table["month"].dt.year.max())
                this_year = table[table["month"].dt.year == year]
                last_year = table[table["month"].dt.year == year - 1]
                shared = set(this_year["month"].dt.month) & set(last_year["month"].dt.month)
                if shared:
                    now = this_year[this_year["month"].dt.month.isin(shared)]["total"].sum()
                    then = last_year[last_year["month"].dt.month.isin(shared)]["total"].sum()
                    if then != 0:
                        pct = (now - then) / abs(then) * 100
                        direction = "up" if pct >= 0 else "down"
                        notes.append(f"{_month_span(shared)} {year} is {direction} {abs(pct):.1f}% compared with "
                                     f"the same months of {year - 1} ({m(now)} vs {m(then)}).")

    if category is not None and category in df.columns:
        sub = df[[category, amount]].dropna(subset=[amount])
        keys = sub[category].astype(str).where(sub[category].notna(), "(missing)")
        groups = sub[amount].groupby(keys).sum().sort_values(ascending=False)
        table = pd.DataFrame({str(category): groups.index, "total": groups.values.astype(float)})
        total = float(groups.sum())
        if len(table) and (groups < 0).any():
            notes.append(f"Some {category} values have a negative total, so shares of the total are not shown.")
        elif total > 0 and len(table):
            share = table["total"] / total * 100
            table["share %"] = share.round(1)
            table["cumulative %"] = share.cumsum().round(1)
            top = table.iloc[0]
            notes.append(f"Top {category}: {top[str(category)]}, with {top['share %']:.1f}% of {label} "
                         f"({m(top['total'])}).")
            if len(table) >= 3:
                needed = int((share.cumsum() < 80).sum()) + 1
                if needed <= max(1, round(len(table) * 0.5)):
                    notes.append(f"Just {needed} of the {len(table)} different {category} values make up "
                                 f"80% of {label}.")
                else:
                    notes.append(f"{label[:1].upper() + label[1:]} is spread fairly evenly: it takes {needed} of "
                                 f"the {len(table)} different {category} values to reach 80%.")
        out["by_category"] = table
    return out


def finance_defaults(df: pd.DataFrame) -> tuple:
    """(amount, date, category) the finance view starts with."""
    dates = [c for c in date_columns(df) if df[c].notna().any()]
    return best_amount_column(df), (dates[0] if dates else None), best_category_column(df)


# ---------------------------------------------------------------------------
# Static charts for the report (matplotlib, rendered to PNG)
# ---------------------------------------------------------------------------

BLUE = "#2a78d6"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
DIVERGING = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f4b2b1", "#e66767", "#a8302f"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    })
    return plt


def _png(fig) -> tuple[str, int]:
    """Return the chart as base64 PNG plus its natural width in CSS pixels."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=144, bbox_inches="tight", facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)
    data = buf.getvalue()
    width_px = int.from_bytes(data[16:20], "big")  # the PNG header holds the width
    return base64.b64encode(data).decode(), int(width_px / 1.5)


def _short(label, n=22) -> str:
    label = str(label)
    return label if len(label) <= n else label[: n - 1] + "…"


def chart_missing(df: pd.DataFrame) -> tuple[str, int] | None:
    pct = (df.isna().mean() * 100).sort_values()
    pct = pct[pct > 0]
    if pct.empty:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(pct) + 0.9))
    ax.barh([_short(c) for c in pct.index], pct.values, color=BLUE, height=0.6)
    for y, v in enumerate(pct.values):
        ax.text(v + 0.5, y, f"{v:.1f}%", va="center", color=INK_2, fontsize=9)
    ax.set_title("Missing values by column")
    ax.set_xlabel("% of rows missing")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(10, pct.max() * 1.18))
    return _png(fig)


def chart_histograms(df: pd.DataFrame, limit: int = 9) -> tuple[str, int] | None:
    cols = [c for c in analysis_columns(df) if df[c].notna().sum() > 1][:limit]
    if not cols:
        return None
    plt = _plt()
    ncols = min(3, len(cols))
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows), squeeze=False)
    for ax, col in zip(axes.flat, cols):
        s = df[col].dropna()
        whole = bool((s == s.round()).all())
        if whole and s.nunique() <= 15:
            counts = s.astype(int).value_counts().sort_index()
            ax.bar(counts.index, counts.values, color=BLUE, width=0.7)
            ax.set_xticks(counts.index, [str(v) for v in counts.index])
        else:
            ax.hist(s, bins=min(30, max(5, s.nunique())), color=BLUE, edgecolor=SURFACE, linewidth=1)
        ax.axvline(s.median(), color=INK_2, linewidth=1.2, linestyle="--")
        ax.set_title(_short(col, 28), fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.tick_params(labelsize=8)
    for ax in list(axes.flat)[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Distribution of number columns (dashed line = median)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    return _png(fig)


def chart_categories(df: pd.DataFrame, limit: int = 6) -> tuple[str, int] | None:
    cols = []
    for c in categorical_columns(df):
        n = df[c].nunique()
        if 1 < n <= max(30, len(df) // 2) and n < len(df) * 0.95:
            cols.append(c)
    cols = cols[:limit]
    if not cols:
        return None
    plt = _plt()
    ncols = min(2, len(cols))
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 2.9 * nrows), squeeze=False)
    for ax, col in zip(axes.flat, cols):
        counts = df[col].astype(str).where(df[col].notna()).value_counts().head(8).sort_values()
        ax.barh([_short(i, 18) for i in counts.index], counts.values, color=BLUE, height=0.62)
        ax.set_title(_short(col, 30) + (" (top 8)" if df[col].nunique() > 8 else ""), fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.tick_params(labelsize=8)
    for ax in list(axes.flat)[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Most common values in text columns (rows)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    return _png(fig)


def chart_correlation(df: pd.DataFrame) -> tuple[str, int] | None:
    corr = correlations(df)
    if corr.empty:
        return None
    from matplotlib.colors import LinearSegmentedColormap
    plt = _plt()
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    n = len(corr)
    size = min(9, max(4.2, 0.62 * n + 2))
    fig, ax = plt.subplots(figsize=(size + 1, size))
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.grid(False)
    labels = [_short(c, 16) for c in corr.columns]
    ax.set_xticks(range(n), labels, rotation=40, ha="right")
    ax.set_yticks(range(n), labels)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if abs(v) > 0.6 else INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, colors=MUTED)
    ax.set_title("Correlation between number columns (−1 to +1)")
    return _png(fig)


def chart_over_time(df: pd.DataFrame) -> tuple[str, int] | None:
    cols = [c for c in date_columns(df) if df[c].notna().sum() > 2]
    if not cols:
        return None
    counts, label = rows_over_time(df[cols[0]])
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(counts.index, counts.values, color=BLUE, linewidth=2)
    ax.set_title(f"Rows per {label} ({_short(cols[0], 30)})")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(bottom=0)
    fig.autofmt_xdate()
    return _png(fig)


def chart_finance_monthly(summary: dict) -> tuple[str, int] | None:
    table = summary.get("monthly")
    if table is None or len(table) < 2:
        return None
    from matplotlib.ticker import FuncFormatter
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.bar(table["month"], table["total"], width=20, color=BLUE)
    ax.set_title(f"{_short(summary['amount'], 30)} per month")
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: money_short(v, summary["symbol"])))
    fig.autofmt_xdate()
    return _png(fig)


def chart_finance_categories(summary: dict) -> tuple[str, int] | None:
    table = summary.get("by_category")
    if table is None or len(table) < 2:
        return None
    from matplotlib.ticker import FuncFormatter
    name = table.columns[0]
    top = table.head(10).iloc[::-1]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(top) + 1))
    ax.barh([_short(v, 24) for v in top[name]], top["total"], color=BLUE, height=0.6)
    if "share %" in top.columns:
        for y, (value, share) in enumerate(zip(top["total"], top["share %"])):
            ax.text(value, y, f"  {share:.1f}%", va="center", color=INK_2, fontsize=9)
        ax.set_xlim(right=top["total"].max() * 1.15)
    ax.set_title(f"{_short(summary['amount'], 24)} by {_short(name, 24)}" + (" (top 10)" if len(table) > 10 else ""))
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: money_short(v, summary["symbol"])))
    return _png(fig)


def _finance_html(original: pd.DataFrame, final: pd.DataFrame) -> str:
    """The finance section of the report, or '' when the data has no money column."""
    amount, date, category = finance_defaults(final)
    if amount is None:
        return ""
    try:
        summary = finance_summary(final, amount, date, category, detect_currency(original))
    except ValueError:
        return ""
    sym = summary["symbol"]
    cards = [("Total " + str(amount), money_short(summary["total"], sym)),
             ("Average per row", money_short(summary["mean"], sym)),
             ("Median per row", money_short(summary["median"], sym))]
    monthly = summary["monthly"]
    if monthly is not None and len(monthly) >= 2:
        best = monthly.loc[monthly["total"].idxmax()]
        cards.append((f"Best month · {best['month']:%b %Y}", money_short(best["total"], sym)))
    cards_html = "".join(f'<div class="card"><div class="label">{html.escape(label)}</div>'
                         f'<div class="value">{html.escape(value)}</div></div>' for label, value in cards)
    notes = "".join(f"<li>{html.escape(t)}</li>" for t in summary["insights"])
    charts = []
    for fn in (chart_finance_monthly, chart_finance_categories):
        try:
            img = fn(summary)
        except Exception:
            img = None
        if img:
            data, width = img
            charts.append(f'<figure><img alt="Finance chart" style="max-width:{width}px" '
                          f'src="data:image/png;base64,{data}"></figure>')
    return (f'<h2>Finance summary</h2><p class="muted">Based on {html.escape(str(amount))}'
            + (f", by month of {html.escape(str(date))}" if date is not None else "")
            + (f" and by {html.escape(str(category))}" if category is not None else "")
            + f'.</p><section class="cards">{cards_html}</section><ul style="margin-top:16px">{notes}</ul>'
            + "".join(charts))


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

def build_script(steps: list[Step], source_name: str = "your_data.csv", options: dict | None = None) -> str:
    options = options or {"format": file_kind(source_name)}
    imports = (["import json"] if options.get("format") == "json" else []) + ["import re"]
    lines = [
        '"""',
        "Data preparation pipeline exported from PrepPilot.",
        f"Generated on {datetime.now():%Y-%m-%d %H:%M}.",
        "Put this script next to the original file and run it to reproduce the cleaned dataset.",
        '"""',
        *imports,
        "",
        "import numpy as np",
        "import pandas as pd",
        HELPERS,
        "",
        *_reader_code(source_name, options),
        "",
    ]
    for i, step in enumerate(steps, 1):
        comment = f"Step {i}: {step.title} - {step.description}"
        lines.extend("# " + part for part in textwrap.wrap(comment, 86))
        lines.append(step.code)
        lines.append("")
    lines.append('df.to_csv("cleaned_data.csv", index=False)')
    lines.append('print(f"Saved cleaned_data.csv with {len(df):,} rows and {df.shape[1]} columns.")')
    return "\n".join(lines) + "\n"


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    out = df.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.DatetimeTZDtype):
            out[c] = out[c].dt.tz_localize(None)
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name="cleaned_data")
    return buf.getvalue()


def _table(df: pd.DataFrame, max_rows: int = 200) -> str:
    view = df.head(max_rows)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in view.columns)
    body = []
    for _, row in view.iterrows():
        cells = "".join(f"<td>{html.escape(_fmt(v) if not pd.isna(v) else '')}</td>" for v in row.values)
        body.append(f"<tr>{cells}</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _logo_svg(size: int) -> str:
    """The PrepPilot logo (a heatmap that spells M), inline so the report stays one file."""
    filled = [(0, 0), (4, 0), (0, 1), (1, 1), (3, 1), (4, 1), (0, 2), (4, 2), (0, 3), (4, 3), (0, 4), (4, 4)]
    cells = []
    for row in range(5):
        for col in range(5):
            x, y = 9.5 + 17 * col, 9.5 + 17 * row
            if (col, row) == (2, 2):
                fill = 'fill="#EB6834"'
            elif (col, row) in filled:
                fill = 'fill="#4C93E8"'
            else:
                fill = 'fill="#FFFFFF" fill-opacity="0.09"'
            cells.append(f'<rect x="{x}" y="{y}" width="13" height="13" rx="3" {fill}/>')
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 100 100" role="img" aria-label="PrepPilot logo" '
            f'style="display:block;margin-bottom:14px"><rect width="100" height="100" rx="22" fill="#0F1B33"/>'
            + "".join(cells) + "</svg>")


def build_report(original: pd.DataFrame, final: pd.DataFrame, steps: list[Step],
                 dataset_name: str = "dataset", options: dict | None = None) -> str:
    """A self-contained HTML EDA report comparing the original and final data."""
    before, after = overview(original), overview(final)
    esc = html.escape

    def card(label, key, lower_is_better=False):
        b, a = before[key], after[key]
        delta = ""
        if b != a:
            cls = "good" if (lower_is_better and a < b) else ""
            delta = f'<span class="delta {cls}">was {b:,}</span>'
        return f'<div class="card"><div class="label">{label}</div><div class="value">{a:,}</div>{delta}</div>'

    cards = "".join([
        card("Rows", "rows"), card("Columns", "columns"),
        card("Missing cells", "missing", lower_is_better=True),
        card("Duplicate rows", "duplicates", lower_is_better=True),
    ])

    def items(texts, empty):
        return "".join(f"<li>{esc(t)}</li>" for t in texts) or f'<li class="ok">{empty}</li>'

    final_found = findings(final)
    final_issues = [t for k, t in final_found if k == "issue"]
    patterns = [t for k, t in final_found if k == "note"]
    if steps:
        original_issues = [t for k, t in findings(original) if k == "issue"]
        quality = (
            '<div class="two">'
            f'<div class="panel"><h3>In the original file</h3><ul>{items(original_issues, "No issues found.")}</ul></div>'
            f'<div class="panel"><h3>After preparation</h3><ul>{items(final_issues, "✓ No issues left.")}</ul></div>'
            "</div>"
        )
    else:
        quality = f'<div class="panel"><ul>{items(final_issues, "✓ No data quality issues found.")}</ul></div>'

    charts = []
    for title, fn in [("Missing values", chart_missing), ("Distributions", chart_histograms),
                      ("Categories", chart_categories), ("Over time", chart_over_time),
                      ("Correlations", chart_correlation)]:
        try:
            img = fn(final)
        except Exception:  # a chart should never break the report
            img = None
        if img:
            data, width = img
            charts.append(f'<figure><img alt="{esc(title)} chart" style="max-width:{width}px" '
                          f'src="data:image/png;base64,{data}"></figure>')

    step_items = "".join(
        f"<li><strong>{esc(s.title)}</strong> — {esc(s.description)} <span class='muted'>({esc(s.changes)})</span></li>"
        for s in steps) or "<li>No steps applied. This report describes the original data.</li>"
    num = numeric_summary(final)
    num_html = (_table(num.reset_index().rename(columns={"index": "column"})) if not num.empty
                else "<p class='muted'>No number columns.</p>")
    script = esc(build_script(steps, dataset_name, options))
    pattern_html = f"<ul>{items(patterns, 'Nothing unusual.')}</ul>"
    try:
        finance_html = _finance_html(original, final)
    except Exception:  # the finance section should never break the report
        finance_html = ""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EDA Report – {esc(dataset_name)}</title>
<style>
:root {{ --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --line:#e1e0d9; --accent:#2a78d6; --good:#006300; --code:#f3f2ee; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d0d0d; --surface:#1a1a19; --ink:#f0efec;
  --ink-2:#c3c2b7; --muted:#898781; --line:#2c2c2a; --accent:#3987e5; --good:#0ca30c; --code:#232321; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1080px; margin:0 auto; padding:40px 16px 64px; }}
header p {{ color:var(--ink-2); margin:4px 0 0; }}
h1 {{ font-size:28px; margin:0; letter-spacing:-0.01em; }}
h2 {{ font-size:19px; margin:40px 0 12px; padding-bottom:8px; border-bottom:1px solid var(--line); }}
h3 {{ font-size:14px; margin:0 0 6px; color:var(--ink-2); text-transform:uppercase; letter-spacing:.04em; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-top:24px; }}
.card, .panel {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
.two {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:12px; }}
.label {{ color:var(--ink-2); font-size:13px; }}
.value {{ font-size:26px; font-weight:650; margin-top:2px; }}
.delta {{ color:var(--muted); font-size:13px; }} .delta.good {{ color:var(--good); }}
ul, ol {{ padding-left:20px; margin:0; }} li {{ margin:6px 0; }}
li.ok {{ list-style:none; margin-left:-20px; color:var(--good); font-weight:600; }}
.muted {{ color:var(--muted); }}
figure {{ margin:0 0 16px; background:#fcfcfb; border:1px solid var(--line); border-radius:10px; padding:12px; }}
figure img {{ width:100%; height:auto; display:block; margin:0 auto; }}
.table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:10px; background:var(--surface); }}
table {{ border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }}
th,td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }}
thead th {{ color:var(--ink-2); font-weight:600; position:sticky; top:0; background:var(--surface); }}
tbody tr:last-child td {{ border-bottom:0; }}
pre {{ background:var(--code); border:1px solid var(--line); border-radius:10px; padding:16px; overflow-x:auto; font-size:12.5px; line-height:1.5; }}
footer {{ margin-top:48px; color:var(--muted); font-size:13px; }}
</style></head>
<body><main>
<header>
  {_logo_svg(40)}
  <h1>Exploratory Data Analysis</h1>
  <p>{esc(dataset_name)} · generated {datetime.now():%d %B %Y, %H:%M}</p>
</header>
<section class="cards">{cards}</section>

<h2>Data quality</h2>
{quality}

<h2>Patterns worth knowing</h2>
{pattern_html}

{finance_html}

<h2>Preparation steps</h2>
<ol>{step_items}</ol>

<h2>Column overview</h2>
{_table(column_profile(final))}

<h2>Number columns</h2>
{num_html}

<h2>Charts</h2>
{''.join(charts) or "<p class='muted'>Not enough data for charts.</p>"}

<h2>Data preview (first 10 rows)</h2>
{_table(final, 10)}

<h2>Reproducible Python code</h2>
<p class="muted">Put this script next to the original file and run it to get exactly the same cleaned dataset.</p>
<pre><code>{script}</code></pre>

<footer>Generated with PrepPilot by Mohamad Al Tashem.</footer>
</main></body></html>"""
