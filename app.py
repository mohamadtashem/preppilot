"""
PrepPilot - a guided data preparation app in five steps:

1. Upload   2. Explore   3. Clean   4. Transform   5. Results

Run locally:  streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

import engine as E

# Shown in the sidebar as "Built by ...".
AUTHOR = "Mohamad Al Tashem"
AUTHOR_LINK = "https://www.linkedin.com/in/mohamad14"

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE = PALETTE[0]
DIVERGING = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f4b2b1", "#e66767", "#a8302f"]
SAMPLE = Path(__file__).parent / "sample_data" / "messy_sales.csv"
LOGO = Path(__file__).parent / "assets" / "logo.png"
SPLASH = Path(__file__).parent / "assets" / "splash.html"
FAVICON = Path(__file__).parent / "assets" / "favicon.png"
PAGES = ["Upload", "Explore", "Clean", "Transform", "Results"]
UPLOAD_TYPES = ["csv", "tsv", "txt", "xlsx", "xlsm", "json", "jsonl"]
DATE_FORMATS = {"Detect automatically": None, "Day first (31/12/2024)": True, "Month first (12/31/2024)": False}
CHOICE_LABELS = {
    "median": "Fill with the median", "mean": "Fill with the mean", "zero": "Fill with 0",
    "drop rows": "Remove those rows", "leave as is": "Leave as is", "Unknown": "Fill with 'Unknown'",
    "most common value": "Fill with the most common value", "previous value": "Fill with the previous row's value",
}

try:
    from PIL import Image
    _icon = Image.open(FAVICON)
except Exception:  # the app still works without the icon file
    _icon = "📊"
st.set_page_config(page_title="PrepPilot", page_icon=_icon, layout="wide")
if hasattr(st, "logo") and LOGO.exists():
    try:
        st.logo(str(LOGO), size="large", link=AUTHOR_LINK)
    except TypeError:  # older Streamlit without the size option
        st.logo(str(LOGO), link=AUTHOR_LINK)
state = st.session_state
state.setdefault("page", 1)

# A short animated intro with the logo, shown once when someone opens the app.
if not state.get("splash_done") and SPLASH.exists():
    state.splash_done = True
    st.markdown(SPLASH.read_text(encoding="utf-8"), unsafe_allow_html=True)

st.markdown("""
<style>
.dps-steps { display:flex; gap:8px; margin:0 0 1.2rem; }
.dps-step { flex:1 1 0; display:flex; align-items:center; gap:8px; padding:8px 12px; border-radius:10px;
  border:1px solid rgba(128,128,128,.28); font-size:.92rem; opacity:.6; }
.dps-step .dot { flex:none; width:24px; height:24px; border-radius:50%; display:inline-flex; align-items:center;
  justify-content:center; font-size:.8rem; font-weight:600; border:1.5px solid currentColor; }
.dps-step.done { opacity:.85; }
.dps-step.done .dot { border-color:#2a78d6; color:#2a78d6; }
.dps-step.active { opacity:1; font-weight:600; border-color:#2a78d6; box-shadow:inset 0 0 0 1px #2a78d6; }
.dps-step.active .dot { background:#2a78d6; border-color:#2a78d6; color:#fff; }
@media (max-width: 640px) { .dps-step .label { display:none; } .dps-step { justify-content:center; padding:6px; } }
[data-testid="stSidebar"] [data-testid="stButton"] button,
[data-testid="stSidebar"] .stButton button { width:100%; justify-content:flex-start; }
</style>
""", unsafe_allow_html=True)

SCROLL_TOP = """<script>
const doc = window.parent.document;
for (const sel of ['[data-testid="stMain"]', '[data-testid="stAppViewContainer"]', 'section.main']) {
  const el = doc.querySelector(sel);
  if (el) el.scrollTo({top: 0});
}
window.parent.scrollTo(0, 0);
</script>"""


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

_MD = str.maketrans({c: "\\" + c for c in "\\`*_[]<>#|~$"})


def esc_md(text) -> str:
    """Show column names and values literally inside Markdown."""
    return str(text).translate(_MD)


def has_data() -> bool:
    return "original" in state


def start(df: pd.DataFrame, name: str, options: dict) -> None:
    state.original = df
    state.read_options = options
    state.name = name
    state.history = []  # list of (dataframe after the step, Step)
    state.dataset_id = state.get("dataset_id", 0) + 1
    state.version = 0
    state.cache = {}


def current() -> pd.DataFrame:
    return state.history[-1][0] if state.history else state.original


def steps() -> list:
    return [step for _, step in state.history]


def data_key() -> tuple:
    return (state.dataset_id, state.version)


def wkey(name: str) -> str:
    """Widget key that resets whenever the data changes, so pickers never hold a removed column."""
    return f"{name}_{state.dataset_id}_{state.version}"


def changed() -> None:
    state.version += 1


def goto(page: int) -> None:
    state.page = page
    state.scroll_top = True


def flash(message: str, kind: str = "success") -> None:
    state.flash = (kind, message)


def cached(name: str, key, compute, spinner: str | None = None):
    """Keep expensive results for this session only, until the data changes."""
    store = state.setdefault("cache", {})
    full_key = (name, key)
    if full_key not in store:
        if len(store) > 40:
            store.clear()
        if spinner:
            with st.spinner(spinner):
                store[full_key] = compute()
        else:
            store[full_key] = compute()
    return store[full_key]


def analysis(df: pd.DataFrame, key=None) -> dict:
    def compute():
        return {"overview": E.overview(df), "findings": E.findings(df), "profile": E.column_profile(df),
                "numbers": E.numeric_summary(df), "corr": E.correlations(df)}
    return cached("analysis", key or data_key(), compute, "Analysing your data...")


def step_list(results) -> str:
    return "\n".join(f"- **{s.title}:** {esc_md(s.description)}" for _, s in results)


def run(planner, *args, **kwargs) -> None:
    """Plan one step, apply it, record it and refresh the page."""
    try:
        plan = planner(current(), *args, **kwargs)
        new_df, step = E.apply_step(current(), plan)
    except E.NothingToDo as exc:
        st.info(str(exc))
        return
    except Exception as exc:  # show problems in plain words instead of a stack trace
        st.error(str(exc))
        return
    state.history.append((new_df, step))
    changed()
    flash(f"**{step.title}:** {esc_md(step.description)} ({step.changes})")
    st.rerun()


def apply_results(results, heading: str, next_page: int | None = None) -> None:
    if results:
        state.history.extend(results)
        changed()
        flash(f"{heading}\n\n{step_list(results)}")
    else:
        flash("Nothing needed changing: your data already passes these checks.", "info")
    if next_page:
        goto(next_page)
    st.rerun()


def do_everything() -> None:
    try:
        results = E.run_everything(current())
    except Exception as exc:
        st.error(f"Something went wrong while preparing the data: {exc}")
        return
    apply_results(results, "**⚡ Done! Your data was cleaned and prepared automatically:**", next_page=5)


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def styled(fig, height: int = 320):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
    return fig


def friendly_categories(frame: pd.DataFrame) -> list:
    """Text columns ordered so the ones that make good groups (few distinct values) come first."""
    cats = E.categorical_columns(frame)
    return sorted(cats, key=lambda c: (frame[c].nunique() > 30, frame[c].nunique()))


def fold(series: pd.Series, keep: int = 8) -> pd.Series:
    """Keep the most common categories and group the rest as 'Other' (for chart colors)."""
    s = series.astype(str).where(series.notna(), "Missing")
    top = s.value_counts().index[: keep - 1]
    return s.where(s.isin(top), "Other") if s.nunique() > keep else s


def column_chart(df: pd.DataFrame, col):
    """The natural chart for one column: histogram, bars of top values, or rows over time."""
    s = df[col]
    name = str(col)
    if E.is_number(s):
        data = s.dropna()
        if data.empty:
            return None
        if data.nunique() <= 15 and bool((data == data.round()).all()):
            counts = data.value_counts().sort_index().rename_axis(name).reset_index(name="rows")
            fig = px.bar(counts, x=name, y="rows", color_discrete_sequence=[BLUE])
            fig.update_xaxes(type="category")
        else:
            fig = px.histogram(data.to_frame(name), x=name, nbins=40, color_discrete_sequence=[BLUE])
            fig.update_traces(marker_line_width=1, marker_line_color="rgba(255,255,255,0.8)")
            fig.add_vline(x=float(data.median()), line_dash="dash", line_color="#898781",
                          annotation_text="median", annotation_position="top")
    elif E.is_date(s):
        counts, label = E.rows_over_time(s)
        if counts.empty:
            return None
        data = counts.rename_axis("date").reset_index(name="rows")
        fig = px.line(data, x="date", y="rows", markers=True, color_discrete_sequence=[BLUE],
                      labels={"rows": f"rows per {label}"})
        fig.update_traces(line_width=2, marker_size=7)
    else:
        values = s.astype(str).where(s.notna(), "(missing)")
        counts = values.value_counts().head(12).rename_axis("value").reset_index(name="rows")
        fig = px.bar(counts, x="rows", y="value", orientation="h", color_discrete_sequence=[BLUE])
        fig.update_yaxes(categoryorder="total ascending", title=None)
    return styled(fig)


def missing_chart(df: pd.DataFrame):
    pct = (df.isna().mean() * 100).round(1)
    pct = pct[pct > 0].sort_values()
    if pct.empty:
        return None
    data = pd.DataFrame({"column": pct.index.astype(str), "missing %": pct.values})
    fig = px.bar(data, x="missing %", y="column", orientation="h", text="missing %",
                 color_discrete_sequence=[BLUE])
    fig.update_traces(texttemplate="%{text}%", textposition="outside", cliponaxis=False)
    fig.update_yaxes(title=None)
    return styled(fig, height=max(220, 28 * len(pct) + 80))


def heatmap(corr: pd.DataFrame):
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=DIVERGING, zmin=-1, zmax=1, aspect="auto")
    return styled(fig, height=max(320, 40 * len(corr) + 120))


def interesting_columns(df: pd.DataFrame) -> list:
    return (E.analysis_columns(df)
            + [c for c in friendly_categories(df) if 1 < df[c].nunique() <= 50]
            + E.date_columns(df))


def auto_charts(df: pd.DataFrame) -> list:
    """The charts for the results page, chosen from the data."""
    charts = []
    miss = missing_chart(df)
    if miss is not None:
        charts.append(("Missing values (% of rows)", miss))
    for col in [c for c in E.analysis_columns(df) if df[c].notna().sum() > 1][:6]:
        charts.append((f"Distribution of {col}", column_chart(df, col)))
    for col in [c for c in friendly_categories(df) if 1 < df[c].nunique() <= 50][:4]:
        charts.append((f"Most common values of {col}", column_chart(df, col)))
    dates = [c for c in E.date_columns(df) if df[c].notna().sum() > 2]
    if dates:
        charts.append((f"Rows over time ({dates[0]})", column_chart(df, dates[0])))
    corr = E.correlations(df)
    if not corr.empty:
        charts.append(("Correlation between number columns", heatmap(corr)))
    return [(title, fig) for title, fig in charts if fig is not None]


def chart_builder(df: pd.DataFrame) -> None:
    numbers, dates, cats = E.numeric_columns(df), E.date_columns(df), friendly_categories(df)
    kinds = ["Histogram", "Box plot", "Bar chart", "Line over time", "Scatter plot", "Correlation heatmap"]
    kind = st.selectbox("Chart type", kinds, key="cb_kind")
    fig, note = None, ""
    try:
        if kind == "Histogram":
            if not numbers:
                note = "Needs a number column."
            else:
                v1, v2, v3 = st.columns(3)
                x = v1.selectbox("Column", numbers, key=wkey("cb_h_x"))
                split = v2.selectbox("Split by (optional)", ["None"] + cats, key=wkey("cb_h_split"))
                bins = v3.slider("Bins", 5, 100, 30, key=wkey("cb_h_bins"))
                data = df.assign(_group=fold(df[split])) if split != "None" else df
                fig = px.histogram(data, x=x, nbins=bins, color="_group" if split != "None" else None,
                                   barmode="overlay", opacity=0.8 if split != "None" else 1,
                                   color_discrete_sequence=PALETTE, labels={"_group": split})
        elif kind == "Box plot":
            if not numbers:
                note = "Needs a number column."
            else:
                v1, v2 = st.columns(2)
                y = v1.selectbox("Value", numbers, key=wkey("cb_b_y"))
                group = v2.selectbox("Group by (optional)", ["None"] + cats, key=wkey("cb_b_group"))
                if group == "None":
                    fig = px.box(df, y=y, color_discrete_sequence=PALETTE)
                else:
                    data = df.assign(_group=fold(df[group]))
                    fig = px.box(data, x="_group", y=y, color="_group", color_discrete_sequence=PALETTE,
                                 labels={"_group": group})
                    fig.update_layout(showlegend=False)
        elif kind == "Bar chart":
            if not cats:
                note = "Needs a text or category column."
            else:
                v1, v2, v3 = st.columns(3)
                x = v1.selectbox("Category", cats, key=wkey("cb_bar_x"))
                y = v2.selectbox("Value", ["Count of rows"] + numbers, key=wkey("cb_bar_y"))
                func = v3.selectbox("Calculation", ["sum", "mean", "median", "max", "min"], key=wkey("cb_bar_f"),
                                    disabled=y == "Count of rows")
                if y == "Count of rows":
                    data = df[x].astype(str).where(df[x].notna(), "Missing").value_counts().rename_axis(x)
                    data = data.reset_index(name="rows")
                    y_col = "rows"
                else:
                    data = df.groupby(x, observed=True)[y].agg(func).reset_index()
                    y_col = y
                data = data.sort_values(y_col, ascending=False).head(25)
                fig = px.bar(data, x=x, y=y_col, color_discrete_sequence=PALETTE,
                             labels={y_col: y if y == "Count of rows" else f"{func} of {y}"})
                if len(data) == 25:
                    note = "Showing the top 25 categories."
        elif kind == "Line over time":
            if not dates:
                note = "Needs a date column. Convert one in the Clean step."
            else:
                v1, v2, v3, v4 = st.columns(4)
                d = v1.selectbox("Date", dates, key=wkey("cb_l_d"))
                y = v2.selectbox("Value", ["Count of rows"] + numbers, key=wkey("cb_l_y"))
                func = v3.selectbox("Calculation", ["sum", "mean", "median", "max", "min"], key=wkey("cb_l_f"),
                                    disabled=y == "Count of rows")
                period = v4.selectbox("Per", ["month", "week", "day"], key=wkey("cb_l_p"))
                freq = {"month": "MS", "week": "W", "day": "D"}[period]
                grouped = df.dropna(subset=[d]).groupby(pd.Grouper(key=d, freq=freq))
                data = (grouped.size().rename("rows") if y == "Count of rows" else grouped[y].agg(func)).reset_index()
                y_col = "rows" if y == "Count of rows" else y
                fig = px.line(data, x=d, y=y_col, markers=True, color_discrete_sequence=PALETTE,
                              labels={y_col: y if y == "Count of rows" else f"{func} of {y}"})
                fig.update_traces(line_width=2, marker_size=8)
        elif kind == "Scatter plot":
            if len(numbers) < 2:
                note = "Needs two number columns."
            else:
                v1, v2, v3 = st.columns(3)
                x = v1.selectbox("X axis", numbers, key=wkey("cb_s_x"))
                y = v2.selectbox("Y axis", numbers, index=1, key=wkey("cb_s_y"))
                color = v3.selectbox("Color by (optional)", ["None"] + cats, key=wkey("cb_s_c"))
                data = df.assign(_group=fold(df[color])) if color != "None" else df
                fig = px.scatter(data, x=x, y=y, color="_group" if color != "None" else None,
                                 color_discrete_sequence=PALETTE, opacity=0.8, labels={"_group": color})
                fig.update_traces(marker=dict(size=8, line=dict(width=1, color="white")))
                note = f"Correlation between {x} and {y}: r = {df[[x, y]].corr().iloc[0, 1]:.2f}"
        elif kind == "Correlation heatmap":
            corr = E.correlations(df)
            if corr.empty:
                note = "Needs at least two number columns."
            else:
                fig = heatmap(corr)
    except Exception as exc:
        note = f"Could not draw this chart: {exc}"
    if fig is not None:
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
        st.plotly_chart(fig, key="cb_chart")
    if note:
        st.caption(esc_md(note))


def finance_view(df: pd.DataFrame) -> None:
    """Money totals, monthly trend and growth, and the biggest categories."""
    numbers = E.analysis_columns(df)
    if not numbers:
        st.caption("The finance view needs a number column with money values, such as revenue or amount.")
        return
    money_cols = E.money_columns(df)
    amounts = money_cols + [c for c in numbers if c not in money_cols]
    default_amount, default_date, default_cat = E.finance_defaults(df)
    dates = ["None"] + [c for c in E.date_columns(df) if df[c].notna().any()]
    cats = ["None"] + [c for c in friendly_categories(df) if 1 < df[c].nunique() <= 200]
    currency = cached("currency", ("original", state.dataset_id), lambda: E.detect_currency(state.original))

    f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
    amount = f1.selectbox("Money column", amounts, key=wkey("fin_amount"),
                          index=amounts.index(default_amount) if default_amount in amounts else 0)
    date = f2.selectbox("Date", dates, key=wkey("fin_date"),
                        index=dates.index(default_date) if default_date in dates else 0)
    category = f3.selectbox("Break down by", cats, key=wkey("fin_cat"),
                            index=cats.index(default_cat) if default_cat in cats else 0)
    symbol = f4.selectbox("Currency", E.CURRENCIES, index=E.CURRENCIES.index(currency), key=wkey("fin_symbol"),
                          format_func=lambda c: c or "none")
    if not money_cols:
        st.caption("No column name looks like money (revenue, sales, amount, price...), "
                   "so check that the money column above is the right one.")
    try:
        fs = cached("finance", (data_key(), amount, date, category, symbol),
                    lambda: E.finance_summary(df, amount, None if date == "None" else date,
                                              None if category == "None" else category, symbol))
    except ValueError as exc:
        st.error(str(exc))
        return

    monthly, by_cat = fs["monthly"], fs["by_category"]
    m = st.columns(4)
    m[0].metric(f"Total {amount}", E.money_short(fs["total"], symbol), help=esc_md(E.money(fs["total"], symbol)))
    m[1].metric("Average per row", E.money_short(fs["mean"], symbol),
                help=esc_md(f"Median: {E.money(fs['median'], symbol)}"))
    if monthly is not None and len(monthly) >= 2:
        best, last = monthly.loc[monthly["total"].idxmax()], monthly.iloc[-1]
        m[2].metric(f"Best month · {best['month']:%b %Y}", E.money_short(best["total"], symbol))
        change = last["change %"]
        m[3].metric(f"Latest month · {last['month']:%b %Y}", E.money_short(last["total"], symbol),
                    f"{change:+.1f}% vs previous month" if pd.notna(change) else None)
    else:
        m[2].metric("Rows", f"{fs['count']:,}")
        m[3].metric("Negative rows", f"{fs['negatives']:,}", help="Refunds or corrections")

    st.markdown("\n".join(f"- {esc_md(t)}" for t in fs["insights"]))

    c1, c2 = st.columns(2)
    if monthly is not None and len(monthly) >= 2:
        with c1:
            st.markdown(f"**{esc_md(amount)} per month**")
            fig = px.bar(monthly, x="month", y="total", hover_data={"change %": True},
                         labels={"total": str(amount), "month": ""}, color_discrete_sequence=[BLUE])
            fig.update_yaxes(tickprefix=symbol)
            st.plotly_chart(styled(fig), key="fin_month")
    if by_cat is not None and len(by_cat) >= 2:
        with c2:
            name = by_cat.columns[0]
            top = by_cat.head(10)
            st.markdown(f"**{esc_md(amount)} by {esc_md(name)}**" + (" (top 10)" if len(by_cat) > 10 else ""))
            has_share = "share %" in top.columns
            fig = px.bar(top, x="total", y=name, orientation="h", text="share %" if has_share else None,
                         labels={"total": str(amount), name: ""}, color_discrete_sequence=[BLUE])
            if has_share:
                fig.update_traces(texttemplate="%{text}%", textposition="outside", cliponaxis=False)
            fig.update_yaxes(categoryorder="total ascending")
            fig.update_xaxes(tickprefix=symbol)
            st.plotly_chart(styled(fig), key="fin_category")
    if monthly is not None or by_cat is not None:
        with st.expander("Finance tables"):
            if monthly is not None:
                st.markdown("**By month**")
                st.dataframe(monthly.assign(month=monthly["month"].dt.strftime("%Y-%m")), hide_index=True)
            if by_cat is not None:
                st.markdown(f"**By {esc_md(by_cat.columns[0])}**")
                st.dataframe(by_cat, hide_index=True)


def show_issues(found, empty: str) -> None:
    issues = [t for k, t in found if k == "issue"]
    if issues:
        st.markdown("\n".join(f"- {esc_md(t)}" for t in issues))
    else:
        st.success(empty)


def show_patterns(found) -> None:
    notes = [t for k, t in found if k == "note"]
    st.markdown("\n".join(f"- {esc_md(t)}" for t in notes) if notes else "Nothing unusual.")


# ---------------------------------------------------------------------------
# Page 1: Upload
# ---------------------------------------------------------------------------

def load_sample() -> None:
    raw = SAMPLE.read_bytes()
    options = E.read_options(SAMPLE.name, raw)
    start(E.load_file(SAMPLE.name, raw, options), SAMPLE.name, options)
    goto(2)
    st.rerun()


def preview_upload(upload) -> None:
    raw = upload.getvalue()
    sheet = None
    if E.file_kind(upload.name) == "excel":
        try:
            sheets = E.excel_sheets(raw)
        except Exception as exc:
            st.error(f"Could not open this Excel file: {exc}")
            return
        if len(sheets) > 1:
            sheet = st.selectbox("This file has several sheets. Which one should we use?", sheets, key=f"sheet_{upload.name}_{upload.size}")
        else:
            sheet = sheets[0]
    key = (upload.name, upload.size, sheet)
    if state.get("preview_key") != key:
        try:
            options = E.read_options(upload.name, raw, sheet)
            df = E.load_file(upload.name, raw, options)
        except Exception as exc:
            message = str(exc)
            st.error(message if message.startswith("Could not") else f"Could not read this file: {message}")
            return
        state.preview_key, state.preview = key, (df, options)
    df, options = state.preview
    if df.empty or df.shape[1] == 0:
        st.warning("The file was read, but it has no data in it.")
        return
    rows = f"{len(df):,} row" + ("" if len(df) == 1 else "s")
    cols = f"{df.shape[1]} column" + ("" if df.shape[1] == 1 else "s")
    st.success(f"Read **{rows}** and **{cols}** · {esc_md(E.describe_format(options))}")
    st.dataframe(df.head(10), hide_index=True)
    if has_data():
        st.caption("Continuing replaces the dataset you are working on now.")
    if st.button("Continue: explore the data →", type="primary", key="upload_continue"):
        start(df, upload.name, options)
        state.pop("preview", None)
        state.pop("preview_key", None)
        goto(2)
        st.rerun()


def page_upload() -> None:
    st.title("Upload your dataset")
    st.markdown("Start with a **CSV, Excel or JSON** file. It is only used during your session and is not stored.")
    if has_data():
        with st.container(border=True):
            c1, c2 = st.columns([3, 1], vertical_alignment="center")
            c1.markdown(f"You are working on **{esc_md(state.name)}** ({len(current()):,} rows). "
                        "Upload a new file below to replace it.")
            c2.button("Continue →", key="resume", on_click=goto, args=(2,))
    upload = st.file_uploader("Choose a file", type=UPLOAD_TYPES,
                              help="CSV, TSV or TXT, Excel (.xlsx, .xlsm), JSON or JSON Lines")
    if upload is not None:
        preview_upload(upload)
        return
    with st.container(border=True):
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        c1.markdown("**No file at hand?** Try a sample of 498 sales orders with typical problems: duplicates, "
                    "mixed date formats, prices stored as text, inconsistent spelling and outliers.")
        if c2.button("Use sample data", type="primary", key="use_sample"):
            load_sample()
    st.markdown("#### How it works")
    cols = st.columns(5)
    for col, (title, text) in zip(cols, [
        ("1 · Upload", "CSV, Excel or JSON. Several Excel sheets? Pick one."),
        ("2 · Explore", "An automatic analysis finds data quality problems."),
        ("3 · Clean", "Tick the fixes you want, or press ⚡ Do everything."),
        ("4 · Transform", "Filter, sort, add calculated columns and summarize."),
        ("5 · Results", "Charts, a finance view, an EDA report, clean data and the Python code."),
    ]):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(text)


# ---------------------------------------------------------------------------
# Page 2: Explore
# ---------------------------------------------------------------------------

def next_actions(where: str) -> None:
    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 1.3, 1.5], vertical_alignment="center")
        c1.markdown("**Next:** choose the cleaning steps yourself, or let the app clean and prepare everything.")
        if c2.button("⚡ Do everything", type="primary", key=f"auto_{where}"):
            do_everything()
        c3.button("Choose cleaning steps →", key=f"choose_{where}", on_click=goto, args=(3,))


def page_explore() -> None:
    df = current()
    a = analysis(df)
    o = a["overview"]
    st.title("Explore your data")
    after = f" · after {len(state.history)} preparation steps" if state.history else ""
    st.caption(f"{esc_md(state.name)}{after}")

    m = st.columns(4)
    m[0].metric("Rows", f"{o['rows']:,}")
    m[1].metric("Columns", o["columns"], help=f"{o['numeric']} number, {o['text']} text, {o['dates']} date")
    m[2].metric("Missing cells", f"{o['missing']:,}", help=f"{o['missing_pct']:.1f}% of all cells")
    m[3].metric("Duplicate rows", f"{o['duplicates']:,}")

    st.subheader("Data quality check")
    c1, c2 = st.columns(2)
    with c1:
        issues = sum(1 for k, _ in a["findings"] if k == "issue")
        st.markdown(f"**⚠️ Issues to fix ({issues})**")
        show_issues(a["findings"], "No data quality issues found.")
    with c2:
        st.markdown("**💡 Patterns**")
        show_patterns(a["findings"])

    next_actions("top")

    st.subheader("Preview")
    st.dataframe(df.head(100), hide_index=True)
    st.subheader("Columns")
    st.dataframe(a["profile"], hide_index=True)
    if not a["numbers"].empty:
        st.subheader("Number columns")
        st.dataframe(a["numbers"])

    st.subheader("Charts")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Missing values by column**")
        miss = missing_chart(df)
        if miss is not None:
            st.plotly_chart(miss, key="ex_missing")
        else:
            st.success("No missing values.")
    with c2:
        st.markdown("**Look at one column**")
        cols = list(df.columns)
        best = interesting_columns(df)
        pick = st.selectbox("Column", cols, index=cols.index(best[0]) if best else 0, key=wkey("ex_col"),
                            label_visibility="collapsed")
        fig = column_chart(df, pick)
        if fig is not None:
            st.plotly_chart(fig, key="ex_column")
        row = a["profile"].iloc[cols.index(pick)]
        st.caption(f"{row['Type']} · {row['Filled']:,} filled · {row['Missing']:,} missing · "
                   f"{row['Unique']:,} unique values")
    if not a["corr"].empty:
        st.markdown("**Correlation between number columns**")
        st.plotly_chart(heatmap(a["corr"]), key="ex_corr")

    next_actions("bottom")
    st.button("← Upload a different file", key="explore_back", on_click=goto, args=(1,))


# ---------------------------------------------------------------------------
# Page 3: Clean
# ---------------------------------------------------------------------------

def clean_checks(df: pd.DataFrame) -> dict:
    def compute():
        return {
            "names": E.column_name_changes(df),
            "messy": E.messy_text_counts(df),
            "constant": E.constant_columns(df),
            "types": E.detect_types(df),
            "variants": E.spelling_variants(df),
            "duplicates": int(df.duplicated().sum()),
            "outliers": {rule: E.outlier_counts(df, *E.OUTLIER_RULES[rule]) for rule in E.OUTLIER_RULES},
            "missing": E.missing_by_type(df),
        }
    return cached("checks", data_key(), compute, "Checking your data...")


def advanced_tools(df: pd.DataFrame) -> None:
    all_cols, texts, numbers = list(df.columns), E.text_columns(df), E.numeric_columns(df)
    tool = st.radio("Tool", ["Missing values", "Column type", "Capital letters", "Outliers",
                             "Duplicates", "Drop columns"], horizontal=True, key="adv_tool")
    if tool == "Missing values":
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values.")
            return
        cols = st.multiselect("Columns", list(missing.index), key=wkey("adv_miss_cols"),
                              format_func=lambda c: f"{c} ({missing[c]:,} missing)")
        strategy = st.selectbox("What to do", ["median", "mean", "most common value", "fixed value",
                                               "previous value", "drop rows"], key=wkey("adv_miss_strategy"),
                                format_func=lambda s: CHOICE_LABELS.get(s, "Fill with a value I type"))
        value = st.text_input("Fill with", key=wkey("adv_miss_value")) if strategy == "fixed value" else None
        if st.button("Apply", key="adv_miss_apply"):
            run(E.plan_missing, cols, strategy, value)
    elif tool == "Column type":
        t1, t2, t3 = st.columns(3)
        col = t1.selectbox("Column", all_cols, key=wkey("adv_type_col"))
        to = t2.selectbox("New type", ["number", "date", "text", "category"], key=wkey("adv_type_to"))
        fmt = t3.selectbox("Date format", list(DATE_FORMATS), key=wkey("adv_type_fmt"), disabled=to != "date")
        if st.button("Convert", key="adv_type_apply"):
            run(E.plan_convert_types, {col: to}, dayfirst=DATE_FORMATS[fmt])
    elif tool == "Capital letters":
        cols = st.multiselect("Text columns", texts, key=wkey("adv_case_cols"))
        case = st.radio("Style", ["Title Case", "lowercase", "UPPERCASE"], horizontal=True, key=wkey("adv_case"))
        if st.button("Apply", key="adv_case_apply"):
            run(E.plan_text_case, cols, case)
    elif tool == "Outliers":
        if not numbers:
            st.caption("No number columns yet. Fix the column types first.")
            return
        cols = st.multiselect("Number columns", numbers, key=wkey("adv_out_cols"))
        o1, o2 = st.columns(2)
        rule = o1.selectbox("Rule", list(E.OUTLIER_RULES), key=wkey("adv_out_rule"))
        action = o2.radio("Action", ["cap", "remove"], horizontal=True, key=wkey("adv_out_action"),
                          format_func=lambda a: "Cap the values" if a == "cap" else "Remove the rows")
        if cols:
            st.plotly_chart(styled(px.box(df, y=cols[0], points="outliers", color_discrete_sequence=[BLUE]), 280),
                            key="adv_out_box")
        if st.button("Apply", key="adv_out_apply"):
            method, k = E.OUTLIER_RULES[rule]
            run(E.plan_outliers, cols, method, action, k)
    elif tool == "Duplicates":
        subset = st.multiselect("Compare only these columns (leave empty to compare all)", all_cols,
                                key=wkey("adv_dup_cols"))
        count = int(df.duplicated(subset=subset or None).sum())
        st.caption(f"{count:,} duplicate rows found.")
        if st.button("Remove duplicates", key="adv_dup_apply", disabled=count == 0):
            run(E.plan_drop_duplicates, subset or None)
    elif tool == "Drop columns":
        cols = st.multiselect("Columns to remove", all_cols, key=wkey("adv_drop_cols"))
        if st.button("Drop", key="adv_drop_apply"):
            run(E.plan_drop_columns, cols)


def page_clean() -> None:
    df = current()
    chk = clean_checks(df)
    v = f"{state.dataset_id}_{state.version}"  # fresh defaults whenever the data changes
    o = E.CleanOptions()

    st.title("Choose how to clean your data")
    st.caption("Steps that fix problems found in your data are already ticked. "
               "Untick anything you want to keep as it is, then apply.")
    with st.container(border=True):
        c1, c2 = st.columns([4, 1.3], vertical_alignment="center")
        c1.markdown("**Prefer not to choose?** Run every recommended step, add useful date and revenue "
                    "columns, and go straight to the results.")
        if c2.button("⚡ Do everything", type="primary", key="auto_clean"):
            do_everything()

    names = chk["names"]
    with st.container(border=True):
        o.column_names = st.checkbox("**Tidy column names**", value=bool(names), key=f"c_names_{v}")
        if names:
            old, new = next(iter(names.items()))
            st.caption(f"{len(names)} names become lowercase\\_with\\_underscores, "
                       f"e.g. '{esc_md(str(old).strip())}' → {esc_md(new)}.")
        else:
            st.caption("✓ Column names are already tidy.")

    spaces, placeholders = chk["messy"]
    with st.container(border=True):
        o.trim_text = st.checkbox("**Trim extra spaces and mark placeholders as missing**",
                                  value=bool(spaces or placeholders), key=f"c_trim_{v}")
        if spaces or placeholders:
            st.caption(f"{spaces:,} values have extra spaces and {placeholders:,} are placeholders "
                       "like 'N/A' or '-' that should count as missing.")
        else:
            st.caption("✓ Text values are already tidy.")

    with st.container(border=True):
        o.drop_empty = st.checkbox("**Drop mostly empty columns**", value=bool(E.sparse_columns(df, 60)),
                                   key=f"c_empty_{v}")
        o.empty_threshold = st.slider("Drop a column when more than this % of it is empty", 10, 95, 60, step=5,
                                      key=f"c_thr_{v}", disabled=not o.drop_empty)
        sparse = E.sparse_columns(df, o.empty_threshold)
        pct = df.isna().mean() * 100
        if sparse:
            st.caption("Would remove: " + ", ".join(f"{esc_md(c)} ({pct[c]:.0f}% empty)" for c in sparse) + ".")
        else:
            st.caption(f"✓ No column is more than {o.empty_threshold}% empty right now.")

    const = chk["constant"]
    with st.container(border=True):
        o.drop_constant = st.checkbox("**Drop columns that have the same value in every row**",
                                      value=bool(const), key=f"c_const_{v}")
        st.caption(("Would remove: " + ", ".join(esc_md(c) for c in const) + ".") if const else "✓ None found.")

    types = chk["types"]
    with st.container(border=True):
        o.fix_types = st.checkbox("**Fix data types**", value=bool(types), key=f"c_types_{v}")
        if types:
            st.caption("Text columns that really hold numbers or dates: "
                       + ", ".join(f"{esc_md(c)} → {k}" for c, k in types.items()) + ".")
            if "date" in types.values():
                fmt = st.selectbox("How are the dates written?", list(DATE_FORMATS), key=f"c_datefmt_{v}",
                                   disabled=not o.fix_types)
                o.dayfirst = DATE_FORMATS[fmt]
        else:
            st.caption("✓ Every column already has a sensible type.")

    variants = chk["variants"]
    with st.container(border=True):
        o.unify_spelling = st.checkbox("**Unify values written in different ways**", value=bool(variants),
                                       key=f"c_spell_{v}")
        if variants:
            lines = []
            for col, mapping in variants.items():
                example = ", ".join(f"'{esc_md(a)}' → '{esc_md(b)}'" for a, b in list(mapping.items())[:3])
                more = f" (+{len(mapping) - 3} more)" if len(mapping) > 3 else ""
                lines.append(f"{esc_md(col)}: {example}{more}")
            st.caption(" · ".join(lines))
        else:
            st.caption("✓ No spelling variants found.")

    dups = chk["duplicates"]
    with st.container(border=True):
        o.duplicates = st.checkbox("**Remove duplicate rows**", value=True, key=f"c_dups_{v}")
        found = f"{dups:,} exact duplicates found" if dups else "No exact duplicates right now"
        st.caption(f"{found}. Rows that become identical after the steps above are removed too.")

    with st.container(border=True):
        extreme = chk["outliers"]["Extreme only (3 × IQR)"]
        o.outliers = st.checkbox("**Handle outliers**", value=bool(extreme), key=f"c_out_{v}")
        c1, c2 = st.columns(2)
        o.outlier_rule = c1.selectbox("Rule", list(E.OUTLIER_RULES), key=f"c_orule_{v}", disabled=not o.outliers)
        action = c2.radio("What to do with them", ["Cap the values (keep the rows)", "Remove the rows"],
                          key=f"c_oact_{v}", disabled=not o.outliers)
        o.outlier_action = "cap" if action.startswith("Cap") else "remove"
        hits = chk["outliers"][o.outlier_rule]
        if hits:
            st.caption("Found: " + ", ".join(f"{esc_md(c)} ({n:,})" for c, n in hits.items())
                       + ". Number columns are checked again after their types are fixed.")
        else:
            st.caption("✓ No outliers with this rule right now. Number columns are checked again after "
                       "their types are fixed.")

    miss = chk["missing"]
    with st.container(border=True):
        o.missing = st.checkbox("**Fill or remove missing values**",
                                value=bool(any(miss.values()) or placeholders), key=f"c_miss_{v}")
        c1, c2, c3 = st.columns(3)
        label = CHOICE_LABELS.get
        o.missing_numbers = c1.selectbox("Number columns", E.MISSING_CHOICES["numbers"], format_func=label,
                                         key=f"c_mnum_{v}", disabled=not o.missing)
        o.missing_text = c2.selectbox("Text columns", E.MISSING_CHOICES["text"], format_func=label,
                                      key=f"c_mtxt_{v}", disabled=not o.missing)
        o.missing_dates = c3.selectbox("Date columns", E.MISSING_CHOICES["dates"], format_func=label,
                                       key=f"c_mdate_{v}", disabled=not o.missing)
        counts = {kind: int(df[cols].isna().sum().sum()) if cols else 0 for kind, cols in miss.items()}
        st.caption(f"Missing right now: {counts['numbers']:,} in number columns, {counts['text']:,} in text "
                   f"columns, {counts['dates']:,} in date columns. Placeholders become missing values first.")

    with st.expander("Advanced: fix individual columns yourself"):
        advanced_tools(df)

    c1, c2, c3 = st.columns([1, 1, 2])
    c1.button("← Back", key="clean_back", on_click=goto, args=(2,))
    c2.button("Skip cleaning", key="clean_skip", on_click=goto, args=(4,))
    if c3.button("Apply selected steps →", type="primary", key="clean_apply"):
        try:
            results = E.run_cleaning(df, o)
        except Exception as exc:
            st.error(f"Cleaning failed: {exc}")
            return
        apply_results(results, "**🧽 Cleaning done:**", next_page=4)


# ---------------------------------------------------------------------------
# Page 4: Transform
# ---------------------------------------------------------------------------

def transform_tool(df: pd.DataFrame, tool: str) -> None:
    all_cols, numbers, dates = list(df.columns), E.numeric_columns(df), E.date_columns(df)
    if tool == "Filter rows":
        f1, f2, f3 = st.columns([2, 2, 3])
        col = f1.selectbox("Column", all_cols, key=wkey("tf_f_col"))
        ordered = E.is_number(df[col]) or E.is_date(df[col])
        ops = E.FILTER_OPS if ordered else [o for o in E.FILTER_OPS if o not in
                                            ("is greater than", "is at least", "is less than", "is at most")]
        op = f2.selectbox("Condition", ops, key=wkey("tf_f_op"))
        value = None
        if op == "is one of":
            if E.is_number(df[col]):
                options = sorted(df[col].dropna().unique().tolist())[:200]
            else:
                options = df[col].dropna().astype(str).value_counts().index[:200].tolist()
            value = f3.multiselect("Values", options, key=wkey("tf_f_vals"))
        elif op not in ("is missing", "is not missing"):
            value = f3.text_input("Value", key=wkey("tf_f_val"), placeholder="e.g. 2025-06-30" if E.is_date(df[col]) else "")
        if st.button("Apply filter", type="primary", key="tf_f_apply"):
            run(E.plan_filter, col, op, value)
    elif tool == "Sort":
        cols = st.multiselect("Sort by", all_cols, key=wkey("tf_s_cols"))
        order = st.radio("Order", ["Smallest first", "Largest first"], horizontal=True, key=wkey("tf_s_dir"))
        if st.button("Sort", type="primary", key="tf_s_apply"):
            run(E.plan_sort, cols, order == "Smallest first")
    elif tool == "Calculated column":
        st.caption("Write a formula with column names, e.g. `quantity * unit_price` or "
                   "`price * (1 - discount / 100)`. Names with spaces need `backticks`.")
        n1, n2 = st.columns([1, 2])
        name = n1.text_input("New column name", key=wkey("tf_n_name"))
        expr = n2.text_input("Formula", key=wkey("tf_n_expr"))
        if numbers:
            st.caption("Number columns: " + ", ".join(f"`{c}`" for c in numbers))
        if st.button("Add column", type="primary", key="tf_n_apply"):
            run(E.plan_new_column, name, expr)
    elif tool == "Date parts":
        if not dates:
            st.caption("No date columns yet. Convert one in the Clean step (Fix data types).")
            return
        col = st.selectbox("Date column", dates, key=wkey("tf_d_col"))
        parts = st.multiselect("Parts", list(E.DATE_PARTS), default=["year", "month name"], key=wkey("tf_d_parts"))
        if st.button("Extract", type="primary", key="tf_d_apply"):
            run(E.plan_date_parts, col, parts)
    elif tool == "Group & summarize":
        st.caption("Turns the data into a summary table, e.g. total sales per region. You can undo it.")
        by = st.multiselect("Group by", friendly_categories(df) + dates, key=wkey("tf_g_by"))
        g1, g2 = st.columns(2)
        cols = g1.multiselect("Calculate on", [c for c in all_cols if c not in by], key=wkey("tf_g_cols"))
        funcs = g2.multiselect("Calculations", E.AGG_FUNCS, default=["sum"], key=wkey("tf_g_funcs"))
        if st.button("Summarize", type="primary", key="tf_g_apply"):
            text_ok = {"count", "nunique", "min", "max"}
            pairs = [(c, f) for c in cols for f in funcs if f in text_ok or E.is_number(df[c])]
            if cols and funcs and not pairs:
                st.error("Those calculations need number columns. Use count or nunique for text columns.")
            else:
                run(E.plan_group, by, pairs)
    elif tool == "Rename":
        r1, r2 = st.columns(2)
        old = r1.selectbox("Column", all_cols, key=wkey("tf_r_old"))
        new = r2.text_input("New name", key=wkey("tf_r_new"))
        if st.button("Rename", type="primary", key="tf_r_apply"):
            run(E.plan_rename, old, new)
    elif tool == "Drop columns":
        cols = st.multiselect("Columns to remove", all_cols, key=wkey("tf_x_cols"))
        if st.button("Drop", type="primary", key="tf_x_apply"):
            run(E.plan_drop_columns, cols)


def page_transform() -> None:
    df = current()
    st.title("Shape your data")
    st.caption("Optional. Pick a change, apply it, and repeat as often as you like. "
               "Every change can be undone from the sidebar.")
    try:
        revenue = E.plan_auto_revenue(df)
    except E.NothingToDo:
        revenue = None
    if revenue is not None:
        formula = revenue[1].split("= ", 1)[1].split(", so", 1)[0]
        with st.container(border=True):
            c1, c2 = st.columns([4, 1.3], vertical_alignment="center")
            c1.markdown(f"💰 **Suggestion:** add a **revenue** column ({esc_md(formula)}) for the finance view.")
            if c2.button("Add it", key="tf_revenue"):
                run(E.plan_auto_revenue)
    dates = [c for c in E.date_columns(df) if df[c].notna().any()][:2]
    missing_parts = [c for c in dates if any(f"{c}_{p}" not in df.columns for p in E.AUTO_DATE_PARTS)]
    if missing_parts:
        with st.container(border=True):
            c1, c2 = st.columns([4, 1.3], vertical_alignment="center")
            c1.markdown(f"💡 **Suggestion:** add year, month and weekday columns from "
                        f"{esc_md(', '.join(map(str, missing_parts)))} to see patterns over time.")
            if c2.button("Add them", key="tf_suggest"):
                run(E.plan_auto_date_parts)

    tools = ["Filter rows", "Sort", "Calculated column", "Date parts", "Group & summarize", "Rename", "Drop columns"]
    tool = st.radio("What do you want to do?", tools, horizontal=True, key="tf_tool")
    with st.container(border=True):
        transform_tool(df, tool)

    st.markdown(f"**Your data now** · {len(df):,} rows · {df.shape[1]} columns")
    st.dataframe(df.head(100), hide_index=True)

    c1, c2 = st.columns([1, 3])
    c1.button("← Back to cleaning", key="tf_back", on_click=goto, args=(3,))
    c2.button("See the results →", type="primary", key="tf_next", on_click=goto, args=(5,))


# ---------------------------------------------------------------------------
# Page 5: Results
# ---------------------------------------------------------------------------

def page_results() -> None:
    df, original = current(), state.original
    now = analysis(df)
    then = analysis(original, key=("original", state.dataset_id))
    st.title("Your results")
    st.caption(f"{esc_md(state.name)} · {len(steps())} preparation steps")

    n, t = now["overview"], then["overview"]

    def delta(key):
        diff = n[key] - t[key]
        return f"{diff:+,}" if diff else None

    m = st.columns(4)
    m[0].metric("Rows", f"{n['rows']:,}", delta("rows"), delta_color="off")
    m[1].metric("Columns", n["columns"], delta("columns"), delta_color="off")
    m[2].metric("Missing cells", f"{n['missing']:,}", delta("missing"), delta_color="inverse")
    m[3].metric("Duplicate rows", f"{n['duplicates']:,}", delta("duplicates"), delta_color="inverse")

    st.subheader("Data quality: before and after")
    if state.history:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**In the original file**")
            show_issues(then["findings"], "No issues found.")
        with c2:
            st.markdown("**Now**")
            show_issues(now["findings"], "✓ No issues left.")
    else:
        st.info("You haven't changed anything yet. Go back to clean the data, or press ⚡ Do everything.")
        show_issues(now["findings"], "No data quality issues found.")
        if st.button("⚡ Do everything", type="primary", key="auto_results"):
            do_everything()
    st.markdown("**💡 Patterns worth knowing**")
    show_patterns(now["findings"])

    if state.history:
        with st.expander(f"Steps applied ({len(steps())})"):
            for i, s in enumerate(steps(), 1):
                st.markdown(f"{i}. **{s.title}:** {esc_md(s.description)} *({s.changes})*")

    st.subheader("💰 Finance view")
    finance_view(df)

    st.subheader("Visual analysis")
    charts = cached("charts", data_key(), lambda: auto_charts(df), "Drawing charts...")
    for i in range(0, len(charts), 2):
        cols = st.columns(2)
        for j, (col, (title, fig)) in enumerate(zip(cols, charts[i:i + 2])):
            with col:
                st.markdown(f"**{esc_md(title)}**")
                st.plotly_chart(fig, key=f"res_chart_{i + j}")
    with st.expander("Build your own chart"):
        chart_builder(df)

    st.subheader("Download")
    stem = Path(state.name).stem
    report = cached("report", data_key(),
                    lambda: E.build_report(original, df, steps(), state.name, state.read_options),
                    "Building the EDA report...")
    script = E.build_script(steps(), state.name, state.read_options)
    d = st.columns(4)
    d[0].download_button("📄 EDA report (HTML)", report.encode("utf-8"), f"{stem}_eda_report.html",
                         "text/html", type="primary", key="dl_report")
    csv_bytes = cached("csv", data_key(), lambda: df.to_csv(index=False).encode("utf-8"))
    d[1].download_button("Clean data (CSV)", csv_bytes, f"{stem}_cleaned.csv", "text/csv", key="dl_csv")
    try:
        excel = cached("excel", data_key(), lambda: E.to_excel_bytes(df))
        d[2].download_button("Clean data (Excel)", excel, f"{stem}_cleaned.xlsx",
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_xlsx")
    except Exception as exc:
        d[2].caption(f"Excel export is not available: {exc}")
    d[3].download_button("Python script (.py)", script.encode("utf-8"), f"{stem}_pipeline.py",
                         "text/x-python", key="dl_py")

    with st.expander("Preview the EDA report"):
        components.html(report, height=900, scrolling=True)
    with st.expander("See the Python code"):
        st.caption("Put this script next to the original file and run it to get exactly the same result.")
        st.code(script, language="python")

    c1, c2 = st.columns([1, 3])
    c1.button("← Back to transform", key="res_back", on_click=goto, args=(4,))
    c2.button("Start over with a new file", key="res_new", on_click=goto, args=(1,))


# ---------------------------------------------------------------------------
# Layout: sidebar, step bar and the current page
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("PrepPilot")
    st.caption("Your co-pilot for data preparation: from a messy file to clean data and a full EDA report.")
    for i, page_name in enumerate(PAGES, 1):
        st.button(f"{i}. {page_name}", key=f"nav_{i}", type="primary" if i == state.page else "secondary",
                  disabled=i > 1 and not has_data(), on_click=goto, args=(i,))
    if has_data():
        st.divider()
        df_now = current()
        st.markdown(f"**{esc_md(state.name)}**  \n{len(df_now):,} rows · {df_now.shape[1]} columns")
        u1, u2 = st.columns(2)
        if u1.button("↩ Undo", key="undo", disabled=not state.history):
            undone = state.history.pop()[1]
            changed()
            flash(f"Undid: {undone.title}", "info")
            st.rerun()
        if u2.button("⟲ Reset", key="reset", disabled=not state.history):
            state.history = []
            changed()
            flash("Back to the original data.", "info")
            st.rerun()
        st.markdown("**Steps applied**")
        if state.history:
            st.markdown("\n".join(f"{i}. {s.title}" for i, s in enumerate(steps(), 1)))
        else:
            st.caption("None yet.")
    st.divider()
    st.caption(f"Built by [{AUTHOR}]({AUTHOR_LINK}) with Python, pandas and Streamlit.")

if state.page > 1 and not has_data():
    state.page = 1

stepper_items = []
for i, page_name in enumerate(PAGES, 1):
    css = "active" if i == state.page else ("done" if i < state.page else "")
    stepper_items.append(f'<div class="dps-step {css}"><span class="dot">{i}</span>'
                         f'<span class="label">{page_name}</span></div>')
st.markdown(f'<div class="dps-steps">{"".join(stepper_items)}</div>', unsafe_allow_html=True)

if state.pop("scroll_top", False):
    components.html(SCROLL_TOP, height=0)

if state.get("flash"):
    kind, message = state.pop("flash")
    getattr(st, kind)(message)

{1: page_upload, 2: page_explore, 3: page_clean, 4: page_transform, 5: page_results}[state.page]()
