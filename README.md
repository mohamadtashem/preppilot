<img src="assets/logo.svg" alt="PrepPilot logo" width="88">

# PrepPilot

**▶ Try it live: [preppilot-app.streamlit.app](https://preppilot-app.streamlit.app/)**

**Your co-pilot for data preparation. Upload a messy dataset, explore it, clean it, transform it, and get clean data, a finance view, a full EDA report with charts, and the Python code that reproduces every step.**

Data preparation is one of the most time-consuming parts of any analysis. PrepPilot turns the usual notebook routine (fix types, handle missing values, remove duplicates, check outliers, explore distributions) into a guided five-step workflow, without hiding the code.

## How it works

| Step | What happens |
|---|---|
| **1. Upload** | CSV, TSV or TXT (the separator and European number format are detected automatically), Excel (choose the sheet), JSON or JSON Lines (nested fields are flattened) |
| **2. Explore** | An automatic exploratory data analysis: overview, column profile, statistics, charts, and a data quality check that finds missing values, duplicates, numbers stored as text, spelling variants ("North" vs "north "), placeholders like "N/A", extreme outliers and more |
| **3. Clean** | A checklist of cleaning steps. The ones that fix problems in your data are already ticked; untick or adjust anything. Advanced tools let you fix individual columns |
| **4. Transform** | Filter rows, sort, add calculated columns, extract date parts, group and summarize, rename or drop columns |
| **5. Results** | Before-and-after data quality, a finance view, automatic interactive charts, a chart builder, and downloads: EDA report (HTML), clean data (CSV or Excel) and the Python script |

**⚡ Do everything:** one button runs every recommended cleaning step, adds a revenue column when the data has quantity and price, adds year, month and weekday columns for dates, and goes straight to the results.

### 💰 Finance view

Pick a money column (it suggests one: revenue, sales, amount...), a date and a breakdown such as product or region. You get:

- Total, average and median per row, best month and the latest month's change
- Monthly totals with month-over-month growth, and the same months compared with the previous year
- The share of each product, region or customer, and how many of them make up 80% of the total
- Plain-language insights, including warnings such as an incomplete last month or refunds (negative amounts)

The currency symbol is detected from the original file.

### What makes it different

- **Every step becomes real pandas code.** The app runs the same code it exports, so the downloaded script reproduces the cleaned dataset exactly. Nothing is a black box.
- **Undo and reset** at any point. The sidebar shows the full history of steps.
- **Before-vs-after report.** The EDA report shows which data quality issues the original file had and which are left, plus statistics, charts and the code.

## Try it

A messy sample dataset (`sample_data/messy_sales.csv`, 498 sales orders) is included. Click **Use sample data** on the first page. It has duplicates, mixed date formats, prices stored as text like "$1,200", placeholder values, inconsistent spelling and outliers. `examples/sample_eda_report.html` shows the report it produces.

## Run it on your computer

```bash
git clone https://github.com/mohamadtashem/preppilot.git
cd preppilot
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at http://localhost:8501.

## Put it online for free

1. Push this folder to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click **Create app**.
3. Choose the repository, branch `main`, and main file `app.py`, then click **Deploy**.

You get a public link you can share.

## Project structure

```
preppilot/
├── app.py                  # Streamlit interface: the five steps, finance view, charts and downloads
├── engine.py               # Core logic: loading, checks, cleaning, transforms, EDA, finance, report, code export
├── sample_data/
│   └── messy_sales.csv     # Demo dataset with realistic data problems
├── examples/
│   └── sample_eda_report.html
├── requirements.txt
└── .streamlit/config.toml
```

`engine.py` has no Streamlit code, so it can be reused in a notebook or another app:

```python
import engine as E

raw = open("data.csv", "rb").read()
options = E.read_options("data.csv", raw)
df = E.load_file("data.csv", raw, options)

results = E.run_everything(df)  # or E.run_cleaning(df, E.CleanOptions(outliers=False))
clean = results[-1][0] if results else df
steps = [step for _, step in results]

for kind, text in E.findings(clean):
    print(kind, text)
print(E.finance_summary(clean, *E.finance_defaults(clean))["insights"])
open("report.html", "w", encoding="utf-8").write(E.build_report(df, clean, steps, "data.csv", options))
open("pipeline.py", "w", encoding="utf-8").write(E.build_script(steps, "data.csv", options))
```

## Tech stack

Python · pandas · NumPy · Streamlit · Plotly · Matplotlib

## Author

Built by **Mohamad Al Tashem** · [LinkedIn](https://www.linkedin.com/in/mohamad14)
