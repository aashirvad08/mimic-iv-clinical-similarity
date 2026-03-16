import os
import shutil
import duckdb
import pandas as pd
import webbrowser
from datetime import datetime

HOSP_PATH = "mimic-iv-3.1/hosp"
OUT_DIR = "previews"

important_tables = [
    #"patients.csv.gz",
    #"admissions.csv.gz",
    #"omr.csv.gz",
    #"diagnoses_icd.csv.gz",
    #"d_icd_diagnoses.csv.gz",
    #"prescriptions.csv.gz",
    #"procedures_icd.csv.gz",
    #"d_icd_procedures.csv.gz",
    #"labevents.csv.gz",
    #"d_labitems.csv.gz",
    #"transfers.csv.gz",
]

con = duckdb.connect()
con.execute("SET preserve_insertion_order=true")
con.execute("SET pandas_analyze_sample=100000")

def terminal_width(default=120) -> int:
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except Exception:
        return default

def print_schema(file_path: str) -> pd.DataFrame:
    cols = con.execute(f"DESCRIBE SELECT * FROM '{file_path}'").df()
    print("\nCOLUMNS:")
    print(cols.to_string(index=False))
    return cols

def id_columns(columns_df: pd.DataFrame) -> list[str]:
    return [c for c in columns_df["column_name"].tolist() if c.lower().endswith("id")]

def print_quick_info(file_path: str):
    rows = con.execute(f"SELECT COUNT(*) FROM '{file_path}'").fetchone()[0]
    print("\nQUICK INFO:")
    print("Total rows:", rows)

def sample_df(file_path: str, limit: int = 25) -> pd.DataFrame:
    return con.execute(f"SELECT * FROM '{file_path}' LIMIT {limit}").df()

def df_to_interactive_html(df: pd.DataFrame, title: str) -> str:
    # Escape-less HTML from pandas, plus a little CSS/JS for usability
    table_html = df.to_html(index=False, border=0)

    css = """
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 16px; }
    h1 { font-size: 18px; margin: 0 0 8px 0; }
    .meta { color: #555; margin-bottom: 12px; font-size: 12px; }
    .wrap { border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }
    .table-container { overflow: auto; max-height: 75vh; }
    table { border-collapse: collapse; width: max-content; min-width: 100%; }
    thead th {
      position: sticky; top: 0; z-index: 2;
      background: #f6f8fa; border-bottom: 1px solid #ddd;
      font-weight: 600;
    }
    th, td { padding: 8px 10px; border-bottom: 1px solid #eee; white-space: nowrap; }
    tr:hover td { background: #fffbe6; }
    .hint { margin-top: 10px; font-size: 12px; color: #555; }
    code { background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }
    """

    js = """
    // Click-to-copy cell contents (lightweight)
    document.addEventListener('click', async (e) => {
      const td = e.target.closest('td');
      if (!td) return;
      const text = td.innerText;
      try {
        await navigator.clipboard.writeText(text);
        td.style.outline = "2px solid #2da44e";
        setTimeout(() => td.style.outline = "", 350);
      } catch (_) {}
    });
    """

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>{css}</style>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta">Generated: {now} • Rows shown: {len(df)} • Columns: {len(df.columns)}</div>
  <div class="wrap">
    <div class="table-container">
      {table_html}
    </div>
  </div>
  <div class="hint">Tip: horizontal scroll for wide tables. Click any cell to copy its value.</div>
  <script>{js}</script>
</body>
</html>
"""

def write_html_preview(df: pd.DataFrame, out_path: str, title: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    html = df_to_interactive_html(df, title=title)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

def open_in_browser(path: str):
    # Use file:// URL so it works consistently
    abspath = os.path.abspath(path)
    url = "file:///" + abspath.replace("\\", "/")
    webbrowser.open(url, new=2)

def preview_table(file_path: str, sample_limit: int = 25, open_html: bool = True):
    base = os.path.basename(file_path)
    print("\n" + "=" * 80)
    print(f"TABLE: {base}")
    print("=" * 80)

    try:
        cols = print_schema(file_path)

        df = sample_df(file_path, limit=sample_limit)

        # Keep terminal output minimal & readable:
        print("\nIDENTIFIER COLUMNS:")
        print(id_columns(cols))

        print_quick_info(file_path)

        # Interactive HTML preview:
        out_file = os.path.join(OUT_DIR, base.replace(".csv.gz", "") + ".sample.html")
        write_html_preview(df, out_file, title=f"MIMIC-IV preview: {base}")

        print("\nHTML PREVIEW:")
        print(f"Wrote: {out_file}")

        if open_html:
            open_in_browser(out_file)

    except Exception as e:
        print("Error:", e)

def top_labs_preview(limit: int = 50, open_html: bool = True):

    labevents_path = os.path.join(HOSP_PATH, "labevents.csv.gz")
    labdict_path = os.path.join(HOSP_PATH, "d_labitems.csv.gz")

    print("\n" + "=" * 80)
    print("TOP LAB TESTS (MOST FREQUENTLY ORDERED)")
    print("=" * 80)

    query = f"""
    WITH lab_counts AS (

        SELECT
            itemid,
            COUNT(*) AS total_events,
            COUNT(DISTINCT subject_id) AS patient_count,
            COUNT(valuenum) AS numeric_results

        FROM '{labevents_path}'

        WHERE itemid IS NOT NULL AND valuenum IS NOT NULL

        GROUP BY itemid

        ORDER BY patient_count DESC

        LIMIT {limit}

    )

    SELECT

        lc.itemid,
        dl.label,
        dl.fluid,
        dl.category,
        lc.patient_count,
        lc.total_events,
        lc.numeric_results

    FROM lab_counts lc

    JOIN '{labdict_path}' dl
    ON lc.itemid = dl.itemid

    ORDER BY lc.patient_count DESC
    """

    df = con.execute(query).df()

    print("\nTOP LABS:")
    print(df.to_string(index=False))

    out_file = os.path.join(OUT_DIR, "top_50_labs.html")

    write_html_preview(
        df,
        out_file,
        title="Top 50 Most Frequent Lab Tests"
    )

    print("\nHTML PREVIEW:")
    print(f"Wrote: {out_file}")

    if open_html:
        open_in_browser(out_file)

if __name__ == "__main__":
    # Set False if you don't want it to auto-open a browser tab each time.
    AUTO_OPEN = True

    for table in important_tables:
        full_path = os.path.join(HOSP_PATH, table)
        preview_table(full_path, sample_limit=25, open_html=AUTO_OPEN)

    top_labs_preview(limit=50, open_html=AUTO_OPEN)