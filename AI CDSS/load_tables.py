import duckdb
import os

# Database
db = duckdb.connect("mimic.db")

# Folder path
folder = r"C:\Users\sancy\Sanidhya_VS\AI CDSS\mimic-iv-3.1\hosp"

important_tables = [
    "patients.csv.gz",
    "admissions.csv.gz",
    "omr.csv.gz",
    "diagnoses_icd.csv.gz",
    "d_icd_diagnoses.csv.gz",
    "prescriptions.csv.gz",
    "procedures_icd.csv.gz",
    "d_icd_procedures.csv.gz",
    "labevents.csv.gz",
    "d_labitems.csv.gz",
    "transfers.csv.gz",
]

for file in important_tables:

    table_name = file.replace(".csv.gz","")

    path = os.path.join(folder,file)

    print(f"Loading {table_name}...")

    db.execute(f"""
    
    CREATE OR REPLACE TABLE {table_name} AS
    
    SELECT *
    FROM read_csv_auto('{path}')
    
    """)

print("All tables loaded.")