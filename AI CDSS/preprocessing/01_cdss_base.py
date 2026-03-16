import pandas as pd
import duckdb
import os
from datetime import datetime

############################################
# PATH CONFIG
############################################

BASE_PATH = r"C:\Users\sancy\Sanidhya_VS\AI CDSS\mimic-iv-3.1\hosp"

PATIENTS_FILE = os.path.join(BASE_PATH, "patients.csv.gz")
ADMISSIONS_FILE = os.path.join(BASE_PATH, "admissions.csv.gz")
OMR_FILE = os.path.join(BASE_PATH, "omr.csv.gz")

DB_PATH = r"C:\Users\sancy\Sanidhya_VS\AI CDSS\mimic.db"

############################################
# LOGGING SETUP
############################################

log_file = open("preprocessing_log.txt", "w")

def log_message(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_text = f"[{timestamp}] {msg}"
    print(log_text)
    log_file.write(log_text + "\n")
    log_file.flush()

log_message("Starting CDSS Base Table Pipeline (DuckDB-optimized)")

############################################
# INITIALIZE DATABASE CONNECTION
############################################

log_message("Initializing DuckDB...")

con = duckdb.connect(DB_PATH)

############################################
# LOAD RAW TABLES INTO DUCKDB
############################################

log_message("Loading raw tables into DuckDB...")

# Load patients
con.execute(f"""
    CREATE OR REPLACE TEMP TABLE raw_patients AS
    SELECT * FROM read_csv_auto('{PATIENTS_FILE}')
""")
patients_count = con.execute("SELECT COUNT(*) FROM raw_patients").fetchall()[0][0]
log_message(f"  raw_patients: {patients_count} records")

# Load admissions
con.execute(f"""
    CREATE OR REPLACE TEMP TABLE raw_admissions AS
    SELECT * FROM read_csv_auto('{ADMISSIONS_FILE}')
""")
admissions_count = con.execute("SELECT COUNT(*) FROM raw_admissions").fetchall()[0][0]
log_message(f"  raw_admissions: {admissions_count} records")

# Load OMR
con.execute(f"""
    CREATE OR REPLACE TEMP TABLE raw_omr AS
    SELECT * FROM read_csv_auto('{OMR_FILE}')
""")
omr_count = con.execute("SELECT COUNT(*) FROM raw_omr").fetchall()[0][0]
log_message(f"  raw_omr: {omr_count} records")

############################################
# BUILD COHORT: ADULT ADMISSIONS
############################################

log_message("Building adult cohort (anchor_age >= 18)...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE cohort AS
    SELECT
        a.subject_id,
        a.hadm_id,
        a.admittime,
        a.dischtime,
        a.admission_type,
        a.admission_location,
        a.discharge_location,
        a.race,
        a.insurance,
        a.language,
        a.marital_status,
        a.hospital_expire_flag,
        p.gender,
        p.anchor_age,
        (EXTRACT(EPOCH FROM (a.dischtime - a.admittime)) / 86400) AS los_days
    FROM raw_admissions a
    INNER JOIN raw_patients p ON a.subject_id = p.subject_id
    WHERE p.anchor_age >= 18
      AND a.dischtime IS NOT NULL
      AND a.admittime < a.dischtime
""")

cohort_count = con.execute("SELECT COUNT(*) FROM cohort").fetchall()[0][0]
log_message(f"  Cohort size: {cohort_count} admissions")

############################################
# OMR PROCESSING WITH WINDOW FUNCTIONS
############################################

log_message("Processing OMR with window functions...")

# Step 1: Filter OMR for relevant measurements and within admission window
con.execute("""
    CREATE OR REPLACE TEMP TABLE omr_filtered AS
    SELECT
        o.subject_id,
        o.chartdate,
        o.seq_num,
        o.result_name,
        o.result_value,
        c.hadm_id,
        c.admittime,
        c.dischtime
    FROM raw_omr o
    INNER JOIN cohort c ON o.subject_id = c.subject_id
    WHERE o.result_name IN ('Height (Inches)', 'Weight (Lbs)', 'BMI (kg/m2)', 'Blood Pressure')
      AND o.chartdate >= CAST(c.admittime AS DATE) 
      AND o.chartdate <= CAST(c.dischtime AS DATE)
""")

omr_filtered_count = con.execute("SELECT COUNT(*) FROM omr_filtered").fetchall()[0][0]
log_message(f"  OMR records within admission windows: {omr_filtered_count}")

admissions_with_omr = con.execute("""
    SELECT COUNT(DISTINCT hadm_id) FROM omr_filtered
""").fetchall()[0][0]
log_message(f"  Admissions with OMR data: {admissions_with_omr}/{cohort_count} ({100*admissions_with_omr/cohort_count:.1f}%)")

# Step 2: Rank by latest date, then seq_num
con.execute("""
    CREATE OR REPLACE TEMP TABLE omr_ranked AS
    SELECT
        subject_id,
        hadm_id,
        chartdate,
        seq_num,
        result_name,
        result_value,
        ROW_NUMBER() OVER (
            PARTITION BY hadm_id, result_name
            ORDER BY chartdate DESC, seq_num ASC
        ) AS rn
    FROM omr_filtered
""")

# Step 3: Keep only rank 1 (latest measurement per measurement type per admission)
con.execute("""
    CREATE OR REPLACE TEMP TABLE omr_latest AS
    SELECT
        subject_id,
        hadm_id,
        result_name,
        result_value
    FROM omr_ranked
    WHERE rn = 1
""")

omr_latest_count = con.execute("SELECT COUNT(*) FROM omr_latest").fetchall()[0][0]
log_message(f"  OMR latest records (after deduplication): {omr_latest_count}")

# Step 4: Pivot OMR to wide format
con.execute("""
    CREATE OR REPLACE TEMP TABLE omr_pivot AS
    SELECT
        subject_id,
        hadm_id,
        MAX(CASE WHEN result_name = 'Height (Inches)' THEN TRY_CAST(result_value AS DOUBLE) ELSE NULL END) AS height_inches,
        MAX(CASE WHEN result_name = 'Weight (Lbs)' THEN TRY_CAST(result_value AS DOUBLE) ELSE NULL END) AS weight_lbs,
        MAX(CASE WHEN result_name = 'BMI (kg/m2)' THEN TRY_CAST(result_value AS DOUBLE) ELSE NULL END) AS bmi,
        MAX(CASE WHEN result_name = 'Blood Pressure' THEN result_value ELSE NULL END) AS blood_pressure
    FROM omr_latest
    GROUP BY subject_id, hadm_id
""")

log_message("  OMR pivot complete")

############################################
# PARSE BLOOD PRESSURE (SYSTOLIC/DIASTOLIC)
############################################

log_message("Parsing blood pressure...")

con.execute("""
    CREATE OR REPLACE TEMP TABLE omr_with_bp_split AS
    SELECT
        subject_id,
        hadm_id,
        height_inches,
        weight_lbs,
        bmi,
        blood_pressure,
        TRY_CAST(SPLIT_PART(blood_pressure, '/', 1) AS DOUBLE) AS systolic_bp,
        TRY_CAST(SPLIT_PART(blood_pressure, '/', 2) AS DOUBLE) AS diastolic_bp
    FROM omr_pivot
""")

log_message("  Blood pressure split complete")

############################################
# MERGE COHORT WITH OMR FEATURES
############################################

log_message("Merging cohort with OMR features...")

con.execute("""
    CREATE OR REPLACE TABLE cdss_base AS
    SELECT
        c.subject_id,
        c.hadm_id,
        c.anchor_age AS age,
        c.gender,
        c.race,
        c.marital_status,
        c.admission_type,
        c.admission_location,
        c.discharge_location,
        c.insurance,
        c.language,
        c.los_days,
        c.hospital_expire_flag AS mortality,
        COALESCE(o.height_inches, NULL) AS height_inches,
        COALESCE(o.weight_lbs, NULL) AS weight_lbs,
        COALESCE(o.bmi, NULL) AS bmi,
        COALESCE(o.systolic_bp, NULL) AS systolic_bp,
        COALESCE(o.diastolic_bp, NULL) AS diastolic_bp
    FROM cohort c
    LEFT JOIN omr_with_bp_split o ON c.subject_id = o.subject_id AND c.hadm_id = o.hadm_id
    ORDER BY c.hadm_id
""")

final_count = con.execute("SELECT COUNT(*) FROM cdss_base").fetchall()[0][0]
log_message(f"  Final cdss_base: {final_count} records")

############################################
# DATA VALIDATION & LOGGING
############################################

log_message("Running validation checks...")

log_message(f"Total rows in cdss_base: {final_count}")

unique_patients = con.execute("SELECT COUNT(DISTINCT subject_id) FROM cdss_base").fetchall()[0][0]
log_message(f"Unique patients: {unique_patients}")

unique_admissions = con.execute("SELECT COUNT(DISTINCT hadm_id) FROM cdss_base").fetchall()[0][0]
log_message(f"Unique admissions: {unique_admissions}")

# Mortality distribution
log_message("Mortality distribution:")
mortality_dist = con.execute("""
    SELECT mortality, COUNT(*) as count, 
           ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as pct
    FROM cdss_base
    GROUP BY mortality
    ORDER BY mortality
""").fetchall()

for mortality, count, pct in mortality_dist:
    log_message(f"  {mortality}: {count} ({pct}%)")

# LOS statistics
log_message("LOS statistics (days):")
los_stats = con.execute("""
    SELECT 
        MIN(los_days) as min_los,
        MAX(los_days) as max_los,
        AVG(los_days) as mean_los,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY los_days) as median_los,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY los_days) as q1_los,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY los_days) as q3_los
    FROM cdss_base
""").fetchall()[0]

log_message(f"  Min: {los_stats[0]:.2f}")
log_message(f"  Max: {los_stats[1]:.2f}")
log_message(f"  Mean: {los_stats[2]:.2f}")
log_message(f"  Median: {los_stats[3]:.2f}")
log_message(f"  Q1: {los_stats[4]:.2f}")
log_message(f"  Q3: {los_stats[5]:.2f}")

# Missing values by column
log_message("Missing value counts:")
missing_stats = con.execute("""
    SELECT 
        'height_inches' as column_name,
        COUNT(*) FILTER (WHERE height_inches IS NULL) as null_count,
        ROUND(100.0 * COUNT(*) FILTER (WHERE height_inches IS NULL) / COUNT(*), 1) as pct_missing
    FROM cdss_base
    UNION ALL
    SELECT 'weight_lbs', COUNT(*) FILTER (WHERE weight_lbs IS NULL), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE weight_lbs IS NULL) / COUNT(*), 1)
    FROM cdss_base
    UNION ALL
    SELECT 'bmi', COUNT(*) FILTER (WHERE bmi IS NULL), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE bmi IS NULL) / COUNT(*), 1)
    FROM cdss_base
    UNION ALL
    SELECT 'systolic_bp', COUNT(*) FILTER (WHERE systolic_bp IS NULL), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE systolic_bp IS NULL) / COUNT(*), 1)
    FROM cdss_base
    UNION ALL
    SELECT 'diastolic_bp', COUNT(*) FILTER (WHERE diastolic_bp IS NULL), 
           ROUND(100.0 * COUNT(*) FILTER (WHERE diastolic_bp IS NULL) / COUNT(*), 1)
    FROM cdss_base
""").fetchall()

for col_name, null_count, pct_missing in missing_stats:
    log_message(f"  {col_name}: {null_count} ({pct_missing}%)")

# Sample of data
log_message("\nSample of cdss_base (first 5 rows):")
sample = con.execute("SELECT * FROM cdss_base LIMIT 5").fetchall()
columns = [desc[0] for desc in con.description]
log_message(f"Columns: {columns}")
for i, row in enumerate(sample):
    log_message(f"  Row {i+1}: {dict(zip(columns, row))}")

############################################
# SCHEMA INSPECTION
############################################

log_message("\nDatabase schema:")
schema = con.execute("DESCRIBE cdss_base").fetchall()
for col_name, col_type, null_allowed, default, extra in schema:
    log_message(f"  {col_name}: {col_type}")

############################################
# CLOSE CONNECTION AND FINALIZE
############################################

con.close()

log_message("\n" + "="*60)
log_message("Pipeline completed successfully!")
log_message(f"Output: DuckDB table 'cdss_base' in {DB_PATH}")
log_message("="*60)

log_file.close()