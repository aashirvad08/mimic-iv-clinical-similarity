import duckdb
import time
import os

# --- Configuration ---
DB_PATH = r"c:\Users\sancy\Sanidhya_VS\mimic-iv-clinical-similarity\AI CDSS\mimic.db"

def run_diagnoses_pipeline():
    start_time = time.time()
    print(f"🚀 Starting cdss_diagnoses preprocessing...")

    # 1. Connect to DuckDB
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at {DB_PATH}")
        return
    
    con = duckdb.connect(database=DB_PATH, read_only=False)

    try:
        # 2. Check dependencies
        # Ensuring cdss_base exists as it defines our study cohort
        base_exists = con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'cdss_base'").fetchone()[0]
        if not base_exists:
            print("❌ Error: 'cdss_base' table not found. Please run the base pipeline first.")
            return

        print("🛠️  Building cdss_diagnoses table...")

        # 3. Core Transformation Logic
        # We use arg_max for performance and LIST for ML-ready features
        sql_query = """
        CREATE OR REPLACE TABLE cdss_diagnoses AS
        WITH diagnoses_enriched AS (
            -- Join with d_icd_diagnoses to get titles
            -- Join with cdss_base to ensure we only process relevant admissions
            SELECT 
                d.hadm_id,
                d.subject_id,
                d.seq_num,
                TRIM(d.icd_code) as icd_code,
                d.icd_version,
                dx.long_title,
                -- Version-aware 3-digit grouping to prevent version collisions
                d.icd_version || '_' || LEFT(TRIM(d.icd_code), 3) AS icd_3digit_grouped
            FROM diagnoses_icd d
            INNER JOIN cdss_base b ON d.hadm_id = b.hadm_id
            LEFT JOIN d_icd_diagnoses dx 
                ON TRIM(d.icd_code) = TRIM(dx.icd_code) 
                AND d.icd_version = dx.icd_version
        ),
        aggregation AS (
            SELECT 
                hadm_id,
                subject_id,
                
                -- PRIMARY DIAGNOSIS (Using arg_max: fastest way to get value at seq_num=1)
                arg_max(icd_code, -seq_num) AS primary_diagnosis_icd,
                arg_max(long_title, -seq_num) AS primary_diagnosis_title,
                arg_max(icd_3digit_grouped, -seq_num) AS primary_icd_3digit,
                
                -- COUNTS & COMPLEXITY
                COUNT(*) AS diagnosis_count,
                COUNT(DISTINCT icd_code) AS unique_icd_count,
                
                -- LISTS (Stored as native DuckDB/Python lists for Jaccard similarity)
                list_distinct(list(icd_code ORDER BY seq_num)) AS diagnoses_icd_list,
                list_distinct(list(icd_3digit_grouped)) AS diagnoses_3digit_list,
                
                -- METADATA
                CASE 
                    WHEN min(icd_version) = 9 AND max(icd_version) = 9 THEN 'ICD9_only'
                    WHEN min(icd_version) = 10 AND max(icd_version) = 10 THEN 'ICD10_only'
                    ELSE 'mixed'
                END AS icd_version_mix

            FROM diagnoses_enriched
            GROUP BY hadm_id, subject_id
        )
        SELECT 
            *,
            -- Research Metric: How 'diverse' is the diagnostic profile?
            unique_icd_count * 1.0 / NULLIF(diagnosis_count, 0) AS diagnosis_diversity_ratio
        FROM aggregation;
        """

        con.execute(sql_query)

        # 4. Validation & Metrics
        print("📊 Running Post-Build Validation...")
        stats = con.execute("""
            SELECT 
                COUNT(*) as total_rows, 
                AVG(diagnosis_count) as avg_dx,
                COUNT(*) FILTER (WHERE primary_diagnosis_icd IS NULL) as missing_primary
            FROM cdss_diagnoses
        """).fetchone()

        print(f"✅ Success! Table 'cdss_diagnoses' created.")
        print(f"   - Total Admissions: {stats[0]:,}")
        print(f"   - Avg Diagnoses per Admission: {stats[1]:.2f}")
        print(f"   - Missing Primary DX: {stats[2]}")

    except Exception as e:
        print(f"❌ Critical Error: {str(e)}")
    
    finally:
        con.close()
        elapsed = time.time() - start_time
        print(f"⏱️  Pipeline finished in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    run_diagnoses_pipeline()