# Current Situation Analysis

Total admissions: 545,848
Admissions with OMR data: 181,425 (33.2%)
Admissions WITHOUT OMR data: 364,423 (66.8%)

Of those with OMR:

- Height: only 16.3% of all admissions
- Weight: 30.9% of all admissions
- BMI: 30.6% of all admissions
- Blood Pressure: 5.9% of all admissions
  Problem: 2/3 of admissions have NO physiological data. OMR is outpatient-focused, not inpatient.

# Why This Happens (MIMIC Data Characteristic)

OMR (Outpatient Medical Records) = clinic visits, not hospitalizations
Lab data would be better for inpatient similarity
OMR data exists mostly for chronic disease management visits before/after admission

# Enhance OMR Strategy

Instead of just using OMR within admission window, also capture:

Current (admission window only) :
WHERE o.chartdate BETWEEN c.admittime AND c.dischtime

Enhanced (add recent history) :
WHERE o.chartdate BETWEEN c.admittime AND c.dischtime -- During admission
OR (o.chartdate < c.admittime
AND o.chartdate > c.admittime - INTERVAL '30 days') -- Last 30 days

# Next Steps

PHASE 1 (Current): cdss_base + diagnoses
├─ Similarity: Diagnoses (Jaccard) + Demographics (Euclidean)
├─ Coverage: 100% (all 545,848 admissions)
├─ Recommendation: "Patients with similar diagnoses got treatment X"
└─ Limitation: Generic, not personalized

PHASE 2 (Next): Add cdss_lab_features
├─ Similarity: Diagnoses + Demographics + Labs
├─ Coverage: ~80-85% (366,000-464,000 admissions)
├─ Recommendation: "Patients with similar diagnoses AND labs got treatment X"
└─ Advantage: Physiological severity captured

PHASE 3 (Optional): Add OMR imputation
├─ Similarity: Diagnoses + Demographics + Labs + OMR (imputed)
├─ Coverage: 100% (all 545,848 admissions)
├─ Recommendation: Most personalized
└─ Risk: Imputed OMR may introduce noise
