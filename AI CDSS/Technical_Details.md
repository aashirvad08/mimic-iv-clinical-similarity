# Gives personal details :

patients.csv.gz -> subject_id gender anchor_age anchor_year anchor_year_group

admissions.csv.gz -> subject_id hadm_id admittime dischtime admission_type admission_location discharge_location marital_status race

omr.csv.gz -> subject_id chartdate seq_num result_name(height, weight, blood pressure, BMI) result_value

# Gives idea about transfers (if eventtype = is first admit then ED then it's a serious case):

transfers.csv.gz -> subject_id hadm_id transfer_id eventtype intime outtime

# Gives Diagnoses and procedures for that diagnoses :

diagnoses_icd.csv.gz -> subject_id hadm_id seq_num icd_code icd_version

d_icd_diagnoses.csv.gz -> icd_code icd_version long_title

procedures_icd.csv.gz -> subject_id hadm_id seq_num chartdate icd_code icd_version

d_icd_procedures.csv.gz -> icd_code icd_version long_title

# Gives idea of how severe a dieasea actually is :

labevents.csv.gz -> labevent_id subject_id hadm_id(most are <NA>) specimen_id itemid charttime storetime value valuenum valueuom ref_range_lower ref_range_upper flag priority comments

d_labitems.csv.gz -> itemid label fluid category

# Gives Prescription Ideas :

prescriptions.csv.gz -> subject_id hadm_id poe_id poe_seq starttime stoptime drug_type drug formulary_drug_cd prod_strength form_rx dose_val_rx dose_unit_rx form_val_disp form_unit_disp doses_per_24_hrs route

# Further Optimization Idea

We have Total admissions: 545,848
but Admissions with OMR data: 181,425 (33.2%)

Of those with OMR:

- Height: only 16.3% of all admissions
- Weight: 30.9% of all admissions
- BMI: 30.6% of all admissions
- Blood Pressure: 5.9% of all admissions

so basically we don't know height and blood pressure of many since we use this condition for linking omr with patients+admissions :
For each (subject_id, hadm_id):

1. Find all OMR records where:
   - subject_id matches
   - chartdate BETWEEN admittime AND dischtime
   - result_name IN ('Height', 'Weight', 'BMI (kg/m2)', 'Blood Pressure')

2. Get the max(chartdate) → latest_date

3. Filter to:
   - chartdate = latest_date
   - seq_num = 1

4. Pivot result_name → columns (Height, Weight, BMI, BloodPressure)

5. If no match found → NULL

but if the above case fails we can apply and lenient cases for patients with no OMR data
we Find all OMR records where:

- subject_id matches
- year in chartdate is same to year in admittime or dischtime (format for admittime & dischtime = 2180-05-06 22:23:00 whereas format for chartdate = 2180-04-27)

then we can implement rules 2, 3, 4 & 5 as stated above and replace rule 1 with this one instead

OTHERWISE

# Tier 1: If both patients have OMR data

similarity = compare(patient_a_omr, patient_b_omr)

# Tier 2: If one/both missing OMR, use diagnoses + demographics

similarity = compare(patient_a_diagnoses, patient_b_diagnoses) + \
 compare(patient_a_age, patient_b_age) + \
 compare(patient_a_gender, patient_b_gender)

# Tier 3: Demographics only (last resort)

similarity = compare(patient_a_age, patient_b_age)
