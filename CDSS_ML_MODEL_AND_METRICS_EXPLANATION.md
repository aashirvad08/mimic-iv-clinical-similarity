## Simple Explanation

This system is a **similar-patient search engine** for clinical admissions.

Instead of training a model to directly predict an outcome like mortality, it does this:

1. Take a patient or admission as input.
2. Compare it to historical admissions using diagnoses, demographics, vitals, and treatments.
3. Find the most similar past cases.
4. Use those similar cases to show likely treatments and procedures.
5. Return both the matches and an explanation of why they matched.

So the ML idea is:

- not “predict from scratch”
- but “retrieve the closest historical cases”

That makes it a good fit for CDSS because clinicians can inspect the matched cases instead of trusting a black-box score.

---

## Technical Explanation

### 1. Model Type

This is a **non-parametric, similarity-based retrieval system**.

It uses three related paradigms:

- **Case-based reasoning**
- **Nearest-neighbor style retrieval**
- **Hybrid similarity scoring**

There are two main implementations across the system:

- **Original full-feature kNN path**
  - built a feature matrix
  - used `NearestNeighbors(metric="cosine")`
  - returned top-K admissions by cosine similarity

- **Optimized backend path**
  - uses DuckDB for candidate generation
  - uses Python for reranking
  - behaves like a retrieval engine rather than a fully fitted global model

Why this instead of traditional supervised ML:

- the primary goal is **finding comparable cases**, not just predicting a label
- it is more **interpretable**
- it works well with **sparse structured clinical features**
- it can return **real historical evidence**
- it avoids the complexity of training and maintaining deep models

### 2. Feature Engineering

The system uses different feature groups across different stages.

**Demographics**
- `age`
- `gender`
- `race`

Contribution:
- helps distinguish otherwise similar diagnosis profiles
- example: two patients with the same diagnosis list may differ clinically if one is much older

**Clinical measurements**
- `bmi`
- `systolic_bp`
- `diastolic_bp`

Contribution:
- adds physiological context
- helps separate mild vs severe presentations among patients with similar diagnoses

**Diagnosis features**
- `primary_diagnosis_icd`
- `primary_icd_3digit`
- `diagnosis_count`
- `unique_icd_count`
- `diagnosis_diversity_ratio`
- `diagnoses_icd_list`
- `diagnoses_3digit_list`

Contribution:
- `primary_diagnosis_icd`: strongest exact signal for reason for admission
- `primary_icd_3digit`: broader disease-family grouping
- diagnosis counts/diversity: disease burden and coding complexity
- full ICD lists: high-specificity overlap
- 3-digit lists: more robust, lower-sparsity semantic overlap

**Treatment features**
- `rx_drug_list`
- `proc_icd_list`
- `rx_unique_drugs`
- `proc_count`
- `surgery_count`
- `treatment_days`
- `treatment_complexity_score`
- `treatment_intensity_label`

Contribution:
- drug overlap captures medication-level similarity
- procedure overlap captures intervention-level similarity
- complexity and duration help model acuity and treatment burden

**Representation methods**
- numeric features: median imputation + scaling
- categorical features: one-hot encoding
- diagnosis/treatment lists: set-based overlap or multi-hot encoding
- precomputed sets:
  - `dx_set`
  - `dx_3_set`
  - `rx_set`
  - `proc_set`

### 3. Similarity Metrics

#### Jaccard Similarity

Used for:
- diagnosis lists
- diagnosis 3-digit groups
- drug lists
- procedure lists

Intuition:
- measures how much two sets overlap relative to everything present in either set

Example:
- A = `{CHF, Diabetes, CKD}`
- B = `{CHF, Diabetes, Pneumonia}`
- intersection = 2
- union = 4
- Jaccard = `2/4 = 0.5`

#### Cosine Similarity

Used in the original full-feature kNN implementation.

Intuition:
- measures how aligned two feature vectors are
- useful when features are high-dimensional and sparse

In the original system:
- features were encoded into a vector
- kNN used cosine distance
- final similarity was reported as `1 - distance`

#### Diagnosis Similarity

Diagnosis-only baseline combines:
- primary diagnosis exact match
- full ICD Jaccard
- 3-digit ICD Jaccard

Why:
- exact primary diagnosis is clinically strong
- full ICD overlap gives specificity
- 3-digit overlap adds robustness

#### Treatment Similarity

Treatment similarity combines:
- drug overlap
- procedure overlap
- treatment complexity similarity
- treatment duration similarity

Why:
- patients with similar diagnoses can still differ greatly in how they are treated
- treatment patterns capture acuity and intervention intensity

#### Hybrid Clinical Similarity

Clinical hybrid combines:
- diagnosis similarity
- treatment similarity

Why:
- diagnosis describes the condition
- treatment reflects how severe/managed the condition was

### 4. Scoring System

#### Diagnosis-only similarity

Weighted formula:

- primary diagnosis match: `0.40`
- full ICD Jaccard: `0.35`
- 3-digit ICD Jaccard: `0.25`

Reasoning:
- primary diagnosis matters most for admission-level similarity
- full ICD overlap is important but can be sparse
- 3-digit groups smooth over coding granularity

#### Treatment similarity

Weighted formula:

- drug similarity: `0.40`
- procedure similarity: `0.35`
- complexity similarity: `0.15`
- duration similarity: `0.10`

Reasoning:
- overlap in actual treatments is the strongest signal
- complexity and duration are supporting context, not primary anchors

#### Clinical hybrid similarity

Weighted formula:

- diagnosis similarity: `0.60`
- treatment similarity: `0.40`

Reasoning:
- diagnosis should dominate because it defines the case
- treatment adds realism and severity context

#### Fast backend reranking

Weighted formula:

- diagnosis-group similarity: `0.50`
- full diagnosis similarity: `0.25`
- primary diagnosis match: `0.15`
- context similarity: `0.10`

Reasoning:
- grouped diagnosis overlap is more stable and less sparse
- full-code overlap adds precision
- primary match is helpful but not enough alone
- context is supportive, not dominant

These weights are hand-chosen for:
- interpretability
- clinical common sense
- predictable behavior
- easy tuning

### 5. Retrieval Pipeline

**Input**
- existing `hadm_id`
- or custom JSON patient profile

**Preprocessing**
- normalize list columns from DuckDB to Python lists
- fill missing values
- derive fallback diagnosis groups if needed
- precompute set views for faster overlap operations

**Candidate selection**
- early versions: score broadly or use kNN over all loaded rows
- optimized backend: use DuckDB SQL to fetch a bounded candidate pool

**Scoring**
- compute diagnosis overlap
- compute treatment overlap if clinical mode is used
- compute demographic/physiology closeness
- combine with weighted formulas

**Ranking**
- sort by similarity descending
- keep top-K similar patients
- use a larger recommendation pool than the display pool

**Output**
- similar patients
- similarity score
- shared diagnoses
- recommended prescriptions
- recommended procedures
- explanation metadata

### 6. Model Evolution

#### Stage 1: Diagnosis-only baseline
Needed to build the simplest valid system first.

What it did:
- Jaccard-based diagnosis matching
- weighted diagnosis score
- top-K similar admissions

Why:
- fast to implement
- interpretable
- good baseline for validation

#### Stage 2: ML workflow validation
Needed to check whether diagnosis-only retrieval was useful.

What it did:
- joined `cdss_base`, `cdss_diagnoses`, and treatment summaries
- tested similarity-outcome alignment

Why:
- avoid overcommitting to a weak baseline

#### Stage 3: Treatment-aware similarity
Needed because diagnosis alone missed treatment intensity and procedures.

What it did:
- drug/procedure overlap
- complexity and duration features
- treatment-side retrieval

#### Stage 4: Hybrid clinical similarity
Needed to combine disease state and management pattern.

What it did:
- diagnosis similarity + treatment similarity

#### Stage 5: Optimized backend
Needed because the original full kNN path was too slow for the dashboard.

What it did:
- DuckDB candidate retrieval
- Python reranking
- larger candidate pool
- better treatment aggregation
- sub-2-second live API latency

### 7. Evaluation Insight

What was validated:
- list columns are correctly parsed
- critical columns are non-null
- count relationships are valid
- similarity scores are bounded and sorted correctly
- real DB smoke runs produce usable results

Important outcome:
- diagnosis-only similarity had weak mortality alignment

Observed quick result:
- correlation around `-0.004359`

Why diagnosis-only failed:
- mortality depends on more than diagnosis overlap
- severity, vitals, physiology, treatment intensity, and procedures matter
- same diagnoses can correspond to very different clinical trajectories

Why treatment + context improved results:
- treatments capture care intensity and intervention burden
- procedures capture escalation or specialty pathways
- vitals/demographics separate mild and severe presentations within the same diagnosis cluster

### 8. Recommendation Logic

Treatment recommendation does not just use the visible top 5 neighbors.

It uses a **larger recommendation pool** than the display pool.

Why:
- more stable aggregate suggestions
- makes procedures more likely to appear
- reduces noise from very small top-K sets

Two important treatment scores are used.

**Frequency score**
- how often a treatment appears among similar patients

**Weighted similarity aggregation**
- similar patients contribute more than weakly matched ones
- LOS is used as a post-hoc adjustment

This means a treatment rises when:
- it appears repeatedly in similar patients
- those patients are very similar
- it is not only supported by long-LOS outliers

### 9. Design Decisions

#### Why no deep learning?
- data is structured and sparse
- interpretability matters
- deployment should stay simple
- similar-case retrieval is the main goal

#### Why no embeddings?
- diagnosis and treatment codes are already discrete structured signals
- set overlap is easy to explain
- embeddings would add complexity without guaranteed benefit here

#### Why no BM25?
- BM25 is mainly for bag-of-words text retrieval
- this system operates over structured clinical tables, not documents

#### Why SQL + Python hybrid?
- SQL is very good at filtering, joining, counting overlaps, and narrowing candidates
- Python is better for custom composite scoring and explainability assembly
- combining both gives better speed and flexibility

---

## Formulas Section

### 1. Jaccard Similarity
\[
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
\]

Used for:
- diagnosis ICD overlap
- diagnosis 3-digit overlap
- drug overlap
- procedure overlap

---

### 2. Cosine Similarity
\[
\cos(x, y) = \frac{x \cdot y}{\|x\|\|y\|}
\]

Original kNN used cosine distance:
\[
d_{\text{cos}}(x, y) = 1 - \cos(x, y)
\]

Reported similarity:
\[
\text{similarity} = 1 - d_{\text{cos}} = \cos(x, y)
\]

---

### 3. Diagnosis Similarity
\[
\text{DiagSim} =
0.40 \cdot \text{PrimaryMatch} +
0.35 \cdot J(\text{ICD}_{full}) +
0.25 \cdot J(\text{ICD}_{3digit})
\]

Where:
- `PrimaryMatch = 1` if exact primary diagnosis matches, else `0`

---

### 4. Treatment Similarity
\[
\text{TreatSim} =
0.40 \cdot J(\text{Drugs}) +
0.35 \cdot J(\text{Procedures}) +
0.15 \cdot \text{ComplexitySim} +
0.10 \cdot \text{DurationSim}
\]

\[
\text{ComplexitySim} = 1 - |c_a - c_b|
\]

\[
\text{DurationSim} = 1 - \frac{|d_a - d_b|}{\max(d_a, d_b, 1)}
\]

---

### 5. Hybrid Clinical Similarity
\[
\text{ClinicalSim} =
0.60 \cdot \text{DiagSim} +
0.40 \cdot \text{TreatSim}
\]

---

### 6. Fast Backend Reranking Similarity
\[
\text{FastSim} =
0.50 \cdot J(\text{DiagGroups}) +
0.25 \cdot J(\text{FullDiag}) +
0.15 \cdot \text{PrimaryMatch} +
0.10 \cdot \text{ContextSim}
\]

---

### 7. Gaussian Context Similarity
For age/BMI/BP-like numeric fields:
\[
\text{GaussianSim}(x, y) = \exp\left(-\frac{(x-y)^2}{2\sigma^2}\right)
\]

Used inside context similarity.

---

### 8. Treatment Frequency Score
\[
\text{Score} = \text{Frequency} \cdot \frac{1}{\text{AvgLOS}}
\]

---

### 9. Weighted Treatment Score
\[
\text{WeightedScore} = \sum_i \frac{\text{Similarity}_i}{\text{LOS}_i}
\]

This gives more weight to:
- more similar neighbors
- lower-LOS supporting cases

---

## Pipeline Diagram

```text
                   +----------------------+
                   |  Query Patient/Input |
                   | hadm_id or JSON      |
                   +----------+-----------+
                              |
                              v
                   +----------------------+
                   | Preprocessing        |
                   | - normalize lists    |
                   | - fill nulls         |
                   | - derive groups      |
                   +----------+-----------+
                              |
                              v
              +----------------------------------+
              | Candidate Retrieval              |
              | DuckDB SQL over cdss tables      |
              | - primary match                  |
              | - diagnosis overlap counts       |
              | - age proximity                  |
              +----------------+-----------------+
                               |
                               v
              +----------------------------------+
              | Python Reranking                 |
              | - Jaccard diagnosis overlap      |
              | - treatment overlap (if used)    |
              | - context similarity             |
              | - weighted hybrid score          |
              +----------------+-----------------+
                               |
                               v
              +----------------------------------+
              | Top-K Similar Patients           |
              | - similarity                     |
              | - shared diagnoses               |
              | - explainability fields          |
              +----------------+-----------------+
                               |
                               v
              +----------------------------------+
              | Recommendation Pool              |
              | larger than display pool         |
              +----------------+-----------------+
                               |
                               v
              +----------------------------------+
              | Treatment Aggregation            |
              | - prescriptions                  |
              | - procedures                     |
              | - frequency                      |
              | - weighted_score                 |
              +----------------+-----------------+
                               |
                               v
              +----------------------------------+
              | Output                           |
              | - similar_patients               |
              | - recommended_treatments         |
              | - model_info                     |
              | - explanation                    |
              +----------------------------------+
```

---

## Key Takeaways (5 points)

1. This is a **case-based, similarity-driven retrieval system**, not a black-box predictive classifier.
2. The core signals are **diagnosis overlap, treatment overlap, and demographic/physiological closeness**.
3. The system evolved from **diagnosis-only Jaccard** to **hybrid clinical similarity** to a **fast DuckDB + Python retriever**.
4. The main formulas are **Jaccard similarity, cosine similarity, weighted hybrid scores, and LOS-adjusted treatment aggregation**.
5. The optimized backend is faster because it uses **SQL for candidate pruning** and **Python only for final reranking and explanation assembly**.

---

