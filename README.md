# Acute Kidney Injury Prediction

An XGBoost classifier that flags acute kidney injury (AKI) from a patient's creatinine blood-test history, packaged as a Docker container.

Coursework 1 of Software Engineering for Machine Learning Systems, MSc Computing (AI & ML), Imperial College London, January 2026.

## Results

On the course's held-out test set (7,402 patients, 21% with AKI):

| F3 | Recall | Precision |
|---|---|---|
| 0.990 | 0.992 | 0.976 |

It missed 13 of 1,564 AKI cases and raised 38 false alarms. The target metric is F3, which weights recall above precision because a missed AKI costs more than a false alarm. Cross-validated F3 on the training set: 0.993.

## How it works

**Features.** Each patient's creatinine history becomes seven features:

- age and sex
- the latest result
- the minimum and mean of results in the 7 days before it
- the median and maximum of results 8 to 365 days before it (the baseline)

These windows mirror the NHS England AKI algorithm, which compares the latest result with the lowest value in the past 7 days and the median of the past 8 to 365 days. Empty windows stay NaN, which XGBoost handles natively.

**Model.** XGBoost with 100 trees of depth 4 and `scale_pos_weight=3` for the class imbalance.

**Threshold.** Five-fold stratified cross-validation picks, in each fold, the probability threshold that maximises F3. The final model is retrained on all the training data and uses the mean of the five thresholds (0.22).

## Usage

The container trains on `/data/training.csv` at start-up and writes one `y`/`n` prediction per input row to `/data/aki.csv`:

```bash
docker build -t aki-model .
docker run --rm -v "$(pwd)/data:/data" aki-model
```

Without Docker (Python 3.12):

```bash
pip install -r requirements.txt
python model.py --train data/training.csv --input data/test.csv --output data/aki.csv
```

The course data is not included. Each CSV has one row per patient: `age`, `sex`, `aki` (`y`/`n`, training only), then `creatinine_date_0`, `creatinine_result_0`, `creatinine_date_1`, and so on.

## What came next

This was the first stage of a module-long project. Later coursework built a real-time alerting service for a simulated hospital: it reads HL7 blood-test messages over MLLP, keeps patient history in PostgreSQL, pages the clinical team over HTTP and runs on Kubernetes (Azure). That code is in Imperial's private GitLab.
