# Initial 25% laboratory demonstration

> Historical guide for the original development stage. Training, evaluation and inference have since been implemented. For current status and commands, use FINAL_PROJECT_GUIDE.md and RESULTS.md; statements below describe the earlier version.

## Before the lab

Open PowerShell in `C:\imda_sem5\self-driving-behavioral-cloning`. The base environment and actual 300-image subset are already present. These commands do not require activation or an execution-policy change:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.explore_data --data-dir data/raw
.\.venv\Scripts\python.exe -m src.prepare_data --data-dir data/raw
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If more than one driving log has been added, use `--csv "data/raw/lab_subset/driving_log.csv"` for both data commands and choose that same log in Streamlit. Keep a copy of generated figures available for an offline presentation. The running dashboard uses local data and does not need Kaggle access.

## Five-minute demonstration

1. **Explain the task.** A human demonstration supplies a road image and its steering command. The future CNN will learn a supervised regression mapping. This is an educational simulator prototype, not a complete self-driving car.
2. **Identify the real dataset.** Show `data/raw/lab_subset/provenance.json` and the headerless `driving_log.csv`. Explain the seven columns and why only center/steering are used. The first 300 consecutive records are a real subset of a 32,443-row source log, not the whole Kaggle dataset. Only center images were downloaded.
3. **Show validation and EDA.** In Dataset overview, point out 300 valid records, no missing/invalid records, and the measured steering range. Show the histogram. In this subset 70.33% of labels satisfy `abs(steering) < 0.05`; no positive labels occur. Do not generalize to all tracks or claim this sample is sufficient for training.
4. **Show preprocessing.** Select another image index. Point out its ground-truth label. Compare original, cropped and processed views. Explain top 35%/bottom 10% cropping, resize to 160 x 80, RGB channels, and float32 normalization to `[0, 1]`. The output shape is `(80, 160, 3)`.
5. **Explain the manifests.** Show `data/processed/split_summary.json`: 205 train, 35 validation, 40 test and 20 excluded boundary rows. Filename timestamps increase across this subset. Five frames are dropped on each side of both boundaries. These are correlated parts of one short drive, not independent sessions.
6. **Show the CNN architecture.** Convolution extracts visual patterns; pooling keeps the model small; the dense layer combines features; dropout will help with overfitting during training; one linear output predicts a continuous value. Status is **Architecture Defined — Training Pending**. Do not describe the displayed label as a prediction.
7. **Show tests and remaining work.** Tests use artificial temporary images only to check code. Actual reports use downloaded simulator images. Training, baseline comparison, tuning, MAE/RMSE evaluation and prediction deployment are still pending.

## Files to show

| Artifact | What it demonstrates |
| --- | --- |
| `reports/dataset_summary.json` | Actual validation counts, steering statistics, source, scope, time and sampled image information |
| `reports/figures/steering_distribution.png` | Measured steering distribution |
| `reports/figures/sample_images.png` | Actual simulator images and corresponding labels |
| `reports/figures/preprocessing_comparison.png` | Original, cropped and processed stages |
| `data/processed/cleaned_metadata.csv` | Valid image paths, steering and source metadata |
| `data/processed/train.csv`, `val.csv`, `test.csv` | Disjoint image metadata, without copying images |
| `data/processed/excluded_boundary.csv` | Records deliberately left out near split boundaries |
| `src/model.py` | Small CNN and planned Adam/MSE configuration, without training |

## Short viva explanations

- **Why regression?** Steering is continuous. A linear output and MSE are suitable; classification accuracy is not the intended metric.
- **Why a lightweight CNN?** The initial machine has no dedicated GPU. Strided convolutions and global average pooling reduce computation and dense-layer parameter count.
- **Why no random row split?** Neighboring simulator frames are almost identical. Putting them in different splits can make performance look better than it really is.
- **Does the gap solve leakage?** No. It separates immediate neighbors, but routes and longer correlations remain. Independent recording sessions are preferable.
- **Why optional TensorFlow?** Dataset preparation and the dashboard should work before the model framework is installed.
- **What does flipping do?** It mirrors the road image and negates steering. Future augmentation belongs only in training.
- **What is the baseline?** Predict the mean training steering for every image, then compare MAE/RMSE with the CNN after training. It has not been evaluated yet.
- **What is not implemented?** Training, final test evaluation, trained weights and model prediction. The dashboard's steering values are labels from the source log.

## Optional model summary

Install before the lab if desired; otherwise show the textual architecture already in the dashboard:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-model.txt
.\.venv\Scripts\python.exe -m src.model
```

This only constructs and compiles a randomly initialized model. No fitting, trained-weight saving or inference result is presented.
