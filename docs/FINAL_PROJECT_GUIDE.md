# Final project guide

## 1. Create and activate the environment

Open PowerShell in the project folder with Python 3.11 installed:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is restricted, substitute `.\.venv\Scripts\python.exe` for `python` below. No execution-policy change is required. Select that interpreter in VS Code.

## 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-model.txt
```

The second file is essential for model work. For the exact tested Windows environment, install `requirements-lock.txt` instead. CPU execution is supported; a supported TensorFlow GPU is used if available. Internet is needed for installation/downloads; the configured demo runs offline.

## 3. Prepare the raw dataset

Existing data is in `data/raw/lab_subset`: `driving_log.csv`, `provenance.json`, and `IMG/` with 300 genuine center images. On a new machine, copy that directory or run:

```powershell
python tools/download_demo_subset.py
```

If the public endpoint fails, download `zaynena/selfdriving-car-simulator` through Kaggle and extract under `data/raw`. Keep CSVs beside their images. Do not mix combined logs with original copies or use unlabelled/object-annotation images as steering training data.

## 4. Analyze and create manifests

```powershell
python -m src.explore_data --data-dir data/raw
python -m src.prepare_data --data-dir data/raw
```

For several logs, add `--csv "data/raw/lab_subset/driving_log.csv"`. On a fresh machine, omit old processed manifests and generate local paths. To preserve existing outputs, use preparation's `--output-dir data/processed_new`, then training/evaluation's `--processed-dir data/processed_new`. Identical existing splits are reused; changed splits are never silently overwritten.

Inspect statistics, source images, crop comparisons and split summary. The included split is 205/35/40 plus 20 exclusions from one short drive. Use verified session IDs when available. Never change test assignments after observing model performance.

## 5. Inspect and train

```powershell
python -m src.model
python -m src.train
```

The architecture command constructs fresh parameters; it does not train. Defaults are Adam 0.001, batch 32, up to 20 epochs, patience 4. Smaller batches reduce memory: `python -m src.train --batch-size 16`.

Look for finite loss/val_loss and the verified-checkpoint message. Confirm the model, matching metadata, history and loss figure exist. To preserve an existing model, use `--models-dir models/new_run --reports-dir reports/new_run`. Select experiments using validation only, then evaluate the fixed choice once on test data.

## 6. Evaluate

```powershell
python -m src.evaluate
```

This checks frozen image/label signatures and reports real CNN/baseline MAE/RMSE on the same held-out rows. The baseline uses only training labels. If the CNN loses, say so; the short sequential holdout cannot prove generalization.

For another selected experiment:

```powershell
python -m src.evaluate --model-path models/new_run/best_model.keras --reports-dir reports/new_run
```

## 7. Verify standalone inference

```powershell
python -m src.predict --sample-index 0
```

The model reloads and returns a finite number in native simulator units. No physical degrees or unverified left/right labels are displayed.

## 8. Test and launch Streamlit

```powershell
python -m pytest -q -p no:cacheprovider --basetemp .test-tmp-local
python -m streamlit run app.py
```

Use a fresh `.test-tmp-*` path if Windows permissions prevent reuse. TensorFlow tests must execute, not skip, before claiming the full workflow is verified. Test metrics are synthetic fixtures confined to temporary folders.

Open `http://localhost:8501` or the printed port. Stop with Ctrl+C. The dashboard caches model loading and never retrains on refresh. Transfer both model files plus matching reports when moving a trained model; modification times refresh the cache. Sidebar data selection is independent of the trained model's saved evaluation split.

The redesigned prediction page starts with **Built-in examples**, so no file upload is necessary. Move **Example frame** to compare the live CNN output with recorded steering. Switch to **Recorded replay**, turn on **Play recording**, and adjust preview speed to watch image-by-image inference. This replays recorded images, not an autonomous simulator. **Upload image** remains available for individual JPG/JPEG/PNG frames; **Save this prediction** exports genuine output as JSON.

If `active_run.json` exists, it selects the model, dataset, manifests and reports as one bundle. Older experiment files remain intact. Restart Streamlit after changing the active run. Defaults in all CLI commands follow that manifest.

## 9. Optional Colab

Open `notebooks/training_colab.ipynb`. Upload source/config/requirements plus raw data as a zip, excluding `.venv`, old models and absolute-path manifests. The notebook installs dependencies, prepares local manifests, calls the same training/evaluation modules, and exports artifacts. Choose a GPU runtime if available; otherwise use CPU. Match exported TensorFlow/Keras versions locally if loading fails. Colab execution is not established merely by supplying a notebook.

## 5–7 minute classroom demonstration

1. **0:00–0:40:** Project overview: human image/steering demonstrations, supervised regression, simulation-only scope.
2. **0:40–1:25:** Dataset: source provenance, real images, 300-frame excerpt, imbalance and chronological splits.
3. **1:25–2:10:** Preprocessing: change sample index; show crop, resize, normalization and `(80, 160, 3)`.
4. **2:10–2:50:** About: show the 21,329-parameter architecture summary; explain convolution, pooling and linear output.
5. **2:50–3:40:** Training: show loss curves, best epoch and early stopping. Discuss curve differences without automatically calling every gap overfitting.
6. **3:40–4:40:** Performance: compare both metrics with baseline; show scatter and held-out low/high-error examples honestly.
7. **4:40–5:45:** Prediction: upload a genuine simulator JPG; show the CNN output and numeric gauge, distinguishing it from a recorded label.
8. **5:45–6:30:** Limitations: no obstacle detection, braking or safe navigation; diverse independent recordings are future work.

Keep saved reports available offline, but identify them as saved results if live inference is not running. Use `RESULTS.md` for actual outcomes.
