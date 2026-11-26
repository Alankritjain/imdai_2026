# Self-Driving Car Behavioral Cloning Using a Lightweight CNN

This B.Tech AI project learns a continuous steering command from a simulator's center-camera road image. Behavioral cloning is supervised learning: recorded human steering is the target, and a CNN learns visual features from RGB pixels. This is regression, so evaluation uses MAE and RMSE rather than classification accuracy.

> Simulation-based steering predictor; not a complete or road-safe autonomous-driving system.

The objective is a reproducible image-to-steering pipeline, baseline comparison, and interactive academic demonstration. The stack is Python, TensorFlow/Keras, NumPy, Pandas, Pillow, Matplotlib, Streamlit and pytest. Metrics use NumPy directly, avoiding an unnecessary scikit-learn dependency.

## Run the existing installation

```powershell
cd C:\imda_sem5\self-driving-behavioral-cloning
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501`. No environment activation is required. See the [final project guide](docs/FINAL_PROJECT_GUIDE.md), [implementation audit](docs/IMPLEMENTATION_AUDIT.md), [actual results](docs/RESULTS.md), and [viva guide](docs/VIVA_PREPARATION.md).

## Installation and commands

On a new computer, install Python 3.11 and create a fresh environment; virtual environments are not portable.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-model.txt
python tools/download_demo_subset.py
python -m src.explore_data --data-dir data/raw
python -m src.prepare_data --data-dir data/raw
python -m src.model
python -m src.train
python -m src.evaluate
python -m streamlit run app.py
python -m pytest -q -p no:cacheprovider --basetemp .test-tmp-local
```

If activation is blocked, substitute `.\.venv\Scripts\python.exe` for `python`. TensorFlow is required for training/evaluation/inference; exploration works without it. `requirements-lock.txt` records the exact tested Windows environment. Use the main requirements files for compatible Windows/Colab versions, or install the lock file on Python 3.11 to reproduce this environment.

Training protects existing models. For another validation-selected experiment:

```powershell
python -m src.train --epochs 20 --batch-size 16 --learning-rate 0.001 --patience 4 --models-dir models/experiment2 --reports-dir reports/experiment2
```

Do not repeatedly evaluate the test set to select hyperparameters. The final evaluation accepts `--model-path`, `--processed-dir`, `--reports-dir`. The dashboard uses `models/` and `reports/`; transfer a matching selected bundle there before demonstrating it.

## Dataset and validation

Source: [zaynena/selfdriving-car-simulator](https://www.kaggle.com/datasets/zaynena/selfdriving-car-simulator). Only center-camera images and their verified steering labels are used. The earlier object-annotation dataset candidate is not used.

The included excerpt is the first **300 consecutive rows** of `dataset/dataset/driving_log.csv`, a 32,443-row source log. Its headerless columns are `center,left,right,steering,throttle,brake,speed`. Original labels are unchanged. The full source log, SHA-256 and download provenance are retained under `data/raw`. The excerpt spans about 21.3 seconds on 2019-04-02; filename timestamps increase from 19:25:33.671 to 19:25:54.973.

| Observed subset statistic | Value |
| --- | --- |
| Valid / total images | 300 / 300 |
| Invalid, missing or exact duplicate records | 0 |
| Steering minimum / maximum | -0.8500001 / 0.0 |
| Mean steering | -0.09783334 |
| Near-zero labels, absolute value < 0.05 | 70.33% |
| Image dimensions | 320 x 160; saved EDA inspects 100 evenly spaced images |

Every image is decoded during validation. These are subset statistics, not full-dataset results. The loader handles headered/headerless logs, whitespace, finite targets, missing/corrupt images, foreign Windows filenames and ambiguous paths. It rejects repeated paths and byte-identical copies. Re-encoded copies and neighboring scenes still need a broader audit. The first valid occurrence is retained, with one counted failure per rejected row.

For more coverage, download the full dataset manually. Keep each log beside its `IMG` folder, select exactly one with `--csv`, and do not mix combined logs with original copies. Optional `session_id`, `drive_id` or `recording_id` columns support session grouping; folders alone do not establish independent drives. If Kaggle's public file endpoint used by the subset downloader stops working, use the dataset page. Existing local data supports offline demonstration.

## Preprocessing and split discipline

All model inputs follow the same Pillow pipeline:

1. Convert to RGB.
2. Crop top 35% and bottom 10%: retain rows 56 through 143 in a 160-pixel-tall image.
3. Resize with bilinear interpolation to width 160, height 80.
4. Convert to float32 and divide by 255 once: `(80, 160, 3)` in `[0, 1]`.

Cropping removes sky/dashboard in these views; other cameras require review. Metadata saves the settings, so changing `config.py` later cannot silently change inference. Training additionally flips horizontally with probability 0.5 (negating steering) and adjusts brightness by a factor in 0.8–1.2. Validation/test/inference are never augmented.

The target split is approximately 70/15/15, grouping whole identified sessions when possible with seed 42. Otherwise, chronological blocks exclude five valid frames on each side of both boundaries. The retained existing excerpt has **205 train, 35 validation, 40 test, 20 excluded**. Proportions differ slightly after gaps.

Preparation reuses identical manifests and refuses changed ones. On another machine, prepare fresh local manifests because paths are absolute. Training/evaluation check overlapping paths, exact copies, finite labels and chronology. Content/label signatures bind all splits to the model independently of absolute directories. Test metadata is checked for separation, but its labels are not used to fit the CNN or select checkpoints.

One short drive does **not** provide independent routes or sessions. Gaps do not eliminate longer scene correlation. Earlier exploratory reports had already described all 300 labels; this is an educational holdout, not a pristine independent research benchmark.

## CNN and training

| Layer | Configuration | Purpose |
| --- | --- | --- |
| Input | 80 x 160 x 3 | Normalized road image |
| Conv2D | 16 filters, 5 x 5, stride 2, ReLU | Learn local visual features |
| Conv2D | 32 filters, 3 x 3, stride 2, ReLU | Combine features |
| Conv2D | 48 filters, 3 x 3, stride 2, ReLU | Higher-level features |
| GlobalAveragePooling2D | Spatial mean per feature map | Reduce parameters |
| Dense | 32, ReLU | Combine features |
| Dropout | 0.2 | Regularize training |
| Dense | 1, linear | Continuous steering |

The model has **21,329 trainable parameters** and `valid` convolution padding. ReLU introduces nonlinearity. Linear output supports continuous values without classification or clipping.

Defaults: Adam, learning rate 0.001, MSE, batch 32, maximum 20 epochs, early-stopping patience 4, seed 42. CLI options change settings in one place. The `tf.data` generator loads images on demand, shuffles training metadata, and prefetches one batch. It never loads all images into RAM. A supported GPU is used if available; otherwise CPU. Seeds/deterministic operations improve repeatability; different hardware/runtime versions can still produce numerical differences.

`ModelCheckpoint` saves minimum validation MSE. `EarlyStopping` restores best weights; inference reloads the saved best checkpoint. Nonfinite losses fail the run. Completed metadata is written only after reloading and a real validation-image prediction succeed. An interrupted run may leave a checkpoint without completed metadata; inference refuses it. Use a new experiment directory when restarting.

Artifacts include `models/best_model.keras`, `models/model_metadata.json`, `reports/training_history.csv` and `reports/figures/training_loss.png`. Metadata records preprocessing, settings, augmentation, sizes, signatures, runtime versions, best epoch and model/history checksums.

## Evaluation and actual results

The CNN and baseline use the same fixed test rows. The baseline predicts the **original training-label mean**, never the test mean or augmented-label mean. MAE is mean absolute error; RMSE is square root of mean squared error. Both use native simulator units, not degrees. Lower is better; the CNN is not guaranteed to outperform the baseline.

See [RESULTS.md](docs/RESULTS.md) for actual measured outcomes. Machine-readable sources are `reports/evaluation_metrics.json` and `reports/predictions.csv` (image, actual, prediction, baseline, signed error, absolute error). Figures show training/validation loss, actual-versus-predicted scatter with identity line, error distribution and held-out low/high-error examples.

## Dashboard and inference

Six tabs provide steering prediction, model performance, dataset exploration, preprocessing, overview and project information. The dashboard now opens directly on **Built-in examples**: move the frame slider to run genuine predictions on held-out road images, with recorded steering and absolute error alongside. **Upload image** accepts JPG/JPEG/PNG. **Recorded replay** provides play/pause, speed and starting-frame controls; it recomputes CNN outputs as recorded frames advance and plots recent actual/predicted values. This is recorded playback, not a live car-control simulator.

The cached trained model and saved preprocessing produce genuine values; labels are never substituted for predictions. Expand **See what the model receives** to view the normalized input, or download an individual prediction as JSON. The sidebar's **Dataset settings** keeps advanced path controls out of the main demonstration. Missing artifacts and invalid images produce helpful messages. Performance checks bind results to the model and prediction file.

The gauge shows numeric simulator steering. Left/right labels are disabled because the dataset-specific sign convention is unverified. The analysis threshold `abs(steering) < 0.05` is descriptive, not a universal physical driving category.

```powershell
python -m src.predict --sample-index 0
# Or provide an image path:
python -m src.predict "data/raw/lab_subset/IMG/center_2019_04_02_19_25_52_140.jpg"
```

`notebooks/training_colab.ipynb` imports the same modules, prepares local paths, trains/evaluates, and exports artifacts with installation versions. Providing a notebook is not evidence that Colab was executed.

## Layout

```text
self-driving-behavioral-cloning/
  app.py, config.py, README.md, requirements*.txt, .gitignore
  src/
    data_loader.py, explore_data.py, preprocess.py, prepare_data.py
    model.py, pipeline.py, train.py, evaluate.py, predict.py, __init__.py
  data/raw/                  # Real subset, source metadata, provenance
  data/processed/            # Frozen manifests and split summary
  models/                    # Generated by training
  reports/                   # Real analysis, training, evaluation, test logs
    figures/
  tests/                     # Synthetic fixtures stay in temporary folders
  docs/
    IMPLEMENTATION_AUDIT.md, FINAL_PROJECT_GUIDE.md, VIVA_PREPARATION.md
    RESULTS.md, PROJECT_ROADMAP.md, LAB_25_PERCENT_DEMO.md
  notebooks/training_colab.ipynb
  tools/download_demo_subset.py
```

## Limitations and future scope

The excerpt is small, imbalanced and contains no positive original labels. Flipping does not create new independent situations. A numerical prediction establishes functioning inference, not useful driving behavior. Future work needs diverse recordings, independent sessions and validation-guided improvements to crop quality, balancing, learning rate or training duration. Use a new independent final test set after such development.

The project does not detect obstacles, recognize signals, apply brakes, control speed or guarantee safe navigation. Simulator-to-road differences and absence of closed-loop control prevent real-car deployment. The scope remains a lightweight educational steering predictor.
