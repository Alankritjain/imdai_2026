# Actual results and final verification

Verified on **24 November 2026** using the local Windows CPU environment. Every performance value below comes from real simulator images. Synthetic tests do not contribute to these results.

## Data and training

| Item | Actual result |
| --- | --- |
| Dataset | `zaynena/selfdriving-car-simulator`, first 300 consecutive center frames |
| Valid / invalid records | 300 / 0 |
| Missing / duplicate image records | 0 / 0 |
| Existing frozen split | 205 training / 35 validation / 40 test; 20 boundary exclusions |
| Split method | Chronological; five frames removed on each side of both boundaries |
| Architecture | Three Conv2D layers → global average pooling → Dense(32) → Dropout(0.2) → Dense(1) |
| Trainable parameters | 21,329 |
| Optimizer / learning rate / loss | Adam / 0.001 / MSE |
| Batch / maximum epochs / patience / seed | 32 / 20 / 4 / 42 |
| Augmentation | Training only: flip probability 0.5 with label reversal; brightness 0.8–1.2 |
| Epochs completed | 5; early stopping |
| Best checkpoint epoch | 1 |
| Best validation MSE | 0.018133047968149185 |
| Training-mean baseline validation MSE | 0.01767790491391184 |
| Model path | `models/best_model.keras` (296,688 bytes) |
| Metadata path | `models/model_metadata.json` |
| Runtime | Python 3.11.17, TensorFlow 2.21.0, Keras 3.15.1, CPU |

Native Windows TensorFlow used CPU. Optional Colab GPU training was not executed. The model architecture and default settings were retained; no tuning was performed against test results.

## Held-out test results

Both methods evaluated the **same 40 test images** in original manifest order. Units are native simulator steering, not degrees.

| Model | MAE | RMSE |
| --- | ---: | ---: |
| Training-mean baseline | 0.1342439118 | 0.1618303412 |
| Trained CNN | 0.0829282007 | 0.1671253023 |

The baseline predicts the original training mean, **-0.1070731780**. The CNN has lower MAE but higher RMSE, so it does not outperform the baseline on both criteria. Its validation MSE is also slightly worse than the baseline's. No robust driving capability is established.

Sources: `reports/evaluation_metrics.json`, `reports/predictions.csv`, `reports/training_history.csv`, and model metadata. The model SHA-256 is `8b7ee5b67215d3763ddf767c575035edc9509d76104c6a185a233e8d02aef41f`. Checksums tie model, history, split content and predictions together.

## Interpretation and investigation

The test targets range from -0.6000001 to 0; **31 of 40** have absolute steering below 0.05. CNN outputs range only from approximately **-0.019579 to -0.017746**, with a standard deviation about 0.000479. This near-constant behavior explains why it handles the dominant near-zero labels reasonably while missing strong turns. The largest absolute error is about **0.581547**, on a target of -0.6000001 and prediction of -0.018453.

The training set has 68.29% near-zero labels and no positive original labels. Flip augmentation adds mirrored examples, but not independent situations. Only 205 training frames are available from about 21 seconds of one drive. This is a poor foundation for a useful generalizing model.

Training loss falls only slightly (0.051545 to about 0.049334); validation loss is best immediately and worsens afterward. This is consistent with limited learning and poor generalization, not successful learning of diverse turning behavior. Training uses augmentation and dropout whereas validation does not, so the lower absolute validation loss is not proof of better generalization. Early stopping correctly retains epoch 1.

The original/cropped/processed image figure was visually inspected: road boundaries remain visible after the crop. Tests verify consistent RGB, shape, saved crop settings and exactly one normalization step. Therefore no demonstrated preprocessing defect justified changing the crop or architecture. The primary observed limitation is data quantity/diversity and imbalance. More independent recordings and validation-guided training experiments should precede another independent final test evaluation. No post-test tuning was performed.

The initial-stage EDA had already described the complete excerpt. Although fitting/checkpoint selection never uses test targets, this should be presented as an educational holdout rather than an untouched independent research benchmark.

## Genuine inference and dashboard demonstration

The saved checkpoint was loaded in a separate CLI process using the first held-out image:

`center_2019_04_02_19_25_52_140.jpg`

Actual steering: **0.0**. Displayed CNN prediction: **-0.019007**. The same image uploaded through Chrome produced the same displayed value. A finite output verifies inference, not driving competence.

The local Streamlit server returned HTTP 200. A headless Chrome session exercised all six tabs, changed the source image to row 25, uploaded the real JPG, displayed actual metrics, and rejected a corrupt JPG without a page error. Screenshots were visually inspected. Evidence:

- `reports/dashboard_verification.json`
- `reports/dashboard_prediction.png`
- `reports/dashboard_performance.png`

The numeric gauge avoids physical-angle claims. Left/right labels are intentionally disabled because the sign convention has not been independently verified.

## Automated checks

**48 passed, 0 failed, 0 skipped**, with 60 upstream Keras/NumPy deprecation warnings. Evidence: `reports/tests.xml`. The earlier exploratory test run (`tests-pretraining.xml`) contained one test-helper name-shadowing failure, subsequently fixed; it is not the final result.

Tests cover CSV parsing/path recovery, missing/invalid data, copied-image detection, split gaps/overlap/frozen manifests, RGB/shape/cropping/normalization, flip labels and brightness, exact architecture/compilation/forward pass, model save/reload, finite outputs, invalid images, metric arithmetic, unchanged validation batches, and a complete tiny training → evaluation → inference → Streamlit upload workflow. Synthetic fixture artifacts stay in pytest temporary directories.

`pip check` reports no broken dependencies. Source compilation succeeds. The Colab notebook is valid JSON and all seven code cells compile; **Colab itself has not been run**. Exact installed versions are recorded in `requirements-lock.txt`.

## Acceptance status

| Requested area | Verified status |
| --- | --- |
| Real data, labels, statistics and separated manifests | Complete for the included 300-frame excerpt |
| Shared crop/resize/normalization | Implemented and tested |
| Actual training, history and best checkpoint | Completed on CPU |
| Saved model reload | Verified in tests and separate real-image CLI inference |
| Baseline, MAE/RMSE, plots and actual predictions | Generated from the fixed real test records |
| Dashboard launch and all six views | Verified through Streamlit tests and Chrome |
| Upload and actual CNN display | Verified with a real image and invalid-image handling |
| README, final guide, viva, demo flow and notebook | Provided; notebook execution remains optional/unverified |
| Full Kaggle training / independent-session generalization | Not performed; outside the available excerpt |
| Useful learned steering behavior / road safety | Not established |

The requested technical workflow is implemented and verified on available real data. This is not a claim of a fully validated driving model or complete-dataset integration. Milestone 1/2 source reports were absent, so alignment is based on the supplied request and existing architecture.
