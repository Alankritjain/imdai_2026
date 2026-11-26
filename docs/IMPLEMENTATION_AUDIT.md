# Implementation audit

Audit date: 24 November 2026. No Milestone 1/2 reports or AGENTS.md were found in the workspace. The supplied project request and existing architecture/roadmap define the implemented scope. No repository history was available (.git is absent).

| Component | Observed before changes | Action |
| --- | --- | --- |
| Structure | Modular Python project, Windows Python 3.11 environment | Retained |
| Real dataset | 300 center-camera images; source log and download provenance | Revalidated; used existing excerpt |
| Loading/validation | Headered/headerless CSV, foreign paths, finite labels, readability, path duplicates | Added SHA-256 detection of differently named exact copies |
| Exploration | Real JSON statistics and three figures | Retained and refreshed |
| Preprocessing | RGB, top 35%/bottom 10% crop, 160 x 80, float32 / 255 | Shared unchanged by training and inference |
| Splits | 205 train / 35 validation / 40 test, 20 excluded; chronological | Preserved; repeat preparation reuses identical manifests and refuses changed ones |
| CNN | Correct three-convolution architecture; TensorFlow absent | Kept architecture; configurable learning rate; installed TensorFlow for execution |
| Training | Missing | Added streaming batches, augmentation, Adam/MSE, checkpoint, early stopping, metadata/history |
| Baseline/evaluation | Missing | Training-only mean; held-out MAE/RMSE, plots, predictions and artifact identity checks |
| Inference | Missing | Reloadable checkpoint with saved preprocessing and finite output validation |
| Dashboard | Four tabs; dataset/preprocessing only | Six tabs including upload prediction and real performance |
| Tests | 29 existing tests | Expanded with model, prediction, artifact consistency and full workflow tests |
| Documentation | Initial-stage README and two guides | Final guide, viva preparation, Colab notebook and updated results |

The initial default pytest run had 10 passes and 19 setup errors because an existing system temporary directory was inaccessible. A fresh workspace `--basetemp` resolved this: all 29 original tests passed before implementation. This was an environment permission issue, not 19 application failures.

The 300 images are from about 21.3 seconds of one recording. There are no positive original steering labels; 70.33% are near zero. Filename timestamps increase, but separate sessions cannot be established. The five-frame gap on each side of a boundary reduces immediate-neighbor overlap, not longer temporal correlation. Existing exploratory reports had already described the full excerpt before this work; this is an educational holdout, not a pristine independent research benchmark. No test labels are used for fitting or checkpoint selection.

The original README's assumed sign convention was not evidence for this particular dataset. Inference therefore displays numbers and a numeric gauge without left/right labels. Do not replace this with an unsupported degrees conversion.

Exact duplicate detection uses file bytes. Re-encoded copies and near-identical neighboring scenes still require a session/content audit. Raw data is excluded from version control. No source dataset was synthesized for project performance reports.

Final verification: all 48 expanded tests pass; real-data training completed for five epochs; the fixed checkpoint was evaluated on 40 real held-out images. Chrome verified six tabs, sample selection, real upload inference and invalid upload handling. See RESULTS.md for the mixed CNN/baseline comparison and evidence files. The seven Colab code cells compile, but Colab execution was not performed.
