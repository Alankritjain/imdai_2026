# Project roadmap

> Updated 24 November 2026: the training, evaluation and prediction implementation described below is now present and tested on the real 300-frame excerpt. See RESULTS.md for actual metrics and remaining limitations. The sections below preserve the original stage plan; their pending-work statements describe the earlier version. Full-dataset integration and independent-session validation remain future work.

The percentages below name academic development stages, not measured completion scores.

## Initial 25%: implemented foundation

- Python 3.11 environment and a modular project layout.
- Real 300-record contiguous subset from `zaynena/selfdriving-car-simulator`, with provenance.
- Center-camera CSV loading, validation and clean metadata.
- Actual steering statistics, distribution, road-image grid and preprocessing comparison.
- RGB conversion, percentage cropping, resizing to 160 x 80 and normalization.
- Session-aware split support, with a documented chronological fallback used for this sample.
- Train/validation/test manifests and accounted boundary exclusions.
- Lightweight CNN architecture defined; optional TensorFlow import.
- Streamlit dataset explorer, instructions and functional tests.

The subset validates the pipeline; it does not establish full dataset integration or independent train/test drives. TensorFlow runtime inspection needs the optional dependency. There is no trained model, model evaluation, baseline result or prediction deployment.

## Next milestone: data audit and initial training

1. Download the full selected simulator data, choose combined or original sources explicitly, and audit duplicated content.
2. Identify real recording sessions and check timestamp order. Save trustworthy session IDs; do not equate tracks with independent drives.
3. Review training-only steering distribution and crop settings. Consider training-only flip and brightness augmentation.
4. Freeze the test manifest before tuning. Revisit the small boundary gap using frame rate and session information.
5. Install TensorFlow, inspect the model summary, and train on CPU with Adam (0.001), MSE, batch size 32 and at most about 20 epochs.
6. Use validation loss for early stopping (patience 3–5). Keep seeds, dataset selection, split strategy and settings with each experiment.

## Later milestone: comparison and validation

- Predict the training-label mean as a simple baseline; never compute that baseline mean from test labels.
- Compare model and baseline MAE/RMSE on validation data.
- Inspect errors across steering ranges and failure cases.
- Tune only with training/validation data; do not repeatedly inspect held-out test performance.
- Once decisions are fixed, run the final test evaluation and document uncertainty and correlation limitations.

## Final milestone: prediction interface and report

- Load clearly identified trained weights and add a steering prediction page.
- Keep preprocessing identical between training and inference.
- Distinguish predictions from labels and explain unsupported image/domain limitations.
- Record final results, limitations, references and demonstration steps.
- Retain the simulation-only disclaimer. No physical vehicle deployment is planned.
