"""Select a verified experiment for the dashboard and default CLI commands."""

import argparse
import json
from pathlib import Path

from config import PROJECT_DIR
from src.pipeline import file_sha256, load_splits, write_json
from src.predict import load_trained_model


def activate_run(models_dir, reports_dir, processed_dir, data_dir, selection_file):
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    _, metadata = load_trained_model(models_dir / "best_model.keras")
    evaluation = json.loads((reports_dir / "evaluation_metrics.json").read_text(encoding="utf-8"))
    selection = json.loads(Path(selection_file).read_text(encoding="utf-8"))
    if Path(selection["selected_model"]).resolve() != (models_dir / "best_model.keras").resolve():
        raise ValueError("This model was not selected by the validation comparison.")
    _, _, signatures = load_splits(processed_dir)
    if signatures != metadata["split_signatures"] or signatures != evaluation["split_signatures"]:
        raise ValueError("Model/evaluation/splits are not the same experiment.")
    if evaluation["model_sha256"] != metadata["model_sha256"]:
        raise ValueError("Evaluation belongs to another checkpoint.")
    if file_sha256(reports_dir / "predictions.csv") != evaluation["predictions_sha256"]:
        raise ValueError("Predictions were modified after evaluation.")
    if file_sha256(reports_dir / "training_history.csv") != metadata["history_sha256"]:
        raise ValueError("Training history belongs to another run.")
    paths = {"DATA_DIR": data_dir, "PROCESSED_DIR": processed_dir,
             "MODELS_DIR": models_dir, "REPORTS_DIR": reports_dir}
    active = {key: Path(value).resolve().relative_to(PROJECT_DIR).as_posix() for key, value in paths.items()}
    active["selection_file"] = Path(selection_file).resolve().relative_to(PROJECT_DIR).as_posix()
    write_json(PROJECT_DIR / "active_run.json", active)
    print(json.dumps(active, indent=2))
    print("Verified experiment activated. Restart Streamlit to load the new default paths.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("models-dir", "reports-dir", "processed-dir", "data-dir", "selection-file"):
        parser.add_argument("--" + name, type=Path, required=True)
    activate_run(**vars(parser.parse_args()))
