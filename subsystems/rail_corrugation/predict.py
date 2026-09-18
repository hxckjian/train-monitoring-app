"""Inference entry point for rail-corrugation classification."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .features import LABELS, extract_features, load_input

MODULE_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODULE_DIR / "artifacts" / "model.joblib"


def load_saved_model() -> dict:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Run the training module first."
        )
    artifact = joblib.load(MODEL_PATH)
    required = {"model", "feature_names", "input_columns", "labels"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError("The saved rail-corrugation artifact is invalid or incomplete.")
    return artifact


def predict(uploaded_files) -> pd.DataFrame:
    """
    Accept a list of uploaded file-like objects and return a
    submission-ready prediction DataFrame.
    """
    if uploaded_files is None or isinstance(uploaded_files, (str, bytes)):
        raise ValueError("uploaded_files must be a non-empty list of CSV file-like objects.")
    try:
        files = list(uploaded_files)
    except TypeError as exc:
        raise ValueError("uploaded_files must be an iterable of file-like objects.") from exc
    if not files:
        raise ValueError("At least one rail-corrugation CSV file is required.")

    artifact = load_saved_model()
    results = []
    for uploaded_file in files:
        name = getattr(uploaded_file, "name", None)
        if not name:
            raise ValueError("Every uploaded file must have a non-empty .name attribute.")
        file_id = Path(str(name)).name
        if Path(file_id).suffix.lower() != ".csv":
            raise ValueError(f"Unsupported file extension for {file_id}; expected .csv.")
        try:
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            frame = load_input(uploaded_file)
            features = extract_features(frame)
        except Exception as exc:
            raise ValueError(f"Invalid input {file_id}: {exc}") from exc
        if features.columns.tolist() != artifact["feature_names"]:
            raise ValueError("Extracted features do not match the saved model artifact.")
        label = str(artifact["model"].predict(features)[0])
        if label not in LABELS:
            raise ValueError(f"Model produced an unsupported label: {label}")
        results.append({"file_id": file_id, "prediction": label})
    return pd.DataFrame(results, columns=["file_id", "prediction"])
