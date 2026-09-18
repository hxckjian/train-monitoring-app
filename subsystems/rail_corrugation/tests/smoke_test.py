"""Root-runnable end-to-end smoke test."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pandas as pd

from subsystems.rail_corrugation.predict import MODEL_PATH, predict


TESTS_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = TESTS_DIR / "fixtures" / "sample_input.csv"

EXPECTED_COLUMNS = ["file_id", "prediction"]
ALLOWED_LABELS = {"Normal", "Side I", "Side II"}


def check(condition: bool, message: str) -> None:
    """Raise an error when a smoke-test condition fails."""
    if not condition:
        raise AssertionError(message)

    print(f"PASS: {message}")


def make_uploaded_file(path: Path) -> BytesIO:
    """Create a file-like object resembling Streamlit UploadedFile."""
    uploaded_file = BytesIO(path.read_bytes())
    uploaded_file.name = path.name
    return uploaded_file


def main() -> None:
    # Check required files.
    check(SAMPLE_PATH.is_file(), f"sample input exists: {SAMPLE_PATH}")
    check(MODEL_PATH.is_file(), f"model artifact exists: {MODEL_PATH}")

    # Check that the saved model can be loaded.
    artifact = joblib.load(MODEL_PATH)

    check(
        isinstance(artifact, dict),
        "model artifact has the expected dictionary format",
    )
    check(
        "model" in artifact,
        "model artifact contains the 'model' key",
    )
    check(
        hasattr(artifact["model"], "predict"),
        "saved model exposes predict()",
    )

    # Simulate a Streamlit upload.
    uploaded_file = make_uploaded_file(SAMPLE_PATH)

    # Run the complete prediction pipeline.
    result = predict([uploaded_file])

    # Check the returned object.
    check(
        isinstance(result, pd.DataFrame),
        "prediction is a DataFrame",
    )
    check(
        not result.empty,
        "prediction is not empty",
    )
    check(
        result.columns.tolist() == EXPECTED_COLUMNS,
        "output columns and order are exact",
    )

    # Check that the uploaded filename is preserved.
    expected_file_ids = [SAMPLE_PATH.name]
    actual_file_ids = result["file_id"].tolist()

    print(f"Expected file_id: {expected_file_ids}")
    print(f"Actual file_id:   {actual_file_ids}")

    check(
        actual_file_ids == expected_file_ids,
        "file identifier preserves uploaded filename",
    )

    # Check that all predictions use allowed labels.
    check(
        result["prediction"].isin(ALLOWED_LABELS).all(),
        "prediction labels are allowed",
    )

    # Check that the result can be exported and read again.
    with TemporaryDirectory() as directory:
        output_path = Path(directory) / "prediction.csv"
        result.to_csv(output_path, index=False)
        round_trip = pd.read_csv(output_path)

        check(
            len(round_trip) == len(result),
            "CSV round trip preserves row count",
        )
        check(
            round_trip.columns.tolist() == result.columns.tolist(),
            "CSV round trip preserves schema",
        )

    print("\nPASS: rail-corrugation smoke test complete")
    print("\nPrediction result:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()