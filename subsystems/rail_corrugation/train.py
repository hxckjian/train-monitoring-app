"""Train and persist the rail-corrugation MVP classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

try:
    from .features import LABELS, expected_columns, extract_file_features
except ImportError:  # allows: python subsystems/rail_corrugation/train.py
    from features import LABELS, expected_columns, extract_file_features

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "02_Datasets" / "Rail_Corrugation"
ARTIFACT_PATH = MODULE_DIR / "artifacts" / "model.joblib"
RANDOM_SEED = 42


def augment_side_swap(features: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Mirror rail sides and swap fault labels using the documented sensor symmetry."""
    mirrored = features.copy()
    for column_i in [c for c in features.columns if c.startswith("side_i_")]:
        column_ii = "side_ii_" + column_i[len("side_i_"):]
        if column_ii in features.columns:
            mirrored[column_i] = features[column_ii].to_numpy()
            mirrored[column_ii] = features[column_i].to_numpy()
    for column in [c for c in features.columns if c.startswith("side_contrast_")]:
        mirrored[column] = -features[column].to_numpy()
    mirrored_labels = labels.replace({"Side I": "Side II", "Side II": "Side I"})
    return (
        pd.concat([features, mirrored], ignore_index=True),
        pd.concat([labels, mirrored_labels], ignore_index=True),
    )


def build_feature_table(labels: pd.DataFrame, train_dir: Path) -> pd.DataFrame:
    rows = []
    total = len(labels)
    for number, item in enumerate(labels.itertuples(index=False), start=1):
        path = train_dir / item.filename
        if not path.is_file():
            raise FileNotFoundError(f"Labelled training file is missing: {path}")
        features = extract_file_features(path)
        features.insert(0, "filename", item.filename)
        features.insert(1, "label", item.label)
        rows.append(features)
        if number == 1 or number % 25 == 0 or number == total:
            print(f"Extracted features: {number}/{total}", flush=True)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    args = parser.parse_args()

    labels_path = args.data_dir / "Train_Labels.csv"
    train_dir = args.data_dir / "Train"
    labels = pd.read_csv(labels_path)
    if labels.columns.tolist() != ["filename", "label"]:
        raise ValueError("Train_Labels.csv must contain filename,label in that order.")
    if labels.filename.duplicated().any() or labels.isna().any().any():
        raise ValueError("Training labels contain duplicates or missing values.")
    unknown = sorted(set(labels.label) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown training labels: {unknown}")

    table = build_feature_table(labels, train_dir)
    feature_names = [c for c in table.columns if c not in ("filename", "label")]
    train_rows, valid_rows = train_test_split(
        table.index,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=table["label"],
    )
    model_kwargs = dict(
        n_estimators=500,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    validation_model = ExtraTreesClassifier(**model_kwargs)
    augmented_x, augmented_y = augment_side_swap(
        table.loc[train_rows, feature_names].reset_index(drop=True),
        table.loc[train_rows, "label"].reset_index(drop=True),
    )
    validation_model.fit(augmented_x, augmented_y)
    valid_prediction = validation_model.predict(table.loc[valid_rows, feature_names])
    score = f1_score(
        table.loc[valid_rows, "label"], valid_prediction, labels=list(LABELS), average="macro"
    )
    print(f"Validation macro F1: {score:.4f}")
    print(classification_report(table.loc[valid_rows, "label"], valid_prediction, labels=list(LABELS), zero_division=0))
    print("Confusion matrix (rows=true, columns=pred; Normal, Side I, Side II):")
    print(confusion_matrix(table.loc[valid_rows, "label"], valid_prediction, labels=list(LABELS)))

    final_model = ExtraTreesClassifier(**model_kwargs)
    final_x, final_y = augment_side_swap(table[feature_names], table["label"])
    final_model.fit(final_x, final_y)
    artifact = {
        "model": final_model,
        "feature_names": feature_names,
        "input_columns": expected_columns(),
        "labels": list(LABELS),
        "validation_macro_f1": float(score),
        "validation_files": table.loc[valid_rows, "filename"].tolist(),
        "random_seed": RANDOM_SEED,
        "training_file_count": len(table),
        "class_counts": table["label"].value_counts().to_dict(),
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.artifact, compress=3)
    print(f"Saved final artifact: {args.artifact}")


if __name__ == "__main__":
    main()
