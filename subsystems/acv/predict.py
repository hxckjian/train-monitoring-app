"""Upload API and command-line inference; never trains."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import pandas as pd

if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from subsystems.acv.features import load_input, extract_features
    from subsystems.acv.model import rank_cars
else:
    from .features import load_input, extract_features
    from .model import rank_cars

MODULE_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODULE_DIR / 'artifacts' / 'model.json'

def load_saved_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError('Missing ACV artifacts/model.json. Copy the complete subsystem package.')
    model = json.loads(MODEL_PATH.read_text(encoding='utf-8'))
    if model.get('version') != 1:
        raise ValueError('Unsupported ACV model artifact version.')
    return model

def predict(uploaded_files) -> pd.DataFrame:
    """Accept a list of binary uploads and return submission-ready predictions."""
    if not isinstance(uploaded_files, list) or not uploaded_files:
        raise ValueError('Provide a nonempty list of .xlsx uploaded files.')
    model = load_saved_model()
    rows, seen = [], set()
    for uploaded in uploaded_files:
        name = str(getattr(uploaded, 'name', '')).replace('\\', '/').split('/')[-1]
        if not name or Path(name).suffix.lower() != '.xlsx' or not hasattr(uploaded, 'read'):
            raise ValueError('Each upload must be a binary file-like object named with an .xlsx extension.')
        if name in seen:
            raise ValueError(f'Duplicate uploaded filename: {name}')
        seen.add(name)
        try:
            ranking = rank_cars(model, extract_features(load_input(uploaded)))
        except ValueError as exc:
            raise ValueError(f'{name}: {exc}') from exc
        rows.append({'file_id': name, 'ranked_cars': '|'.join(ranking)})
    return pd.DataFrame(rows, columns=['file_id', 'ranked_cars'])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='One .xlsx file or directory of .xlsx cases')
    parser.add_argument('--output', required=True, type=Path, help='Output CSV path')
    args = parser.parse_args()
    paths = sorted(args.input.glob('*.xlsx')) if args.input.is_dir() else [args.input]
    with ExitStack() as stack:
        result = predict([stack.enter_context(p.open('rb')) for p in paths])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))

if __name__ == '__main__':
    main()
