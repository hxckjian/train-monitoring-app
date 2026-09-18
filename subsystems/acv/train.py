"""Train-group validation, EDA and final artifact creation."""
import argparse
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import openpyxl
from .features import load_input, extract_features, clean_data, car_columns, ALIASES
from .model import fit_model, rank_cars, rank_decay

MODULE_DIR = Path(__file__).resolve().parent

def inspect_case(raw, path):
    clean = clean_data(raw)
    numeric = raw.select_dtypes(include='number')
    times = pd.to_datetime(raw['Time'], errors='coerce')
    intervals = clean['Time'].diff().dt.total_seconds().dropna()
    stats = []
    for car, columns in car_columns(raw).items():
        col = next(columns[a] for a in ALIASES['indoor'] if a in columns)
        values = pd.to_numeric(raw[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        stats.append(dict(car=car, missing_or_invalid=int(values.isna().sum()), minimum=float(values.min()), median=float(values.median()), maximum=float(values.max()), std=float(values.std()), first_half_mean=float(values.iloc[:len(values)//2].mean()), second_half_mean=float(values.iloc[len(values)//2:].mean())))
    return dict(file=path.name, sheet=raw.attrs['sheet'], shape=list(raw.shape), train=str(raw['Train number'].dropna().iloc[0]), car_model=str(raw['Car model'].dropna().iloc[0]), start=str(times.min()), end=str(times.max()), dtypes=raw.dtypes.astype(str).value_counts().to_dict(), missing_cells=int(raw.isna().sum().sum()), infinite_numeric_cells=int(np.isinf(numeric.to_numpy(float)).sum()), duplicate_rows=int(raw.duplicated().sum()), duplicate_timestamps=int(times.duplicated().sum()), invalid_timestamps=int(times.isna().sum()), ordered=bool(times.is_monotonic_increasing), interval_counts={str(k):int(v) for k,v in intervals.value_counts().head(10).items()}, constant_columns=int((raw.nunique(dropna=True)<=1).sum()), indoor_statistics=stats)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=MODULE_DIR.parents[1] / '02_Datasets' / 'ACV')
    args = parser.parse_args()
    labels = pd.read_csv(args.data_dir / 'Train_Labels.csv', dtype=str)
    if list(labels.columns) != ['filename', 'faulty_car'] or labels.isna().any().any() or labels.filename.duplicated().any():
        raise ValueError('Invalid Train_Labels.csv schema or duplicate/missing labels.')
    paths = sorted((args.data_dir / 'Train').glob('*.xlsx'))
    if set(labels.filename) != {p.name for p in paths}:
        raise ValueError('Training files and label filenames must match exactly.')
    cases, eda = [], []
    for path in paths:
        print(f'Reading {path.name}', flush=True)
        raw = load_input(path)
        features = extract_features(raw)
        faulty = labels.set_index('filename').loc[path.name, 'faulty_car']
        if faulty not in features.index:
            raise ValueError(f'{path.name}: labelled car not in headers.')
        group = str(raw['Car model'].dropna().iloc[0]) + ':' + str(raw['Train number'].dropna().iloc[0])
        cases.append(dict(file=path.name, group=group, features=features, faulty=faulty))
        eda.append(inspect_case(raw, path))
    def fit(cases_to_fit):
        features = pd.concat([c['features'] for c in cases_to_fit])
        y = np.concatenate([(c['features'].index == c['faulty']).astype(float) for c in cases_to_fit])
        return fit_model(features, y)
    validation = []
    for group in sorted({c['group'] for c in cases}):
        training = [c for c in cases if c['group'] != group]
        model = fit(training)
        for case in [c for c in cases if c['group'] == group]:
            ranking = rank_cars(model, case['features'])
            row = dict(file_id=case['file'], held_out_group=group, training_files=[c['file'] for c in training], faulty_car=case['faulty'], ranked_cars='|'.join(ranking), rank=ranking.index(case['faulty'])+1, score=rank_decay(ranking, case['faulty']))
            validation.append(row)
            print(f"{case['file']}: rank={row['rank']}, rank-decay={row['score']:.3f}", flush=True)
    result = dict(metric='mean linear rank-decay', primary_metric=float(np.mean([r['score'] for r in validation])), top1_accuracy=float(np.mean([r['rank']==1 for r in validation])), split='leave-one-train-out', seed=42, random_operations=False, folds=validation, versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, openpyxl=openpyxl.__version__))
    artifacts = MODULE_DIR / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    (artifacts / 'model.json').write_text(json.dumps(fit(cases), indent=2), encoding='utf-8')
    (artifacts / 'validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    # JSON null represents unavailable statistics (rather than nonstandard NaN).
    eda_json = json.loads(json.dumps(eda), parse_constant=lambda _: None)
    (MODULE_DIR / 'eda.json').write_text(json.dumps(eda_json, indent=2, allow_nan=False), encoding='utf-8')
    pd.concat([c['features'].assign(file_id=c['file'], faulty=(c['features'].index == c['faulty']).astype(int)) for c in cases]).rename_axis('car').to_csv(MODULE_DIR / 'training_features.csv')
    print(f"Official validation primary_metric: {result['primary_metric']:.6f}; top-1: {result['top1_accuracy']:.6f}", flush=True)

if __name__ == '__main__':
    main()
