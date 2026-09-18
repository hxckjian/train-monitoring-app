"""Shared deterministic workbook parsing and case-level feature extraction."""
from io import BytesIO
from pathlib import Path
import re
import numpy as np
import pandas as pd
import openpyxl

ALIASES = {
    'indoor': ('Indoor Average Temperature', 'Passenger Cabin Temperature Detected Value'),
    'outdoor': ('Outdoor Average Temperature', 'Outside Temperature Sensor Reading', 'Fresh Air Temperature Detected Value'),
    'target': ('ACV Control Temperature (Cooling)', 'Target Temperature Value'),
    'mode': ('ACV Running Mode',),
}

def load_input(source):
    """Read the single telemetry sheet, preserving upload cursor and identifiers."""
    if hasattr(source, 'read'):
        position = source.tell()
        try:
            source.seek(0)
            data = BytesIO(source.read())
        finally:
            source.seek(position)
    else:
        data = Path(source)
    try:
        workbook = openpyxl.load_workbook(data, read_only=True, data_only=True)
    except Exception as exc:
        raise ValueError('Cannot read input as an .xlsx workbook.') from exc
    try:
        candidates = []
        for sheet in workbook:
            headers = list(next(sheet.values, ()))
            if 'Time' in headers and any(re.fullmatch(r'Car \d{2} - .+', str(c)) for c in headers):
                candidates.append((sheet, headers))
        if len(candidates) != 1:
            raise ValueError('Expected exactly one telemetry sheet containing Time and Car NN columns.')
        sheet, headers = candidates[0]
        if len(headers) != len(set(headers)):
            raise ValueError('Duplicate column headers are not supported.')
        frame = pd.DataFrame(sheet.iter_rows(min_row=2, values_only=True), columns=headers)
        frame.attrs['sheet'] = sheet.title
        validate_input(frame)
        return frame
    finally:
        workbook.close()

def car_columns(frame):
    result = {}
    for column in frame.columns:
        match = re.fullmatch(r'Car (\d{2}) - (.+)', str(column))
        if match:
            result.setdefault(match[1], {})[match[2]] = column
    return dict(sorted(result.items()))

def validate_input(frame):
    required = {'Car model', 'Train number', 'Time'}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError('Nonempty telemetry requires Car model, Train number, and Time.')
    cars = car_columns(frame)
    if len(cars) != 8:
        raise ValueError(f'Expected 8 cars; found {len(cars)}.')
    for car, columns in cars.items():
        for kind in ('indoor', 'target', 'mode'):
            if not any(alias in columns for alias in ALIASES[kind]):
                raise ValueError(f'Car {car}: missing required {kind} column; expected {ALIASES[kind]}.')
    for col in ('Car model', 'Train number'):
        if frame[col].dropna().astype(str).nunique() != 1:
            raise ValueError(f'Expected one nonmissing {col} per case.')

def clean_data(frame):
    result = frame.copy()
    result['Time'] = pd.to_datetime(result['Time'], errors='coerce')
    result = result.dropna(subset=['Time']).drop_duplicates().sort_values('Time', kind='stable')
    if result.empty:
        raise ValueError('No valid timestamps remain.')
    if result['Time'].duplicated().any():
        raise ValueError('Conflicting rows share the same timestamp.')
    return result.reset_index(drop=True)

def extract_features(raw):
    """One row per car; no labels, car numbers, dates or train IDs are features."""
    frame = clean_data(raw)
    cars = car_columns(frame)
    signals = {kind: {} for kind in ALIASES}
    for car, columns in cars.items():
        valid = pd.Series(True, index=frame.index)
        if 'ACV Information Valid' in columns:
            valid = frame[columns['ACV Information Valid']].astype(str).str.strip().str.lower().eq('valid')
        for kind, aliases in ALIASES.items():
            col = next((columns[a] for a in aliases if a in columns), None)
            values = frame[col] if col else pd.Series(np.nan, index=frame.index)
            if kind == 'mode':
                signals[kind][car] = values.astype(str).str.lower().str.contains('cool', na=False) & valid
            else:
                values = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan)
                # Broad engineering bounds, fixed before validation (degrees Celsius assumed).
                signals[kind][car] = values.where(values.between(-40, 80) & valid)
    indoor = pd.DataFrame(signals['indoor'])
    target = pd.DataFrame(signals['target'])
    outdoor = pd.DataFrame(signals['outdoor'])
    cooling = pd.DataFrame(signals['mode'])
    if (indoor.notna().sum() >= 10).sum() < 2:
        raise ValueError('At least two cars need 10 valid indoor temperature samples for peer comparison.')
    usable = indoor.where(cooling)
    peer = usable.median(axis=1)
    errors = (indoor - target).where(cooling)
    error_peer = errors.median(axis=1)
    records = []
    for car in cars:
        delta = (usable[car] - peer).dropna()
        error = errors[car].dropna()
        excess = (errors[car] - error_peer).dropna()
        def quantile(series, q):
            return float(series.quantile(q)) if len(series) else np.nan
        records.append({
            'peer_median': quantile(delta, .5), 'peer_p90': quantile(delta, .9),
            'peer_positive_fraction': float((delta > 1).mean()) if len(delta) else np.nan,
            'error_median': quantile(error, .5), 'error_p90': quantile(error, .9),
            'relative_error': quantile(excess, .5),
            'ambient_gap': quantile((indoor[car] - outdoor[car]).where(cooling[car]).dropna(), .5),
            'indoor_iqr': quantile(usable[car].dropna(), .75) - quantile(usable[car].dropna(), .25),
        })
    features = pd.DataFrame(records, index=list(cars))
    if features.isna().all().all():
        raise ValueError('No usable cooling observations found in this case.')
    return features
