"""Checks schema normalization, cleaning, batching, and copied-package portability."""
from io import BytesIO
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
from ..features import load_input, extract_features, validate_input
from ..predict import predict

def main():
    module = Path(__file__).resolve().parents[1]
    sample = module / 'tests' / 'sample_input.xlsx'
    raw = load_input(sample)
    reference = extract_features(raw)
    reordered = raw.iloc[::-1, ::-1]
    pd.testing.assert_frame_equal(reference, extract_features(reordered))
    pd.testing.assert_frame_equal(reference, extract_features(pd.concat([raw, raw.iloc[:1]])))
    aliases = raw.rename(columns=lambda c: str(c).replace('Indoor Average Temperature', 'Passenger Cabin Temperature Detected Value').replace('ACV Control Temperature (Cooling)', 'Target Temperature Value').replace('Outside Temperature Sensor Reading', 'Fresh Air Temperature Detected Value'))
    pd.testing.assert_frame_equal(reference, extract_features(aliases))
    bad = raw.drop(columns=['Car 01 - Indoor Average Temperature'])
    try:
        validate_input(bad)
    except ValueError as exc:
        assert 'indoor' in str(exc)
    else:
        raise AssertionError('Missing required column accepted')
    invalid = raw.copy()
    for column in invalid.columns:
        if 'Indoor Average Temperature' in column:
            invalid[column] = np.inf
    try:
        extract_features(invalid)
    except ValueError as exc:
        assert 'valid indoor' in str(exc)
    else:
        raise AssertionError('All-invalid car accepted')
    uploads = []
    for name in ('first.xlsx', 'second.xlsx'):
        upload = BytesIO(sample.read_bytes())
        upload.name = name
        uploads.append(upload)
    result = predict(uploads)
    assert list(result.file_id) == ['first.xlsx', 'second.xlsx']
    assert result.ranked_cars.nunique() == 1
    print('PASS: ordering, duplicates, schema aliases, missing/invalid data, multiple uploads')
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / 'subsystems' / 'acv'
        shutil.copytree(module, destination, ignore=shutil.ignore_patterns('__pycache__'))
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        subprocess.run([sys.executable, '-m', 'subsystems.acv.tests.smoke_test'], cwd=directory, env=env, check=True)
    print('PASS: copied subsystem works in a fresh process outside original project')

if __name__ == '__main__':
    main()
