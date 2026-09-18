"""Run from the master project root: python -m subsystems.acv.tests.smoke_test."""
from io import BytesIO, StringIO
import importlib
import sys
from pathlib import Path
import tempfile
import pandas as pd

def main():
    module_dir = Path(__file__).resolve().parents[1]
    artifact = module_dir / 'artifacts' / 'model.json'
    assert artifact.is_file(), 'Missing model artifact'
    original = artifact.read_bytes()
    inference = importlib.import_module('subsystems.acv.predict')
    assert 'subsystems.acv.train' not in sys.modules, 'Inference imported training'
    assert artifact.read_bytes() == original, 'Import modified model artifact'
    print('PASS: prediction module imported without training')
    assert inference.load_saved_model()['coefficients'], 'Model coefficients missing'
    print('PASS: model artifact loaded')
    sample = module_dir / 'tests' / 'sample_input.xlsx'
    assert sample.is_file(), 'Missing portable sample input'
    upload = BytesIO(sample.read_bytes())
    upload.name = sample.name
    upload.seek(17)
    result = inference.predict([upload])
    assert upload.tell() == 17, 'Upload cursor was changed'
    assert isinstance(result, pd.DataFrame) and len(result) == 1
    assert list(result.columns) == ['file_id', 'ranked_cars']
    assert result.notna().all().all()
    assert result.file_id.iloc[0] == sample.name
    cars = result.ranked_cars.iloc[0].split('|')
    assert len(cars) == 8 and set(cars) == {f'{i:02}' for i in range(1,9)}
    pd.testing.assert_frame_equal(result, inference.predict([upload]))
    print('PASS: Streamlit-like upload accepted; deterministic DataFrame and exact schema')
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / 'predictions.csv'
        result.to_csv(csv, index=False)
        reread = pd.read_csv(csv, dtype=str)
        pd.testing.assert_frame_equal(result, reread)
    print('PASS: prediction CSV round trip succeeded')
    for bad in ([], None, [BytesIO(b'bad')], [upload, upload]):
        try:
            inference.predict(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid upload accepted')
    from subsystems.acv.model import rank_decay
    canonical = [f'{i:02}' for i in range(1,9)]
    for i, car in enumerate(canonical):
        assert rank_decay(canonical, car) == (8-i)/8
    assert rank_decay(canonical, '99') == 0
    print('PASS: invalid-input handling and official metric examples')
    print(result.to_string(index=False))

if __name__ == '__main__':
    main()
