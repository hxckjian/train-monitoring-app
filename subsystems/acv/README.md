# Subsystem

- Name: ACV (air conditioning and ventilation)
- Automatically chosen slug: `acv`
- Owner: to be filled in by the team member

# Task

Rank every car in each uploaded train case from most to least likely to have a
refrigerant leak. Exactly one car per case is faulty in the supplied dataset.
This is localisation/ranking, using a supervised regression score internally.
One input workbook produces one output row. Scores are not probabilities.

Sources inspected: `03_References/ACV/ACV_Subsystem_Info_Kit.md` (complete),
`01_Problem_Statement_3_Specifications.md` (complete),
`02_Datasets/ACV/Train_Labels.csv`, and `04_Example_Submission/acv_predictions.csv`.
The workspace initially contained documentation and datasets, with no existing
subsystem code, dependency manifest, or module convention.

# Input

Pass a nonempty **list** of binary file-like uploads, each named `.xlsx`.
Each file independently represents one train case. Multiple files are optional,
not required jointly. Filenames must be unique in a batch.

The loader inspects every sheet header and selects exactly one telemetry sheet.
Additional non-telemetry sheets are ignored; ambiguous telemetry sheets fail.
Required metadata: `Car model`, `Train number`, `Time`. Each workbook must identify
one train and contain eight distinct two-digit car IDs in `Car NN - parameter`
headers. Car IDs are read from headers, not column positions or filenames.

Required per-car parameters (either documented schema alias is accepted):

| Meaning | Accepted parameter names |
|---|---|
| Indoor temperature | `Indoor Average Temperature`, `Passenger Cabin Temperature Detected Value` |
| Cooling target | `ACV Control Temperature (Cooling)`, `Target Temperature Value` |
| Mode | `ACV Running Mode` |
| Optional ambient temperature | `Outdoor Average Temperature`, `Outside Temperature Sensor Reading`, `Fresh Air Temperature Detected Value` |

`ACV Information Valid` is respected when present. Other telemetry is allowed
but unused. Structural absence of required columns is rejected; missing readings
within present columns are handled. At least two cars need ten valid indoor
readings, and the case must have usable cooling observations. Error messages
include the uploaded filename. Upload content and cursor position are preserved.

# Data processing

Parse timestamps, discard invalid timestamps and exact duplicate rows, sort
chronologically, and reject conflicting duplicate timestamps. Convert temperature
text (`None`, `Invalid`, etc.) and infinities to missing. Mask readings when the
information-valid signal is not `Valid`. Use broad fixed temperature bounds of
-40 to 80 degrees Celsius, an engineering assumption rather than a learned cutoff.
No interpolation across shutdowns or gaps is performed.

Use observations whose running-mode text contains `cool`, including Automatic
Cooling and Full Cooling. Aggregate over the case; no separate window predictions
or invented window labels are needed. Within-case peer medians use only cooling
cars at the same timestamp and are also available during inference.

Eight features per car: median and 90th percentile cabin excess over the peer
median, fraction of excess above 1 degree, median and 90th percentile cabin minus
target, median target-error excess over peers, median cabin minus ambient, and
cooling cabin-temperature IQR. Ambient data may be missing. Cars lacking usable
cooling telemetry have missing features imputed using training medians, which
weakens their ranking evidence. They are still included in the required ranking.

Training and inference both call `features.py`. Feature medians, means, standard
deviations, order, coefficients, and intercept are saved together. No learned
statistics are refit on uploads. There are no train/car identity features.

# Validation

Leave one physical train out, grouped by `(Car model, Train number)`:

| Held-out group | Validation cases | Remaining cases train the fold |
|---|---|---|
| A:0620 | 01, 02 | 03, 04, 05, 06 |
| A:0619 | 03 | 01, 02, 04, 05, 06 |
| B:0208 | 04 | 01, 02, 03, 05, 06 |
| C:0407 | 05 | 01, 02, 03, 04, 06 |
| C:0408 | 06 | 01, 02, 03, 04, 05 |

All timestamps and cars of each group stay together. All learned preprocessing is
fitted only on training groups. This avoids shared-train leakage from cases 01
and 02 and measures transfer to unseen trains. The actual unlabelled test is
train A:0620 on a different date; this validation is more conservative than that
same-train setting. Unknown shared fleet or acquisition effects remain possible.

Official metric: `(n - (r - 1)) / n`, averaged equally over held-out **files**,
with zero for a missing true car. It is not averaged equally over folds of
different sizes. See `artifacts/validation.json` for scores, rankings and complete
fold membership. The metric test checks all eight ranks and the missing-car case.
No hyperparameter search was performed. Test labels are unavailable.

**Validation score: 0.9375** (official mean rank-decay). Top-1 accuracy, secondary:
4/6 = 0.6667. Per-case ranks for cases 01–06: **1, 1, 1, 3, 2, 1**.
With only five train groups, this estimate has substantial uncertainty.

# Model

Ridge regression on binary fault targets (1 faulty, 0 normal), alpha 10, with
training-fitted median imputation and standardization. Sorting its scores yields
the ranking. A small regularized linear baseline is appropriate for only 48 car
examples, six positive and 42 negative, across six cases. No random operations
are used; the recorded reproducibility seed is 42. Ties use ascending car ID.
The final model is fitted on all six labelled cases after validation.

# Artifacts

- `artifacts/model.json`: all inference parameters, feature order and format version.
  JSON avoids pickle compatibility and an unnecessary scikit-learn dependency.
- `artifacts/validation.json`: reproducibility versions and validation evidence;
  not required by inference.
- `eda.json` and `training_features.csv`: dataset audit and per-car features;
  not required by inference.
- `expected_output.csv`: illustrative schema only, not expected model correctness.
- `acv_predictions.csv`: inference on the supplied unlabelled test, when generated.

# Training

From the project root, with Python 3.11+ and dependencies installed:

```bash
python -m pip install -r subsystems/acv/requirements.txt
python -m subsystems.acv.train
```

After copying to a differently structured project, provide the dataset root:

```bash
python -m subsystems.acv.train --data-dir /path/to/ACV
```

This directory must contain `Train_Labels.csv` and `Train/*.xlsx`. Training never
uses the Test directory. Imports do not start training. Input datasets are read-only.

# Prediction interface

```python
from io import BytesIO
from pathlib import Path
from subsystems.acv.predict import predict

path = Path('subsystems/acv/tests/sample_input.xlsx')
uploaded_file = BytesIO(path.read_bytes())
uploaded_file.name = path.name
result = predict([uploaded_file])
result.to_csv('prediction.csv', index=False)
```

The same call accepts Streamlit UploadedFile objects without any Streamlit imports.
Artifacts resolve relative to the module, independent of working directory.

CLI (input may be a workbook or directory):

```bash
python -m subsystems.acv.predict --input 02_Datasets/ACV/Test --output subsystems/acv/acv_predictions.csv
```

Direct `python subsystems/acv/predict.py --input ... --output ...` also works.
The Info Kit mentions a top-level CLI requirement absent from the supplied current
specification; both forms are provided with explicit input and output paths.

# Smoke test

```bash
python -m subsystems.acv.tests.smoke_test
```

This checks import without training, artifact loading, a real BytesIO upload,
deterministic output, eight unique zero-padded cars, exact schema, CSV round trip,
invalid uploads, and metric examples. The portable sample is an unchanged copy of
case 06 (about 0.77 MB), the smallest original case. It is an integration fixture,
not independent accuracy evidence. No external raw dataset is needed for this test.
Exclude the raw sample from the judges' final submission, per the specification;
retain it in the development handoff so its smoke test remains self-contained.

Additional regression and portability checks:

```bash
python -m subsystems.acv.tests.regression_test
```

This tests column/row order, schema aliases, duplicate cleaning, invalid telemetry,
multiple uploads, and copying only this subsystem into a temporary master project
and running the smoke test in a fresh process.

# Output

Exactly `file_id,ranked_cars`, one row per uploaded workbook in upload order.
`file_id` is the source basename including extension. `ranked_cars` is a string
of all eight header car IDs separated by literal `|`, most likely first:

```csv
file_id,ranked_cars
example_case.xlsx,03|01|05|02|04|06|07|08
```

# Dependencies

Only NumPy, pandas and openpyxl; see `requirements.txt`. No notebook, scipy,
scikit-learn, joblib, Streamlit, network access or training is needed at inference.
The execution environment had no `python` on PATH, so verification used the
available bundled Python executable. On the master machine, ensure `python`
resolves to an environment with these three packages.

# Known limitations

Six cases are too few for a stable generalization estimate. Schema/fleet variation
is substantial and the rich telemetry schema appears only once. Some cars lack
usable readings. Missing telemetry cannot be repaired into diagnostic evidence.
Pressure and equipment-specific fault flags are excluded because they are absent
from most cases. Cooling status and temperature aliases are explicit assumptions;
unknown renamed schemas need a deliberate parser update. Celsius is inferred
from readings. No spectral features are used: temperature/control signals, gaps,
and differing sampling intervals make a simple case summary more appropriate for
this MVP. Long workbooks take time to parse. No claim is made about fault severity,
healthy trains, simultaneous faults, or probability calibration.

# Handoff summary

Copy the entire `acv` folder into `subsystems/acv` in the master project. Python
namespace packages allow this even when the master `subsystems` has no init file.
Files created: parsing/features, model/metric, training, prediction, requirements,
tests with portable input, expected output, artifacts, EDA and this README.
No unrelated subsystem or original dataset was edited.

Final artifacts are trained and saved. See `EDA.md` for findings and
`eda_features.svg` for per-car comparisons. Tested runtime: Python 3.12.14,
NumPy 2.3.5, pandas 3.0.1, openpyxl 3.1.5.

Verification: **PASS** for the root smoke command, regression suite, copied-folder
fresh-process smoke test, module CLI on the test case, and direct-script CLI on
the sample. Test output: `acv_test_case.xlsx,01|04|02|03|06|08|07|05`.
This is a model prediction, not verified test ground truth. Regenerate the EDA
plot and post-training test audit with `python -m subsystems.acv.report`.

The team leader still owns integrating this API into the master app, generating
official prediction downloads through that app, and preparing the required demo
video and predictions.zip. This package supplies the requested model subsystem.
