# Rail Corrugation MVP

This self-contained subsystem classifies each 1-second, 10 kHz axle-box recording as `Normal`,
`Side I`, or `Side II`. Its public interface accepts Streamlit-style uploaded files and returns
the exact submission schema:

```python
from subsystems.rail_corrugation import predict

result = predict([uploaded_file])
```

## Data findings

- 272 labelled training CSVs and 68 unlabelled test CSVs are present.
- Every inspected file has 10,000 rows and the documented 129 columns: one binary rotational-speed
  channel followed by vibration/shock pairs for 8 positions on each of 8 cars.
- Labels are imbalanced: 234 `Normal`, 14 `Side I`, and 24 `Side II`.
- Representative files from every class contained no missing/infinite values or duplicate rows.
  Their signal ranges differed substantially, supporting robust distribution and spectral features.
- Test examples have the same shape and schema as training examples. No multiple-file grouping is
  needed: one uploaded CSV produces one prediction row.

## Method

Cleaning converts every field to numeric, rejects fully invalid channels, interpolates gaps within
each channel, fills edge gaps with that channel's median, and rejects remaining non-finite values.
Features respect the physical layout: odd axle positions form Side I and even positions Side II.
For vibration and shock on each side, the pipeline summarizes per-sensor standard deviation, RMS,
peak, peak-to-peak, kurtosis, crest factor, dominant frequency, and relative energy in six fixed
frequency bands. Signed Side I-minus-Side II contrasts help the small model localise a fault.
Speed-pulse duty cycle and transition counts are also included. Filenames are not model inputs.
Within the training partition only, a physically symmetric augmentation swaps Side I and Side II
feature blocks and fault labels. Validation recordings are never augmented or copied into training.

The validation split is a fixed (`random_state=42`) stratified 80/20 split of whole source files.
No rows or windows from one recording can appear in both partitions. This approximates evaluation
on unseen files; its limitation is that the documentation provides no train/run/route grouping,
so unknown repeated acquisition conditions cannot be explicitly grouped. The model is a
class-balanced Extra Trees classifier and is evaluated with the official macro F1 metric. After
validation, the saved inference model is refit on all 272 files.

The selected baseline scored **0.616 macro F1** on the fixed 55-file holdout (Normal F1 0.96,
Side I F1 0.00, Side II F1 0.89). The rare Side I class has only three holdout cases, so this
single-split estimate is noisy and highlights the principal MVP limitation; the artifact still
uses balanced class weights and is refit on all 14 available Side I cases.

## Commands

From the project root:

```bash
python -m subsystems.rail_corrugation.train
python -m subsystems.rail_corrugation.tests.smoke_test
```

Training is the only expensive step. Inference never retrains and resolves its artifact relative
to this module, so the folder can be copied as a unit. Runtime dependencies are `numpy`, `pandas`,
`scikit-learn`, and `joblib`.

## Input and output contract

`predict(uploaded_files)` requires a non-empty iterable of binary file-like `.csv` objects with a
`.name`. Each file must contain the exact documented 129 columns in order and at least 256 samples.
It returns `file_id,prediction`, with the basename (including `.csv`) and an allowed class label.
