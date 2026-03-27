# Testing Document

## 1. Scope of Testing

This project is tested with automated unit tests using `unittest` test classes and `pytest` as the test runner.  
The tests currently focus on the implemented core areas:

- dataset discovery and train/test splitting
- image preprocessing
- PCA fitting and projection
- nearest-neighbor classification
- pipeline training and prediction flow
- CLI commands (`train`, `predict`, `evaluate`)
- evaluation metrics and confusion matrix generation

The testing goal is to verify both algorithm correctness and expected runtime behavior of the current command-line workflow.

## 2. Unit Testing Coverage Report

Coverage is measured with branch coverage enabled.

Commands:

```bash
python3 -m coverage run --branch -m pytest src
python3 -m coverage report -m
```

Current report:

```text
Name                        Stmts   Miss Branch BrPart  Cover   Missing
-----------------------------------------------------------------------
src/facerec/cli.py             84      1      6      1    98%   138
src/facerec/data.py            43      0     24      0   100%
src/facerec/eval.py            26      0      8      0   100%
src/facerec/knn.py             33      0     16      0   100%
src/facerec/model_io.py        22      0      2      0   100%
src/facerec/pca.py             31      0     10      0   100%
src/facerec/pipeline.py        29      0      8      0   100%
src/facerec/preprocess.py      18      0      4      0   100%
-----------------------------------------------------------------------
TOTAL                         286      1     78      1    99%
```

At this stage, algorithm modules are fully covered by the current branch-report run, and the only uncovered line is the direct `__main__` execution path in the CLI module.

## 3. What Was Tested and How

### `data.py`

- detection of ORL-style image files from labeled folders
- deterministic split behavior with fixed random seed
- validation behavior for missing dataset root, non-directory root, empty directory, and invalid split settings

### `preprocess.py`

- output shape after grayscale + resize + flatten
- normalization range checks (`0.0` to `1.0`)
- deterministic preprocessing on identical inputs
- error behavior for missing files

### `pca.py`

- expected PCA output shapes
- projection output shape
- invalid component count handling
- feature dimension mismatch handling
- deterministic projection behavior

### `knn.py`

- nearest-neighbor prediction for Euclidean and cosine metrics
- invalid metric handling
- error handling for empty gallery and shape/type mismatches (query, gallery, labels)

### `pipeline.py`

- model training output structure
- projection in trained PCA space
- prediction output behavior on known-type inputs
- invalid training and prediction input handling (non-matrix inputs, non-vector labels, wrong query shape)

### `cli.py` + `model_io.py`

- `train` command creates a model artifact
- `predict` command returns label and distance
- `evaluate` command writes JSON report with accuracy and confusion matrix

### `eval.py`

- confusion matrix counts
- correctness of total/correct/accuracy values
- empty-input and length-mismatch validation behavior

## 4. Types of Inputs Used

Current tests use representative synthetic inputs:

- temporary ORL-like folder structures with multiple identity folders
- generated grayscale image files (`.pgm`) with clearly separated intensity distributions
- numeric matrices with cluster-like structure for PCA and KNN validation
- valid and invalid input shapes for branch/error testing
- deterministic random seeds for reproducibility
- path-level negative cases (missing files, wrong path types)

These inputs were selected to exercise both normal execution and failure branches.

## 5. How to Reproduce the Tests

Install dependencies:

```bash
python3 -m pip install -e '.[dev]'
```

Run unit tests:

```bash
python3 -m pytest src
```

Run branch coverage:

```bash
python3 -m coverage run --branch -m pytest src
python3 -m coverage report -m
```

Generate HTML coverage:

```bash
python3 -m coverage html
```

Open report:

- `_htmlcov/index.html`

## 6. Empirical/Additional Testing (Current Status)

The current empirical result is the coverage report above (branch-aware).  
Performance benchmarking and larger-scale robustness tests are not yet implemented in this phase.

Current automated test count:

- 47 tests passing (`python3 -m pytest src`)

Planned additions:

- dedicated integration tests over larger subsets of the face dataset
- selected invariant-style checks for preprocessing and embedding behavior
- simple timing comparisons between selected parameter settings (for example different PCA component counts)

## 7. Current Limitations

- The direct `__main__` line in the CLI module is not executed during tests.
- Testing currently uses small synthetic datasets for fast feedback.
- Large-scale and performance-oriented tests are still pending.

These limitations are planned to be addressed in subsequent batches while keeping quick unit-test feedback for daily development.
