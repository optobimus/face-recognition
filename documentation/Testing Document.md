# Testing Document

## 1. Scope of Testing

This project is tested with automated unit tests using `unittest` test classes and `pytest` as the test runner.  
The tests focus on the implemented core areas:

- dataset discovery and train/test splitting
- image preprocessing
- custom matrix operations
- PCA fitting and projection with custom matrix operations
- nearest-neighbor classification
- pipeline training and prediction flow
- CLI commands (`train`, `predict`, `evaluate`)
- evaluation metrics and confusion matrix generation
- an end-to-end CLI integration flow (`train -> predict -> evaluate`)

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
src/facerec/knn.py             29      0     16      0   100%
src/facerec/matrix_ops.py     194      8     90      8    94%   15, 26, 29, 36, 41, 45, 219, 241
src/facerec/model_io.py        22      0      2      0   100%
src/facerec/pca.py             43      0     16      0   100%
src/facerec/pipeline.py        29      0      8      0   100%
src/facerec/preprocess.py      18      0      4      0   100%
-----------------------------------------------------------------------
TOTAL                         488      9    174      9    97%
```

Most modules are at 100% branch coverage in this unit-test run.  
The remaining uncovered branches are in:

- direct `__main__` execution path in `cli.py`
- defensive/input-validation branches in `matrix_ops.py`

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
- custom covariance/eigenpair path behavior (through `fit_pca` and `transform_pca`)

### `matrix_ops.py`

- vector operations (`dot`, norm, scaling, distances, `argmin`)
- matrix operations (column means, centering, covariance, matrix-vector multiplication)
- eigenpair computation by power iteration + deflation
- top-k eigenpair extraction and edge-case behavior
- invalid input shape/type handling for all major operations

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

### Integration test (`src/integration/cli_flow_test.py`)

- creates a temporary ORL-style dataset with two identities
- runs CLI commands via subprocess (`train`, `predict`, `evaluate`)
- verifies model file generation, prediction output fields, and evaluation report content
- validates full command-chain behavior outside direct function-level unit tests

## 4. Types of Inputs Used

Current tests use representative synthetic inputs:

- temporary ORL-like folder structures with multiple identity folders
- generated grayscale image files (`.pgm`) with clearly separated intensity distributions
- numeric matrices with cluster-like structure for PCA and KNN validation
- numeric matrices/vectors for matrix operation correctness and edge-case validation
- valid and invalid input shapes for branch/error testing
- deterministic random seeds for reproducibility
- path-level negative cases (missing files, wrong path types)
- subprocess-level CLI invocation with temporary datasets and artifacts

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

Run integration flow test:

```bash
python3 -m integration.cli_flow_test
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

- 70 tests passing (`python3 -m pytest src`)
- integration CLI flow test passing (`python3 -m integration.cli_flow_test`)

Planned additions:

- additional integration scenarios over larger subsets of the face dataset
- selected invariant-style checks for preprocessing and embedding behavior
- simple timing comparisons between selected parameter settings (for example different PCA component counts)

## 7. Current Limitations

- The direct `__main__` line in the CLI module is not executed during tests.
- Some defensive branches in `matrix_ops.py` are still uncovered.
- Current integration coverage is still based on small synthetic datasets for fast feedback.
- Large-scale and performance-oriented tests are still pending.

These limitations are planned to be addressed in subsequent batches while keeping quick unit-test feedback for daily development.
