# Implementation Document

## 1. General Structure of the Program

My project is implemented as a modular Python package under `src/facerec`.  
The program currently supports a complete command-line workflow for training, prediction and evaluation.

Main modules:

- `data.py`: dataset discovery and deterministic train/test splitting for ORL-style folder structures
- `preprocess.py`: grayscale conversion, resize, normalization, and flattening of face images
- `matrix_ops.py`: custom matrix/vector operations implemented manually with Python loops (no matrix library routines)
- `pca.py`: PCA fitting and projection using custom covariance and eigenpair computation (Eigenfaces basis construction)
- `knn.py`: nearest-neighbor classification in embedding space (Euclidean and cosine distance)
- `pipeline.py`: orchestration layer for model training, embedding projection, and single-vector prediction
- `model_io.py`: model persistence to/from `.npz`
- `eval.py`: accuracy and confusion matrix calculation
- `cli.py`: user-facing commands (`train`, `predict`, `evaluate`)

Current program flow:

1. `train`: load dataset -> split -> preprocess training images -> fit PCA -> build gallery embeddings -> save model
2. `predict`: load model -> preprocess input image -> project to PCA space -> nearest-neighbor prediction
3. `evaluate`: load model -> split dataset test portion -> predict each test image -> produce JSON report with metrics

## 2. Achieved Time and Space Complexities

Let:

- `n` = number of training samples
- `m` = number of evaluation/query samples
- `d` = feature dimension (pixels after preprocessing)
- `k` = number of PCA components

### Preprocessing

- Time: `O(d)` per image
- Space: `O(d)` per image vector

Reason: each pixel is read and transformed once.

### PCA training (`fit_pca`)

- Time:
  - mean + centering: `O(nd)`
  - covariance construction: `O(nd^2)`
  - top-`k` eigenpairs by power iteration + deflation: `O(k * t * d^2 + k * d^2)`
  - total: `O(nd^2 + ktd^2)` (dominant terms)
- Space: `O(d^2 + nd)` (covariance matrix and centered data)

Reason: covariance is explicitly constructed as a dense `d x d` matrix, and each power-iteration step uses dense matrix-vector multiplication.

### PCA projection (`transform_pca`)

- Time: `O(dk)` per sample
- Space: `O(k)` per projected vector

Reason: each projection computes `k` dot products over `d` features.

### Nearest-neighbor prediction (`predict_nearest_neighbor`)

- Time: `O(nk)` per query
- Space: `O(nk)` for stored gallery embeddings

Reason: brute-force comparison of query embedding to each gallery embedding.

### Evaluation run (`evaluate`)

- Time: `O(m(dk + nk))`
- Space: `O(nk + mk)` for model embeddings and temporary projected query vectors

Reason: each test sample requires preprocessing/projection plus nearest-neighbor scan.

## 3. Performance and Complexity Comparison (Current)

The current implementation uses a simple brute-force nearest-neighbor search. For current course-scale datasets this is acceptable and keeps the implementation clear.

Compared to index-based approaches (for example k-d trees):

- brute-force: simpler and predictable, but linear in gallery size at prediction time
- index-based: potentially faster query time in lower dimensions, but extra build complexity and weaker benefits in moderately high dimensions

No alternative indexing structure is implemented yet, because my current scope prioritizes clarity and correctness of the core algorithm.

For PCA, the current implementation prioritizes algorithm transparency and explicit matrix operations.  
Compared to optimized library decompositions, this custom approach is typically slower on large dimensions, but it satisfies the project requirement to implement complex matrix operations manually.

## 4. Possible Shortcomings and Suggestions for Improvement

Current shortcomings:

- no dedicated acceleration structure for large galleries
- covariance-based PCA implementation is computationally heavy for large feature dimensions
- testing data is mostly synthetic in automated tests
- no open-set thresholding yet (for known/unknown rejection)
- no LBPH baseline yet for method comparison

Possible improvements:

- additional recognition baseline (LBPH) for algorithm comparison
- optional threshold-based rejection for unknown faces
- small performance test script over increasing dataset sizes
- second integration scenario with less separable class distributions

## 5. Support During Development and Use of LLMs

LLM usage:

- LLMs were used for development support
- model used: OpenAI GPT-5.3 (as development support)

How LLMs were used:

- learning about the algorithms and checking my understanding of them
- discussing implementation alternatives and trade-offs
- getting ideas for test structures and edge-case coverage 
- learning about how to implement the Pylint structure
- improving and updating documentation text

All code in the project is written by myself.  
LLMs were used only as support during development.  
LLM usage is reported here as required by the course.

## 6. Sources Used

Implementation-relevant sources:

- NumPy documentation (array storage/serialization and basic array handling)
- Pillow documentation (image loading, conversion, resizing)
- Python standard library documentation (`argparse`, `pathlib`, `json`, `subprocess`)
- course testing guidance for `unittest`, `pytest`, and `coverage`

Theoretical algorithm sources are listed in the Specification Document.
