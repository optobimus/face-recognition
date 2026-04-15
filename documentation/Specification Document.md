# Specification Document

## Project

**Project name:** Face Recognition

**Study programme:** Bachelor's in Computer Science (TKT)

**Documentation language:** English

## Programming Language

The project will be implemented primarily in **Python**.

Python is a practical choice for this project because it provides strong support for numerical computing, matrix operations, image processing, and experimental evaluation needed fo the project. It is good for implementing and comparing classical machine learning algorithms without hiding the main ideas behind too much abstraction.

**Other languages for peer review:** Java, C

## Problem Statement

The project solves a classical face recognition problem: given an input face image, the program should identify which known person the face belongs to, or report that the face does not match the enrolled identities closely enough.

Basically, the program will learn a compact representation of faces from a labeled image dataset and then compare new faces against that representation. The main challenge here is that raw images are very high-dimensional, while the number of training examples is relatively small. Good face recognition therefore depends on choosing a representation that keeps identity-related information while reducing noise, like for example lighting variation.

## Core of the Project

The core of the project is an **algorithmic face recognition pipeline** built around dimensionality reduction and similarity search. The main implementation target is a  **Eigenfaces** approach based on **Principal Component Analysis (PCA)**, followed by a **nearest-neighbor classifier** in the lower-dimensional face space.

I will probably spend most of the development time on the following core tasks:

- preprocessing face images into a consistent numeric representation
- implementing PCA-based face embedding generation
- implementing recognition by nearest-neighbor search in the embedding space
- evaluating recognition accuracy and failure cases on a public dataset

## Planned Implementation

The ideal implementation is a transparent, educational system rather than a "black-box" deep learning product. The project should focus on algorithms that can be implemented, analyzed, and explained clearly according to our course spec.

### Main pipeline

1. Read labeled face images from a dataset.
2. Convert images to grayscale, normalize them, and resize them to a fixed resolution.
3. Flatten each image into a vector of pixel intensities.
4. Compute the mean face and center the training data.
5. Use PCA to project faces into a low-dimensional subspace.
6. Store each training face as an embedding vector together with its label.
7. For a query image, preprocess and project it in the same way.
8. Classify the query by comparing it to stored embeddings using nearest-neighbor search.

### Recommended dataset

A small and well-known academic dataset is the best fit for this project, for example:

- **AT&T Database of Faces (ORL)** for a manageable first implementation
- **Extended Yale Face Database B** to chheck for advanced robustness

These datasets are small enough for experimentation but still meaningful for comparing algorithms and preprocessing choices.

## Algorithms and Data Structures

### 1. Image preprocessing

The project will implement simple image preprocessing steps:

- grayscale conversion
- resizing to a fixed width and height
- intensity normalization
- optionally histogram equalization

These steps reduce irrelevant variation and make all inputs compatible with the later matrix operations.

**Data structures:**

- dense image matrices
- flattened feature vectors
- arrays of labels

### 2. Principal Component Analysis (Eigenfaces)

The main algorithm will be PCA, used to find a low-dimensional subspace that captures as much variance in the training faces as possible. Within face recognition, the basis vectors produced by this method are commonly called **eigenfaces**.

PCA is a good fit because raw face images may contain thousands of pixel values, making direct comparison noisy and inefficient. By projecting each face into a smaller subspace, the system can retain the strongest structure in the data while removing redundancy.

The implementation uses matrix-based PCA through covariance construction and custom eigendecomposition steps (power iteration with deflation) implemented with explicit matrix/vector operations.

**Data structures:**

- training matrix `X` of size `n x d`, where `n` is the number of images and `d` is the number of pixels per image
- mean vector of length `d`
- projection matrix containing the top `k` principal directions
- embedding matrix of size `n x k`

### 3. Nearest-neighbor classification

Once faces are represented as `k`-dimensional embeddings, the recognition step becomes a similarity-search problem. The simplest and most interpretable solution is **k-nearest neighbors (k-NN)**, with `k = 1` as a strong baseline.

Possible distance functions:

- Euclidean distance
- cosine distance / cosine similarity

This stage decides which known face is closest to the query in the PCA space.

**Data structures:**

- list or matrix of stored embeddings
- label array
- optionally a k-d tree for indexing low-dimensional embeddings

### 4. Optional future extension: LBPH

An optional future extension is **Local Binary Patterns Histograms (LBPH)** as a comparison baseline.  
The current implemented scope focuses on PCA/Eigenfaces + nearest-neighbor, and LBPH is intentionally kept outside the core implementation for now.

### 5. Evaluation

The project should evaluate:

- recognition accuracy
- confusion matrix
- per-class performance when possible
- effect of the number of principal components `k`
- effect of preprocessing choices

## Inputs and Their Use

The program will receive the following inputs:

- a labeled training dataset of face images
- a validation or test set of face images
- preprocessing parameters such as image size
- model parameters such as the number of principal components `k`
- classifier parameters such as the value of `k` in k-NN and the distance metric
- optionally a decision threshold for "unknown face" output

### How the inputs are used

- training images are used to compute the mean face, principal components, and stored embeddings
- labels are used to associate each embedding with an identity
- test images are projected into the same subspace and compared against stored embeddings
- parameters control the trade-off between speed, memory use, and recognition accuracy

## Expected Time and Space Complexities

The exact constants depend on implementation details, but the main behavior is clear.

Let:

- `n` = number of training images
- `m` = number of query or test images
- `d` = number of pixels per image after preprocessing
- `k` = number of retained principal components

### Preprocessing

For each image, resizing, normalization, and flattening are linear in the number of pixels.

- **Time:** `O(d)` per image, `O(nd)` for all training images
- **Space:** `O(d)` per image, `O(nd)` to store the full dataset

This is linear because every pixel must be read and transformed at least once.

### PCA training

The centered data matrix has size `n x d`. The implemented approach explicitly constructs a covariance matrix and extracts top eigenpairs with iterative methods.

For the current implementation:

- mean + centering: **Time** `O(nd)`, **Space** `O(nd)`
- covariance construction: **Time** `O(nd^2)`, **Space** `O(d^2)`
- top-`k` eigenpairs with power iteration + deflation: **Time** `O(k * t * d^2 + k * d^2)`, where `t` is iteration count

So a practical combined estimate is:

- **Time:** `O(nd^2 + ktd^2)`
- **Space:** `O(nd + d^2)`

The reason PCA is expensive is that it must process the entire training matrix and compute dominant directions of variation in a high-dimensional vector space.

### Projecting one image into PCA space

Projecting a centered image of dimension `d` onto `k` principal components requires a dot product with each retained component.

- **Time:** `O(dk)`
- **Space:** `O(k)` for the resulting embedding

### Brute-force nearest-neighbor query

If the query is compared against all `n` stored embeddings in `k` dimensions:

- **Time:** `O(nk)` per query
- **Space:** `O(nk)` to store the embeddings

This is linear in the number of stored faces because every embedding must be compared to the query.

### Optional k-d tree indexing

If embeddings are indexed in a k-d tree:

- **Build time:** `O(n log n)`
- **Average query time:** often around `O(log n)` in low dimensions
- **Worst-case query time:** `O(n)`
- **Space:** `O(n)`

However, k-d trees are less effective in higher-dimensional spaces. Since face embeddings may still have dozens of dimensions, this optimization is optional rather than guaranteed to help.

### Optional LBPH extension (future work)

If LBPH is added later as a baseline, feature extraction would be approximately linear in pixel count (`O(d)` per image), and brute-force matching would be linear in gallery size.

### Full test pass

If `m` test images are evaluated with PCA projection plus brute-force nearest-neighbor search:

- **Time:** `O(m(dk + nk))`

This comes directly from doing one projection and one full gallery scan for each test image.

## Why These Complexities Matter for the Project

The complexity analysis suggests several practical design choices:

- raw pixel-space comparison is easy to implement but scales poorly and is sensitive to noise
- PCA adds significant training cost, but it reduces query-time dimensionality and often improves recognition quality
- brute-force nearest-neighbor search is acceptable for small academic datasets and is simple to analyze
- data structures such as k-d trees are worth considering only if the reduced dimension is small enough
- public face datasets used in coursework are usually small enough that algorithm clarity is more important than extreme optimization

This makes PCA plus nearest-neighbor search a strong project core: it is mathematically meaningful, implementable within course scope and suitable for complexity analysis and empirical evaluation.

## Scope and Deliverables

The project aims to deliver:

- a working training and recognition pipeline for labeled face images
- at least one implemented recognition method based on PCA/Eigenfaces
- evaluation and analysis for the implemented PCA/Eigenfaces pipeline
- evaluation scripts and result summaries
- documentation describing the algorithms, design choices, and limitations

## Limitations and Risks

The following limitations are expected:

- classical face recognition methods are sensitive to pose, occlusion, and large lighting changes
- performance depends strongly on preprocessing and dataset quality
- small datasets make evaluation easier, but they also limit generalization
- face detection should not become the main project focus unless explicitly chosen, because it can overshadow the recognition algorithms themselves

For this reason, the project should keep the task focused on recognition from already cropped or clean face images.

## Sources Intended for Use

### Core algorithm sources

- Matthew Turk and Alex Pentland, *Eigenfaces for Recognition* (1991)
- Peter N. Belhumeur, Joao P. Hespanha, and David J. Kriegman, *Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection* (1997)
- Timo Ahonen, Abdenour Hadid, and Matti Pietikainen, *Face Description with Local Binary Patterns: Application to Face Recognition* (2006)

### Reference material

- Wikipedia: **Principal Component Analysis**
- Wikipedia: **k-Nearest Neighbors Algorithm**
- Wikipedia: **Local Binary Patterns**
- OpenCV documentation for face-recognition related preprocessing and LBPH interfaces
- course-approved or otherwise reliable linear algebra and machine learning references if needed during implementation

### Dataset sources

- AT&T Database of Faces
- Extended Yale Face Database B

## Summary

This project will ideally implement a classical face recognition system in Python using PCA/Eigenfaces as the main algorithmic core and nearest-neighbor search as the recognition method. The project is well suited to the algorithms and AI course because it combines linear algebra, data representation, similarity search, and experimental evaluation in a form that is feasible to implement and analyze. The central development effort should remain on the recognition algorithms themselves, not on user interface or deployment details.
