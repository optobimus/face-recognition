# User Guide

## 1. Purpose

This program implements a classical face recognition pipeline based on Eigenfaces (PCA) and nearest-neighbor classification.

The program supports three main tasks:

- training a model from a labeled face dataset
- predicting the identity of a single image using a trained model
- evaluating the model on a test split and writing a JSON report

## 2. Requirements

- Python 3.12 or newer
- the project dependencies installed from the project root

Install the project and development dependencies:

```bash
python3 -m pip install -e '.[dev]'
```

After installation, the program is available through the `facerec` command.

All commands below should be run from the project root directory.

## 3. Expected Dataset Structure

The program expects an ORL-style dataset structure where each identity has its own subdirectory:

```text
dataset/
  person1/
    1.pgm
    2.pgm
    ...
  person2/
    1.pgm
    2.pgm
    ...
```

Supported image formats:

- `.pgm`
- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`

The folder name is used as the label for that identity.

## 4. Training a Model

Train a model with the default settings:

```bash
facerec train \
  --dataset-root dataset \
  --model-out data/model.npz
```

Useful training parameters:

- `--n-components`: number of PCA components, default `20`
- `--metric`: distance metric, either `euclidean` or `cosine`
- `--train-per-identity`: how many images per identity are used for training, default `6`
- `--seed`: random seed for deterministic train/test splitting, default `42`
- `--image-width`: resize width, default `64`
- `--image-height`: resize height, default `64`

Example:

```bash
facerec train \
  --dataset-root dataset \
  --model-out data/model.npz \
  --n-components 25 \
  --metric cosine \
  --train-per-identity 6 \
  --seed 42 \
  --image-width 64 \
  --image-height 64
```

Expected output:

```text
trained_samples=... components=... model=data/model.npz
```

## 5. Predicting One Image

Use a trained model to predict the identity of one image:

```bash
facerec predict \
  --model data/model.npz \
  --image dataset/person1/1.pgm
```

Expected output:

```text
label=person1 distance=...
```

The smaller the distance, the closer the image is to the nearest stored training embedding.

## 6. Evaluating the Model

Evaluate the trained model on the test split of the dataset:

```bash
facerec evaluate \
  --model data/model.npz \
  --dataset-root dataset \
  --report-out data/report.json \
  --train-per-identity 6 \
  --seed 42
```

Expected output:

```text
evaluated_samples=... accuracy=... report=data/report.json
```

The generated JSON report contains:

- `total`
- `correct`
- `accuracy`
- `confusion_matrix`

## 7. Notes About Splitting

The program creates the train/test split per identity.

This means:

- each identity must have more images than the value of `--train-per-identity`
- if an identity has too few images, the program raises an error
- using the same seed gives the same split again

## 8. Common Problems

### Dataset root does not exist

Make sure the path given to `--dataset-root` is correct and points to a directory.

### No supported image files found

Check that the identity folders actually contain supported image files.

### Not enough images for one identity

If `--train-per-identity` is too large, the program cannot leave a non-empty test split.
Use a smaller value or add more images for that identity.

### Model file does not exist

Make sure training has been run successfully before using `predict` or `evaluate`.

## 9. Output Files

- trained model: compressed NumPy file, usually `model.npz`
- evaluation report: JSON file, for example `report.json`

The model file stores the PCA state, training embeddings, labels, metric, and image size.

## 10. Current Limitations

- the program assumes images are already organized into identity folders
- the program does not perform face detection or cropping
- the program currently recognizes only among known identities in the training set
- large datasets may be slower because the implementation emphasizes clarity over heavy optimization
