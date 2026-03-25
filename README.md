# face-recognition

## Development Commands

Install development dependencies:

```bash
python3 -m pip install -e '.[dev]'
```

Run tests:

```bash
python3 -m pytest src
```

Run branch coverage:

```bash
python3 -m coverage run --branch -m pytest src
python3 -m coverage report -m
```

Run linting:

```bash
python3 -m pylint src/facerec
```
