# Spark AI Visual Defect Detection

Computer Vision Engineer technical assessment solution for classifying
manufacturing product images as either **Normal** or **Defective**.

> Dataset-dependent results are intentionally pending because the assessment
> dataset has not yet been provided.

## 1. Problem

The system is designed to classify production-line product images into two
classes:

- Normal
- Defective

The final solution includes model training and evaluation, error analysis,
an inference API, validation and logging, Docker packaging, and deployment
instructions.

## 2. Current Approach

The initial architecture uses **ResNet18 pretrained on ImageNet** with the final
classification layer replaced for two-class classification.

ResNet18 was selected as a practical baseline because it is:

- lightweight
- well suited to transfer learning
- fast to train and infer
- easy to debug and maintain
- appropriate for a small image dataset

The final training strategy will be reviewed after inspecting the actual
assessment dataset.

## 3. Project Structure

```text
.
├── api/
│   └── main.py
├── data/
│   └── README.md
├── docs/
│   └── architecture.md
├── models/
├── outputs/
│   ├── error_analysis/
│   ├── metrics/
│   └── plots/
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   ├── transforms.py
│   └── utils.py
├── tests/
│   └── test_api.py
├── Dockerfile
├── requirements.txt
├── requirements-api.txt
└── README.md
```

## 4. Dataset Strategy

The assessment dataset has not yet been received.

Once provided, the following checks will be performed before training:

- total image count
- class distribution
- image dimensions and formats
- corrupt images
- representative visual inspection
- duplicate or near-duplicate images
- possible data leakage
- whether multiple images belong to the same physical product or batch

The final train/validation/test split and class-imbalance strategy will be based
on these findings.

## 5. Preprocessing and Augmentation

Current baseline preprocessing:

- RGB conversion
- resize to 224 × 224
- ImageNet normalization

Current training augmentation is intentionally conservative:

- small rotations
- mild brightness adjustment
- mild contrast adjustment

Augmentation will be revised after visually inspecting the actual dataset.

## 6. Model

Current baseline:

- Architecture: ResNet18
- Initialization: ImageNet pretrained weights
- Output classes: 2
- Classes: Normal / Defective

Training-specific hyperparameters will be finalized after dataset inspection.

## 7. Evaluation

The final test-set evaluation will include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- False-positive analysis
- False-negative analysis

Special attention will be given to the Defective class because missed defects
may be operationally important.

Results will be added after the assessment dataset is received and the final
model is trained.

## 8. Inference API

The application exposes:

### Health

```text
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### Prediction

```text
POST /predict
```

Input:

- JPEG
- PNG
- WEBP
- maximum upload size: 10 MB

Example response:

```json
{
  "predicted_class": "Defective",
  "confidence": 0.93
}
```

The API includes:

- file-type validation
- empty-file validation
- image decoding validation
- upload-size validation
- logging
- inference latency logging
- model availability handling
- error handling

## 9. Local Setup

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the API:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## 10. Tests

Run:

```bash
python -m pytest -v
```

The API test suite currently validates:

- health endpoint
- missing model handling
- unsupported file formats
- empty uploads
- invalid/corrupt images
- valid prediction response structure

## 11. Docker

Build:

```bash
docker build -t spark-defect-api .
```

Run:

```bash
docker run --rm -p 8000:8000 spark-defect-api
```

The Docker image uses CPU-only PyTorch for portable inference.

Training and development dependencies are kept separate from inference-only
dependencies to reduce the production image size.

## 12. Architecture

See:

```text
docs/architecture.md
```

## 13. Error Analysis

After final evaluation, false positives and false negatives will be reviewed
individually to identify patterns such as:

- subtle defects
- lighting variation
- reflections
- blur
- unusual viewpoints
- background correlations
- preprocessing issues

Only failure modes actually observed in the assessment dataset will be reported.

## 14. Known Limitations

Dataset-dependent limitations will be documented after the provided dataset is
analyzed.

Current known limitation:

- no final model can be trained or evaluated until the assessment dataset is
  provided

## 15. Future Improvements

Potential improvements will be considered only after establishing the baseline
and analyzing actual failure modes.
