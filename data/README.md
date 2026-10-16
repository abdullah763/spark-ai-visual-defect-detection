# Dataset

Raw image data is intentionally excluded from this repository.

The original Spark AI assessment stated that a small dataset containing
**Normal** and **Defective** product images would be provided. That dataset was
not received within the available assessment window.

To complete the assessment without blocking the implementation, a public
industrial casting dataset was used as a clearly documented fallback.

## Dataset Source

**Casting Product Image Data for Quality Inspection**

Kaggle:

https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product

The dataset itself is not redistributed in this repository.

---

## Selected Dataset Portion

The project uses the original:

```text
casting_512x512/casting_512x512/
```

directories rather than the dataset's pre-generated augmented train/test
folders.

The selected class directories are:

```text
casting_512x512/
├── ok_front/
└── def_front/
```

Class mapping:

```text
ok_front  -> 0 -> Normal
def_front -> 1 -> Defective
```

Using the original images allows this project to control:

- train/validation/test splitting
- augmentation
- evaluation
- leakage prevention

---

## Dataset Audit

The selected dataset contains:

| Property | Result |
|---|---:|
| Total images | 1,300 |
| Normal | 519 (39.92%) |
| Defective | 781 (60.08%) |
| Image format | JPEG |
| Image size | 512 × 512 |
| Color mode | RGB |
| Corrupt images | 0 |
| Exact duplicate groups | 0 |

The dataset audit includes:

- image count
- class distribution
- image dimensions
- file format
- color mode
- corrupt-image detection
- exact SHA-256 duplicate detection
- representative visual inspection

The reusable audit utility is:

```text
src/audit_dataset.py
```

---

## Visual Observations

Visual inspection showed that many defects occur around the outer rim or edge
of the casting.

Observed defect patterns include:

- chipped edges
- broken edges
- rough or irregular rims
- small localized surface/edge defects
- subtle defect morphology

The objects are generally centered and photographed against relatively
consistent backgrounds.

Because many defects occur near the product boundary, aggressive cropping was
not used.

---

## Class Imbalance

Class distribution:

```text
Normal:     519  (39.92%)
Defective:  781  (60.08%)
```

This is approximately a `1.5:1` ratio.

The imbalance is moderate rather than severe, so the final baseline does not
use class weighting.

Instead, the project uses:

- stratified train/validation/test splitting
- class-specific precision
- class-specific recall
- F1-score
- confusion-matrix analysis
- false-positive / false-negative review

---

## Data Split

A deterministic stratified split with `random_seed=42` is used:

| Split | Images | Percentage |
|---|---:|---:|
| Train | 910 | 70% |
| Validation | 195 | 15% |
| Test | 195 | 15% |

The final held-out test set contains:

```text
Normal:      78
Defective:  117
Total:      195
```

The test set is not used for:

- gradient updates
- checkpoint selection
- hyperparameter tuning

The validation set is used for model selection.

---

## Preprocessing

Evaluation preprocessing:

1. Convert to RGB.
2. Resize to `224 × 224`.
3. Convert to tensor.
4. Apply ImageNet normalization.

Training additionally uses conservative augmentation:

- small random rotations
- mild brightness variation
- mild contrast variation

Aggressive transformations were avoided because they could alter or remove
small defect cues.

---

## Reproducing the Dataset Layout

Download the public dataset from Kaggle and identify these two directories:

```text
/path/to/casting_512x512/casting_512x512/ok_front
/path/to/casting_512x512/casting_512x512/def_front
```

Training can then be started with:

```bash
python -m src.run_training \
  --normal-dir "/path/to/ok_front" \
  --defective-dir "/path/to/def_front" \
  --epochs 3 \
  --batch-size 32 \
  --learning-rate 0.0001 \
  --num-workers 2
```

Evaluation:

```bash
python -m src.run_evaluation \
  --normal-dir "/path/to/ok_front" \
  --defective-dir "/path/to/def_front" \
  --checkpoint "models/best_model.pth" \
  --batch-size 32 \
  --num-workers 2
```

---

## Dataset Limitations

Known dataset limitations include:

- The official assessment dataset was not received.
- The fallback dataset has relatively consistent backgrounds and product
  positioning.
- Exact SHA-256 duplicate detection was performed, but perceptual
  near-duplicate detection was not.
- Product or batch identifiers were not available for group-aware splitting.
- Small defects may become less visible after resizing from `512 × 512` to
  `224 × 224`.
- The dataset may be easier than a real multi-camera or multi-production-line
  industrial environment.

These limitations are documented so that the reported model performance is
interpreted in the correct context.
