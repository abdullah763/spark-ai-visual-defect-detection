# System Architecture

## Training Pipeline

```mermaid
flowchart TD
    A[Spark AI Dataset] --> B[Dataset Analysis]
    B --> C[Data Quality and Class Distribution Check]
    C --> D[Train / Validation / Test Split]

    D --> E[Training Set]
    D --> F[Validation Set]
    D --> G[Test Set]

    E --> H[Training Preprocessing and Augmentation]
    H --> I[Pretrained ResNet18]
    I --> J[Model Training]

    F --> K[Evaluation Preprocessing]
    K --> L[Validation]

    J --> L
    L --> M[Best Model Checkpoint]

    G --> N[Evaluation Preprocessing]
    M --> O[Final Test Evaluation]
    N --> O

    O --> P[Precision / Recall / F1]
    O --> Q[Confusion Matrix]
    O --> R[False Positive / False Negative Analysis]
```

## Inference Pipeline

```mermaid
flowchart LR
    A[Product Image] --> B[FastAPI]
    B --> C[File Validation]
    C --> D[Image Preprocessing]
    D --> E[ResNet18]
    E --> F[Softmax]
    F --> G[Predicted Class + Confidence]
    G --> H[JSON Response]
```

## Deployment

```mermaid
flowchart LR
    A[Docker Container] --> B[FastAPI / Uvicorn]
    B --> C[PyTorch CPU Inference]
    D[Trained Model Checkpoint] --> C
```

## Notes

- Training images may use augmentation after dataset inspection.
- Validation and test images use deterministic preprocessing only.
- The validation set is used for model selection.
- The test set is reserved for final evaluation.
- Class-imbalance handling will be decided after inspecting the provided dataset.
