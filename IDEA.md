# THE LOGICAL FLOW OF METRIC OPTIMIZATION

```mermaid
    graph TD
    0[PCA transformed dataset] --> 1[Exploratory data analysis] --> A
    A[
        Choose **target metric**
        Data preparation
        Stratified splitting
    ] --> B((Train set))
    A --> C((Test set))
    B --> D[
        Baseline with **SVM**
        High precision + low recall
    ]
    D --> E[Tune class_weight with **balance**]

    E --> F[Tune **class_weight** for trade-off between recall and precision]

    F --> G[Tune decision boundary **threshold**]

    G --> H[✅]

    B --> L[Try a non-linear classifier **Random Forest**]

    L --> M[Use **SMOTE** to handle class imbalance]

    M --> N[Use **Borderline-SMOTE** to handle the caveat of the **SMOTE**]

    N --> O[Tune **sampling_strategy**]

    O --> P[✅]

    B --> Q[Define **Autoencoder** structure
    Add feature as **Latent space** and **Recon_error**
    ]

    Q --> R[Tune **sampling_strategy**]

    R --> S[✅]

    B --> T[Apply the full pipeline with tuned parameter to the final test set for a **metric estimation** in production]

    T --> V[✅]
```

# NOTES

## 👉 Directions:

- `TD` = top → down
- `LR` = left → right
- `BT` = bottom → top

## 🤨 Node shapes:

- `A[box]`
- `A(rounded)`
- `A((circle))`
- `A{diamond}`

