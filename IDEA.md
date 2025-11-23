# THE LOGICAL FLOW OF METRIC OPTIMIZATION

```mermaid
flowchart TD

A[PCA / Preprocessing] --> B[EDA]

B --> C[Define target metric<br>Stratified split]
C --> D((Train))
C --> E((Test))

%% SVM Branch
D --> S1[Baseline SVM<br>High precision, low recall]
S1 --> S2[Hypothesis: imbalance → bad margin]
S2 --> S3[Tune class_weight]
S3 --> S4[Tune threshold<br>→ Improve recall]
S4 --> SOK[✔ Linear branch validated]

%% RF Branch
D --> R1[Random Forest]
R1 --> R2[Hypothesis: model biased by imbalance]
R2 --> R3[Apply SMOTE]
R3 --> R4[Apply Borderline-SMOTE]
R4 --> R5[Tune sampling_strategy]
R5 --> ROK[✔ Tree-based branch validated]

%% AE Branch
D --> A1[Autoencoder<br>Latent + Recon error]
A1 --> A2[Hypothesis: non-linear manifold → better anomaly signal]
A2 --> A3[Tune sampling_strategy or thresholds]
A3 --> AOK[✔ Representation branch validated]

%% Final
E --> F[Apply full pipeline<br>Estimate production performance]
F --> END[✔ Done]

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

