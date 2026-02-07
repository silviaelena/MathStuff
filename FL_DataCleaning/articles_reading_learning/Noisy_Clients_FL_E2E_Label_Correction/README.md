# Paper Analysis: Tackling Noisy Clients in Federated Learning with End-to-end Label Correction

## Paper Information
- **Title**: Tackling Noisy Clients in Federated Learning with End-to-end Label Correction
- **Authors**: Xuefeng Jiang, Sheng Sun, Jia Li, Jingjing Xue, Runhan Li, Zhiyuan Wu, Gang Xu, Yuwei Wang, Min Liu
- **Venue**: CIKM '24 (33rd ACM International Conference on Information and Knowledge Management)
- **Published**: October 21, 2024
- **DOI**: https://doi.org/10.1145/3627673.3679550
- **Citations**: 14 (as of Feb 2, 2026)
- **Code**: https://github.com/Sprinter1999/FedELC

## Classification Table

| Element | Answer |
|---------|--------|
| **ML or FL** | FL (Federated Learning) |
| **Handles fairness?** | No |
| **Outliers** | No |
| **Missing values** | No |
| **Label errors** | Yes - Primary focus of the paper. Addresses heterogeneous label noise across clients with varying noise rates |
| **Repair data?** | Yes - End-to-end label correction via backpropagation. Learns possible ground-truth labels for noisy clients' datasets |
| **Topics** | Federated Learning, Noisy Label Learning, Label Noise Correction, Noisy Client Detection, Data Quality, Privacy-Preserving Learning, Non-IID Data, Robust Training, Two-Stage Framework |
| **Datasets** | CIFAR-10, CIFAR-100, CIFAR-10-N (human annotation, 40.2% noise), CIFAR-100-N (human annotation, 40.2% noise), Clothing1M (1M images, 38.46% systematic noise) |
| **Code** | Yes - https://github.com/Sprinter1999/FedELC |
| **Implemented?** | Yes - 16 baseline methods implemented and compared including FedAvg, FedProx, FedExP, Krum, TrimmedMean, Median, Co-teaching, Co-teaching+, Joint Optim, SELFIE, Symmetric CE, DivideMix, Robust FL, FedLSR, FedRN, FedNoRo |

## Detailed Analysis

### 1. Problem Statement
The paper addresses a critical challenge in federated learning: **heterogeneous label noise across clients**. Different clients often have varying degrees of label noise in their datasets due to:
- Different annotation quality standards
- Limited budgets for dataset annotation
- Potential malicious clients providing low-quality data
- Crowdsourcing and machine-generated labels

### 2. Key Contributions

#### Two-Stage Framework (FedELC)
1. **Stage #1: Noisy Client Detection** (Warm-up phase - Tw rounds)
   - Uses FedAvg for initial model training
   - Applies **Gaussian Mixture Model (GMM)** with two components to classify clients
   - Computes fine-grained **class-wise loss** for each client
   - Divides clients into: **relatively clean group** (S_clean) and **relatively noisy group** (S_noisy)
   - Employs **logit adjustment** using local class distribution prior to handle class imbalance

2. **Stage #2: End-to-End Label Correction**
   - **Clean clients**: Continue with vanilla cross-entropy loss (Eq. 4)
   - **Noisy clients**: Apply end-to-end label correction with triplet supervision:
     - **Classification loss** (L_c): Uses learnable soft label distribution y_d instead of original label
     - **Compatibility loss** (L_comp): Ensures corrected labels don't deviate too far from original
     - **Entropy loss** (L_e): Encourages sharper, more confident predictions
   - **Distance-aware (DA) aggregation**: Weights client contributions based on model distance

#### End-to-End Label Correction Mechanism
- Initializes differentiable variable **ỹ** to model possible ground-truth label distribution
- Converts to soft label: **y_d = SoftMax(ỹ)**
- Updates both model parameters (θ) and label distribution (ỹ) via backpropagation
- After E local epochs, fuses model prediction with updated ỹ to estimate corrected labels

### 3. Label Noise Patterns Addressed

The paper handles three types of label noise:

#### a) Manually-Injected Label Noise
- **Symmetric flipping**: Original label flipped to any wrong class with equal probability
- **Asymmetric (pairwise) flipping**: Original label flipped only to specific wrong category
- **Mixed noise**: Half clients follow symmetric, half follow asymmetric
- Noise rates: Linear increase from 0 to ε (maximum noise rate)

#### b) Human Annotation Error
- **CIFAR-10-N-Worst**: 40.208% noise rate (Amazon Mechanical Turk labels)
- **CIFAR-100-N-Fine**: 40.200% noise rate (Human annotator labels)
- More realistic than synthetic noise

#### c) Systematic Label Noise
- **Clothing1M**: ~38.46% overall noise rate
- Real-world dataset with 1 million images
- Labels derived from web image captions (unstructured complicated noise)

### 4. Datasets Used

| Dataset | Training Samples | Test Samples | Classes | Base Model | Noise Pattern |
|---------|-----------------|--------------|---------|------------|---------------|
| CIFAR-10 | 50,000 | 10,000 | 10 | ResNet-18 | Synthetic (symmetric/asymmetric/mixed) |
| CIFAR-100 | 50,000 | 10,000 | 100 | ResNet-34 | Synthetic (symmetric/asymmetric/mixed) |
| CIFAR-10-N | 50,000 | 10,000 | 10 | ResNet-18 | Human annotation (40.2% noise) |
| CIFAR-100-N | 50,000 | 10,000 | 100 | ResNet-34 | Human annotation (40.2% noise) |
| Clothing1M | 1,000,000 | 10,000 | 14 | ResNet-50 (pre-trained) | Systematic web labels (38.46% noise) |

**Data Heterogeneity**: Dirichlet distribution with concentration parameter γ ∈ {0.5, 1.0} for Non-IID settings

### 5. Experimental Setup
- **Clients**: N = 100 total clients
- **Participation**: 10 clients selected per round (|St| = 10)
- **Local epochs**: E = 5
- **Batch size**: 64
- **Total rounds**: 120 (40 for Clothing1M)
- **Optimizer**: SGD with learning rate 0.01, momentum 0.9, weight decay 5e-4
- **Warm-up rounds**: Tw = 20
- **Hyperparameters**: α = 0.2, β = 0.5, η = 1000 (CIFAR-10/10-N/Clothing1M) or 5000 (CIFAR-100/100-N)

### 6. Baseline Methods (16 Total)

#### General FL Methods
1. **FedAvg** [38] - Vanilla federated averaging
2. **FedProx** [32] - Proximal term regularization
3. **FedExP** [17] - Parameter extrapolation

#### Robust Aggregation Methods
4. **TrimmedMean** [71] - Removes largest/smallest parameters
5. **Krum** [5] - Selects most central model
6. **Median** [30] - Median-based aggregation

#### Noisy Label Learning (NLL) Methods
7. **Co-teaching** [13] - Two peer networks with sample selection
8. **Co-teaching+** [72] - Enhanced co-teaching
9. **Joint Optim** [54] - Joint optimization of network and labels
10. **SELFIE** [50] - Robust sample selection
11. **Symmetric CE** [61] - Symmetric cross-entropy loss
12. **DivideMix** [27] - Combines co-teaching, MixUp, and MixMatch

#### Federated Noisy Label Learning (FNLL) Methods
13. **Robust FL** [69] - Global class-wise centroids regularization
14. **FedLSR** [18] - Local self-regularization via self-distillation
15. **FedRN** [20] - Reliable neighbor models exploitation
16. **FedNoRo** [63] - Two-stage framework with robust loss terms

### 7. Key Results

#### CIFAR-10 (Synthetic Noise, γ=1.0, Symmetric 0.0-0.4)
- **FedELC**: 76.81% Precision, 76.72% Recall (Best overall)
- **FedAvg**: 75.85% Precision, 73.58% Recall
- **FedNoRo**: 73.67% Precision, 73.52% Recall

#### CIFAR-10-N (Human Annotation, Non-IID γ=0.5)
- **FedELC**: 83.36% Precision, 82.25% Recall (Best)
- **FedNoRo**: 82.87% Precision, 81.68% Recall
- **FedAvg**: 81.37% Precision, 73.40% Recall

#### Clothing1M (Real-world Systematic Noise)
- **FedELC**: 71.64% Best Test Accuracy
- **RobustFL**: 71.77% (requires sensitive information exchange)
- **FedNoRo**: 70.52%
- **Joint Optim**: 70.31%

### 8. Technical Innovations

#### Noisy Client Detection
- Fine-grained **class-wise loss** instead of overall loss
- Two-component GMM for probabilistic client classification
- Avoids issues with coarse-grained division that fails on Non-IID data

#### Label Correction
- Differentiable label distribution learning
- Triplet supervision balances correction, compatibility, and confidence
- Approximately **×0.3 longer training time** for noisy clients (acceptable trade-off)

#### Distance-Aware Aggregation
```
D(i) = d(i) / max_j d(j)
where d(i) = min_j≠i ||w_i^t - w_j^t||²

Aggregation weight: n_i * e^(-D(i))
```
- Clean clients: D(i) = 0 → constant weight
- Noisy clients: Higher distance → Lower weight

### 9. Ablation Study Results

| Component | Logit Adjustment | DA Aggregation | Precision | Recall | F1-Score | Accuracy |
|-----------|------------------|----------------|-----------|--------|----------|----------|
| Full FedELC | ✓ | ✓ | 76.81 | 76.72 | 76.78 | 77.03 |
| Without DA | ✓ | × | 76.44 | 76.28 | 76.36 | 76.43 |
| Without Logit | × | ✓ | 75.50 | 75.33 | 75.41 | 75.56 |
| Without Both | × | × | 74.89 | 74.77 | 74.83 | 74.91 |

Both techniques contribute to performance improvement.

### 10. Key Observations

1. **Label Correction Effectiveness**: FedELC shows higher label estimation accuracy than Joint Optim across all noise scenarios
2. **Balanced Performance**: FedELC maintains good balance between precision and recall, while many methods show high precision but low recall
3. **Simple Methods Work**: General FL methods (FedAvg, FedProx, FedExP) surprisingly robust on human annotation noise
4. **Robust Aggregation Limitations**: Methods like Krum show poor performance as they only select one model
5. **Scalability**: Method degrades less than others when total classes increase (CIFAR-100)
6. **Privacy Preserved**: Unlike RobustFL, FedELC doesn't require exchanging sensitive class-wise centroids

### 11. Practical Applications

#### Data Quality Improvement
- Clients can compare original labels ŷ with estimated labels y_estimate
- Re-label inconsistent samples with **less human effort**
- Gradually refine local datasets over training

#### Detection Performance
- Successfully divides clients into clean/noisy groups
- Clean group has lower average noise rate
- For human annotation noise (~40%), method still detects meaningful separation

### 12. Limitations

1. **Computation Cost**: ~30% extra training time for noisy clients
2. **Hyperparameter Sensitivity**: Requires tuning α, β, Tw, and η (though method is relatively robust)
3. **Assumes Two Groups**: Binary division into clean/noisy may oversimplify in practice
4. **No Fairness Consideration**: Doesn't address fairness across clients
5. **No Outlier Detection**: Focuses on label noise, not data outliers
6. **No Missing Value Handling**: Assumes complete data

### 13. Related Concepts

#### Gaussian Mixture Model (GMM)
- Probabilistic model assuming data comes from mixture of Gaussian distributions
- Two-component GMM separates clients based on loss distribution
- Avoids hard threshold selection

#### Logit Adjustment
- Adjusts model output logits by log(π) where π is class prior
- Makes model treat each class equally despite class imbalance
- Effective in long-tailed and Non-IID scenarios

#### Non-IID Data
- Data distributions differ across clients
- Dirichlet distribution controls heterogeneity degree
- Lower γ → Higher heterogeneity

#### Cross-Entropy Variants
- **Vanilla CE**: Standard classification loss
- **Symmetric CE**: Includes model prediction in loss to handle noise
- **Entropy Regularization**: Encourages sharper predictions

### 14. Future Directions (from Paper)

1. Model pruning for efficiency
2. Gradient clipping for stability
3. Contribution estimation for incentives
4. Reliable data generation techniques
5. Additional regularization strategies
6. More application domains

### 15. Comparison with Related Work

| Method | Sample/Client Selection | Robust Aggregation | Robust Loss | Label Correction | Data Heterogeneity | Label Noise | Varying Noise Levels | Mixed Noise |
|--------|------------------------|--------------------|--------------|--------------------|--------------------|--------------|-----------------------|-------------|
| FedAvg | - | - | - | - | ✓ | - | - | - |
| FedNoRo | ✓ | - | ✓ | - | ✓ | ✓ | ✓ | - |
| **FedELC** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

FedELC is the **most comprehensive** method addressing all challenges.

### 16. Key Equations

**Stage #1 - Clean Client Training:**
```
L_cls = CE(p, ŷ, π)
where p = f(x; θ) + log(π)
```

**Stage #2 - Noisy Client Training:**
```
ỹ = K * ŷ  (initialize)
y_d = SoftMax(ỹ)  (soft label)

L_c = CE(p, y_d)  (classification)
L_comp = -Σ ŷ_m log(y_m^d)  (compatibility)
L_e = -Σ p_m log(p_m)  (entropy)

L = L_c + α * L_comp + β * L_e  (total loss)

Update: ỹ ← ỹ - η * ∇_ỹ L
```

**Distance-Aware Aggregation:**
```
w^(t+1) = Σ (n_i * e^(-D(i)) / Σ n_j * e^(-D(j))) * w_i^t
```

### 17. Implementation Details

- **Framework**: PyTorch on NVIDIA RTX 3090 GPUs
- **Precision**: Mixed precision training for acceleration
- **Evaluation**: Averaged over 3 seeds
- **Metrics**: Class-wise precision, recall, F1-score, accuracy (scikit-learn)
- **Normalization**: Dataset mean and standard deviation
- **Image Size**: 224×224 for Clothing1M, 32×32 for CIFAR

## Conclusion

**FedELC** is a comprehensive two-stage federated learning framework that:
1. ✅ **Detects** noisy clients with higher label noise using fine-grained class-wise loss and GMM
2. ✅ **Corrects** labels of noisy clients via end-to-end learning with triplet supervision
3. ✅ **Repairs** local datasets by providing estimated ground-truth labels
4. ✅ **Preserves privacy** without exchanging sensitive information
5. ✅ **Handles heterogeneity** across data distributions and noise levels
6. ✅ **Achieves superior performance** compared to 16 baseline methods

The paper makes significant contributions to practical federated learning deployment where data quality cannot be guaranteed, providing both robust training and data refinement capabilities.

