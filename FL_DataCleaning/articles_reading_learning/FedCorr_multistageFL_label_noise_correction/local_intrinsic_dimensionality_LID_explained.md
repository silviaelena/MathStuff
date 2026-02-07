# Local Intrinsic Dimensionality (LID) in FedCorr

## Overview

**Local Intrinsic Dimensionality (LID)** is the key technique used in FedCorr for identifying which clients have noisy data without violating privacy constraints. FedCorr exploits the observation that models trained on noisy data produce predictions with **higher LID scores** than models trained on clean data.

> **Important**: This document corrects earlier misconceptions. FedCorr uses **LID (Local Intrinsic Dimensionality)**, NOT SVD-based stable rank or PCA. LID is a neighborhood-based measure from prior research.

---

## What is Local Intrinsic Dimensionality (LID)?

### Background

LID [Houle 2017] is a measure of the **intrinsic dimensionality of the data manifold** at each data point. Unlike other dimensionality measures, LID makes minimal assumptions on the data distribution beyond continuity.

**Key Insight**: At each datapoint, the number of neighboring datapoints grows with the radius of the neighborhood, and the growth rate serves as a proxy for "local" dimension.

### Geometric Intuition

The volume of an **m-dimensional Euclidean ball** grows proportionally to **rᵐ** when its radius is scaled by factor r.

For two m-dimensional balls with volumes V₁, V₂ and radii r₁, r₂:

```
V₂/V₁ = (r₂/r₁)ᵐ

⇒ m = log(V₂/V₁) / log(r₂/r₁)
```

This principle allows us to estimate the local dimension by observing how distances to neighbors scale.

---

## The LID Formula

### Mathematical Definition

Given a dataset in ℝⁿ, for any point x and its k nearest neighbors, the **Local Intrinsic Dimensionality** is:

```
LID(x) = -1/k * Σᵢ₌₁ᵏ log(rᵢ(x) / rₘₐₓ(x))
```

Where:
- **rᵢ(x)**: Distance from x to its i-th nearest neighbor
- **rₘₐₓ(x)**: Distance from x to its k-th (farthest) nearest neighbor  
- **k**: Number of neighbors to consider (FedCorr uses k=20)

### Intuition Behind the Formula

The formula computes the **maximum likelihood estimate** of the local dimension assuming the distances follow a particular distribution.

- **Low LID**: Neighbors are concentrated in a few directions → low-dimensional manifold
- **High LID**: Neighbors spread in many directions → high-dimensional manifold

---

## LID in FedCorr: From Theory to Practice

### Computing LID for Model Predictions

FedCorr doesn't compute LID on the raw data (which would violate privacy). Instead, it computes LID on the **model prediction space**.

#### Process:

**Step 1: Generate Predictions**

For a client with local dataset D = {(x₁, y₁), ..., (xₙ, yₙ)} and model f:

```
Prediction set: X = {f(x₁), f(x₂), ..., f(xₙ)}
```

Where each f(xᵢ) is a C-dimensional probability vector (softmax output).

**Step 2: Compute LID for Each Prediction**

For each prediction vector f(x) ∈ X:

1. Find k=20 nearest neighbors in X (using Euclidean distance in probability space)
2. Compute distances r₁, r₂, ..., r₂₀ to these neighbors
3. Apply LID formula: LID(f(x)) = -1/20 * Σᵢ₌₁²⁰ log(rᵢ/r₂₀)

**Step 3: Average Over Dataset**

The **LID score** for client k with local dataset Dₖ is:

```
LID_score(k) = (1/|Dₖ|) * Σₓ∈Dₖ LID(f(x))
```

This is a single scalar value that summarizes the dimensionality of the prediction space.

---

## The Core Insight: Clean vs. Noisy Data

### Clean Data → Low LID Scores

When a model is trained on **clean labels**:

**Prediction Characteristics:**
- **Confident**: Most probability mass on one class
- **Consistent**: Similar samples get similar predictions
- **Clustered**: Predictions cluster around C class prototypes

**Example predictions (CIFAR-10):**
```
Sample 1: [0.95, 0.02, 0.01, 0.00, 0.00, 0.01, 0.00, 0.01, 0.00, 0.00]
Sample 2: [0.01, 0.93, 0.02, 0.01, 0.01, 0.01, 0.00, 0.01, 0.00, 0.00]
Sample 3: [0.02, 0.01, 0.91, 0.02, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00]
```

**LID Analysis:**
- Predictions form tight clusters (one per class)
- Neighbors are close and in similar directions
- Local manifold is low-dimensional
- **Result: Low LID score** (e.g., 2-4)

### Noisy Data → High LID Scores

When a model is trained on **noisy labels**:

**Prediction Characteristics:**
- **Uncertain**: Probability spread across classes
- **Inconsistent**: Model confused by contradictory labels
- **Diffuse**: Predictions don't cluster tightly

**Example predictions (CIFAR-10 with 50% noise):**
```
Sample 1: [0.35, 0.18, 0.12, 0.09, 0.08, 0.06, 0.05, 0.03, 0.02, 0.02]
Sample 2: [0.15, 0.28, 0.15, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.02]
Sample 3: [0.22, 0.18, 0.16, 0.13, 0.09, 0.08, 0.06, 0.04, 0.02, 0.02]
```

**LID Analysis:**
- Predictions spread throughout the probability simplex
- Neighbors in many different directions
- Local manifold is high-dimensional
- **Result: High LID score** (e.g., 6-9)

---

## Why LID Works: Deep Learning Perspective

### Training Dynamics with Label Noise

Research [Ma et al. 2018] shows that training with label noise exhibits two phases:

**Phase 1: Dimensionality Compression (Early Training)**
- Model learns underlying true distribution
- Focuses on general features
- Representation space is low-dimensional

**Phase 2: Dimensionality Expansion (Later Training)**  
- Model starts overfitting to noisy labels
- Tries to memorize individual samples
- Representation space becomes high-dimensional

**FedCorr's Strategy**: Measure LID during training to catch this expansion.

### Experimental Evidence from Paper

From Figure 3 in the paper (CIFAR-10, 100 clients, ρ=0.6, τ=0.5):

**After 5 iterations of Stage 1:**
- Clean clients: LID scores form tight cluster around 3-4
- Noisy clients: LID scores spread from 5-8
- Clear bimodal distribution → GMM can separate them

**Correlation with noise level:**
- Clean (0% noise): LID ≈ 3.2
- Low noise (20%): LID ≈ 4.5
- Medium noise (50%): LID ≈ 6.8
- High noise (80%): LID ≈ 8.5

Strong linear relationship between local noise level and LID score!

---

## Cumulative LID Scores: The Key Innovation

### Why Not Use Single-Round LID?

**Problem 1: Overlap Increases During Training**

As training progresses:
- Models may overfit to noisy labels → LID increases for all clients
- Label correction reduces noise levels → Noisy clients become cleaner
- LID scores of clean and noisy clients start to overlap

**Problem 2: Instability**

Single-round measurements are noisy and can fluctuate.

### Solution: Cumulative LID Scores

FedCorr uses the **sum of LID scores** across all T₁ iterations of Stage 1:

```
Cumulative_LID(k) = Σₜ₌₁ᵀ¹ LID_score(k, t)
```

**Advantages:**

1. **Better Separation**: Cumulative scores maintain separation even when per-round scores overlap (see Figure 3 in paper)

2. **Stronger Linear Correlation**: Cumulative LID has stronger correlation with local noise level than single-round LID

3. **More Robust**: Averages out noise and fluctuations

4. **Historical Information**: Captures the entire training trajectory, not just final state

### Empirical Evidence

From Appendix C.4 (Figure 6):

**LID Scores (single round, after iteration 4):**
- Significant overlap between clean and noisy clients
- GMM struggle to separate

**Cumulative LID Scores (sum of iterations 1-5):**
- Clear separation maintained throughout
- GMM cleanly identifies two groups
- Near-perfect identification of noisy clients

---

## The Complete Algorithm

### Stage 1: Noisy Client Identification via Cumulative LID

```
Input: 
  - N clients, each with local dataset Dᵢ
  - Number of iterations T₁ (e.g., 5 for CIFAR-10, 10 for CIFAR-100)
  - k neighbors for LID (default k=20)

Initialize:
  - cumulative_LID = [0, 0, ..., 0]  (length N)

For t = 1 to T₁:
    // Standard FL round with all clients
    For each client i:
        1. Receive global model w⁽ᵗ⁾
        2. Train locally for E epochs
        3. Compute predictions: Xᵢ = {f(x) : x ∈ Dᵢ}
        4. Compute LID for each prediction:
           For each prediction vector p ∈ Xᵢ:
               - Find k=20 nearest neighbors in Xᵢ
               - Compute LID(p) = -1/k * Σⱼ log(rⱼ/rₖ)
        5. Average LID: LID_score(i) = mean{LID(p) : p ∈ Xᵢ}
        6. Update: cumulative_LID[i] += LID_score(i)
        7. Send model update and LID_score(i) to server
    
    Server aggregates model updates

// After T₁ iterations, identify noisy clients
Server:
    1. Fit Gaussian Mixture Model (GMM) with 2 components on cumulative_LID
    2. Identify component with higher mean → noisy clients
    3. Assign clients: Sₙ (noisy) and Sᶜ (clean)

Output: Sets Sₙ and Sᶜ
```

### Using Gaussian Mixture Models (GMM)

**Why GMM instead of fixed threshold?**

1. **Data-driven**: Adapts to the actual distribution of LID scores
2. **No hyperparameter tuning**: Automatically finds separation
3. **Handles varying noise levels**: Works across different noise settings
4. **Probabilistic**: Provides confidence in classifications

**GMM with 2 Components:**

```python
from sklearn.mixture import GaussianMixture

# Fit GMM
gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(cumulative_LID.reshape(-1, 1))

# Get cluster assignments
labels = gmm.predict(cumulative_LID.reshape(-1, 1))

# Identify noisy cluster (higher mean)
mean_0 = cumulative_LID[labels == 0].mean()
mean_1 = cumulative_LID[labels == 1].mean()
noisy_label = 0 if mean_0 > mean_1 else 1

# Noisy clients
noisy_clients = np.where(labels == noisy_label)[0]
clean_clients = np.where(labels != noisy_label)[0]
```

---

## Per-Sample Noise Identification

Once noisy clients are identified, FedCorr performs **per-sample analysis** on those clients only.

### Algorithm: Identify Noisy Samples via Loss

For each identified noisy client k ∈ Sₙ:

```
1. During local training, track per-sample loss:
   loss_history[i] = [L(f(xᵢ), yᵢ) for each epoch]

2. Compute average loss per sample:
   avg_loss[i] = mean(loss_history[i])

3. Fit GMM with 2 components on avg_loss

4. Identify high-loss component → noisy samples

5. Partition dataset:
   Dₖᶜ = clean samples
   Dₖⁿ = noisy samples
   
6. Estimate noise level:
   μₖ = |Dₖⁿ| / |Dₖ|
```

**Why GMM again?**

Same reasons - data-driven, automatic threshold, works across different noise distributions.

---

## Privacy Preservation

### What Information is Shared?

**From client to server:**
- Model updates (gradients or weights) - **standard in FL**
- LID score (single scalar per round) - **new, but minimal**

**NOT shared:**
- Raw data
- Individual predictions
- Per-sample losses
- Distance matrices
- Nearest neighbor indices

### Privacy Analysis

**LID score is privacy-preserving because:**

1. **Aggregate statistic**: Average over all predictions
2. **Scalar value**: Only one number per round
3. **Indirect measure**: Doesn't reveal specific predictions
4. **Local computation**: All sensitive operations happen on client

**Comparison to alternatives:**
- Sending individual predictions: |Dₖ| × C numbers → **Much worse**
- Sending prediction matrix: N × C matrix → **Privacy violation**
- LID score: 1 number → **Minimal additional leakage**

The LID score reveals only the "discriminability" of the local model, similar to how training loss provides indirect information about data quality.

---

## Implementation Details

### Computing LID in Python

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_lid_score(predictions, k=20):
    """
    Compute LID score for a set of predictions.
    
    Args:
        predictions: (N, C) array of softmax outputs
        k: number of neighbors (default 20 as in paper)
    
    Returns:
        lid_score: average LID over all predictions
    """
    N, C = predictions.shape
    
    # Find k nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(predictions)
    distances, indices = nbrs.kneighbors(predictions)
    
    # Remove self (distance 0)
    distances = distances[:, 1:]  # (N, k)
    
    # Compute LID for each point
    lids = np.zeros(N)
    for i in range(N):
        r = distances[i]  # distances to k neighbors
        r_max = r[-1]     # distance to farthest neighbor
        
        # Avoid log(0) - add small epsilon
        epsilon = 1e-10
        r = np.maximum(r, epsilon)
        r_max = max(r_max, epsilon)
        
        # LID formula
        lids[i] = -np.mean(np.log(r / r_max))
    
    # Average over all points
    lid_score = np.mean(lids)
    
    return lid_score

def identify_noisy_clients_gmm(cumulative_lids):
    """
    Identify noisy clients using Gaussian Mixture Model.
    
    Args:
        cumulative_lids: array of cumulative LID scores for all clients
    
    Returns:
        noisy_indices: indices of noisy clients
        clean_indices: indices of clean clients
    """
    from sklearn.mixture import GaussianMixture
    
    # Reshape for sklearn
    X = cumulative_lids.reshape(-1, 1)
    
    # Fit GMM with 2 components
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X)
    
    # Predict cluster labels
    labels = gmm.predict(X)
    
    # Identify which cluster is "noisy" (higher mean)
    mean_0 = cumulative_lids[labels == 0].mean()
    mean_1 = cumulative_lids[labels == 1].mean()
    noisy_label = 0 if mean_0 > mean_1 else 1
    
    # Get indices
    noisy_indices = np.where(labels == noisy_label)[0]
    clean_indices = np.where(labels != noisy_label)[0]
    
    return noisy_indices, clean_indices
```

### Full Stage 1 Implementation

```python
def fedcorr_stage1(clients, global_model, T1=5, k_neighbors=20):
    """
    FedCorr Stage 1: Identification using cumulative LID scores.
    
    Args:
        clients: list of client objects with data and train methods
        global_model: initial global model
        T1: number of iterations (5 for CIFAR-10, 10 for CIFAR-100)
        k_neighbors: k for LID computation (default 20)
    
    Returns:
        noisy_clients: list of noisy client indices
        clean_clients: list of clean client indices
        global_model: model after T1 iterations
    """
    num_clients = len(clients)
    cumulative_lids = np.zeros(num_clients)
    
    # Stage 1 iterations
    for t in range(T1):
        print(f"Iteration {t+1}/{T1}")
        
        # Shuffle clients (sample without replacement)
        client_order = np.random.permutation(num_clients)
        
        # Train each client once per iteration
        for client_idx in client_order:
            client = clients[client_idx]
            
            # Local training
            client.set_model(global_model)
            client.train_local()
            
            # Compute predictions
            predictions = client.predict_all()  # (N, C) array
            
            # Compute LID score
            lid_score = compute_lid_score(predictions, k=k_neighbors)
            cumulative_lids[client_idx] += lid_score
            
            # Update global model (simplified - in practice use FedAvg)
            global_model = client.get_model()
    
    # Identify noisy clients using GMM on cumulative LID
    noisy_clients, clean_clients = identify_noisy_clients_gmm(cumulative_lids)
    
    print(f"Identified {len(noisy_clients)} noisy clients and {len(clean_clients)} clean clients")
    
    return noisy_clients, clean_clients, global_model
```

---

## Computational Complexity

### Per Client, Per Round

**LID Computation:**
- Find k nearest neighbors: O(N² × C) with naive search
  - With KD-tree or Ball-tree: O(N log N × C)
- Compute LID for N points: O(N × k)
- **Total**: O(N² × C) or O(N log N × C) with efficient data structures

**For Typical Federated Settings:**
- CIFAR-10: N=500, C=10, k=20
  - Naive: 500² × 10 = 2.5M operations
  - With tree: 500 × log(500) × 10 ≈ 45K operations
- **Very fast!** (milliseconds on modern hardware)

### Server-Side

**GMM fitting:**
- Fit GMM on M clients: O(M × iterations)
- M typically 10-100 clients
- **Negligible** compared to model training

### Overall Overhead

**Stage 1** (T₁ = 5 iterations):
- Standard FL training: T₁ × (local training + communication)
- LID computation: T₁ × O(N log N × C) per client
- **Overhead**: < 5% of total training time

---

## Comparison to My Previous (Incorrect) Description

### What I Got Wrong

| Aspect | My Previous Description | Actual FedCorr |
|--------|------------------------|----------------|
| **Method** | SVD-based stable rank | LID (nearest neighbor-based) |
| **Formula** | rank = \\|P\\|²_F / \\|P\\|²_2 | LID = -1/k Σ log(rᵢ/rₘₐₓ) |
| **Computation** | SVD of prediction matrix | k-NN in prediction space |
| **What's measured** | Global matrix rank | Local dimensionality per point |
| **Aggregation** | Not specified | **Cumulative sum** across rounds |
| **Separation method** | Z-score threshold | **Gaussian Mixture Model** |

### Why This Matters

LID and stable rank are **fundamentally different**:

- **LID**: Neighborhood-based, captures local geometry, well-established in literature
- **Stable rank**: Matrix-based, captures global structure, different theoretical foundation

FedCorr specifically uses LID because:
1. Prior research showed LID increases with label noise [Ma et al. 2018]
2. LID is local → more sensitive to heterogeneous noise
3. LID has strong theoretical foundation in dimensionality estimation

---

## Theoretical Justification

### Why Does LID Increase with Label Noise?

**Clean Data Hypothesis:**

With clean labels, the model learns a **smooth decision boundary**. In prediction space:
- Points from the same class cluster together
- The manifold of predictions is low-dimensional
- Neighbors are close and aligned

**Noisy Data Effect:**

With noisy labels, contradictory training signals force the model to:
- Create more complex, fractured decision boundaries
- Produce predictions that don't cluster cleanly
- Spread predictions across the probability simplex

**Result**: The local neighborhoods in prediction space become more isotropic (spread in all directions) → Higher LID

### Connection to Prior Work

**Key Papers:**
1. Ma et al. (2018): "Dimensionality-Driven Learning with Noisy Labels" (ICML)
   - First to show LID increases with label noise
   - Demonstrated LID can detect adversarial examples

2. Houle (2017): "Local Intrinsic Dimensionality I" (SISAP)
   - Established LID as a robust dimensionality measure
   - Provided theoretical foundations

FedCorr adapts these insights to the federated learning setting.

---

## Experimental Validation

### From the Paper

**Dataset: CIFAR-10**
- 100 clients, IID partition
- Noise model: (ρ, τ) = (0.6, 0.5)
  - 60% of clients are noisy
  - Noisy clients have noise level uniformly sampled from [0.5, 1.0]

**Results after 5 iterations (Figure 3):**

**LID Score Distribution:**
- Clean clients: μ ≈ 3.8, σ ≈ 0.6
- Noisy clients: μ ≈ 6.5, σ ≈ 1.2
- **Overlap**: Significant

**Cumulative LID Distribution:**
- Clean clients: μ ≈ 19, σ ≈ 2
- Noisy clients: μ ≈ 32, σ ≈ 4
- **Clear separation**: GMM perfectly identifies groups

**Correlation with Local Noise Level:**
- Cumulative LID vs. noise level: R² ≈ 0.85 (strong linear relationship)
- Single-round LID vs. noise level: R² ≈ 0.45 (weak)

### Model-Agnostic Property

**Tested Architectures (Appendix C.2):**
- ResNet-18: Best performance (93.82% accuracy)
- VGG-11: Good performance (88.96% accuracy)
- LeNet-5: Lower performance (72.03% accuracy)

**All architectures showed clear LID separation between clean and noisy clients**, demonstrating that the method is model-agnostic.

---

## Limitations and Considerations

### 1. Requires Sufficient Local Data

**Issue**: If N << k (fewer samples than neighbors), LID estimation is unreliable.

**FedCorr Assumption**: Each client has at least 50-100 samples (typical for cross-silo FL).

### 2. Early Training Instability

**Issue**: Very early in training, all predictions may be uniform (high LID for everyone).

**FedCorr Solution**: Start measuring LID after initial rounds of training (e.g., after round 10).

### 3. Gradual Overfitting

**Issue**: As training progresses, even clean clients may overfit, increasing LID.

**FedCorr Solution**: Cumulative LID captures early-stage differences before overfitting dominates.

### 4. Computational Cost

**Issue**: k-NN search can be expensive for large N.

**Solution**: Use efficient data structures (KD-tree, Ball-tree) or approximate nearest neighbors.

### 5. Privacy-Utility Tradeoff

**Issue**: Sharing even aggregate statistics (LID) provides some information.

**Analysis**: LID reveals less than loss or gradient norms, but more than nothing. Acceptable tradeoff for the utility gained.

---

## Summary

### Key Takeaways

✅ **LID (Local Intrinsic Dimensionality)** - Neighborhood-based measure, NOT SVD-based rank  
✅ **Cumulative LID scores** - Sum across iterations, crucial for robust separation  
✅ **GMM for separation** - Data-driven, no manual threshold tuning  
✅ **Privacy-preserving** - Only scalar values shared  
✅ **Empirically validated** - Clear separation between clean and noisy clients  
✅ **Model-agnostic** - Works with different architectures  
✅ **Theoretically grounded** - Builds on established LID research  

### The Big Picture

LID-based noisy client detection is a **two-level hierarchy**:
1. **Coarse-grained** (Stage 1): LID identifies noisy clients
2. **Fine-grained** (Stage 1): Per-sample loss identifies noisy samples within those clients

This approach is:
- **Efficient**: Only detailed analysis on flagged clients
- **Privacy-preserving**: Minimal information shared
- **Effective**: High accuracy in experiments

---

## Connection to FedCorr's Multi-Stage Framework

### Stage 1: Identification (Where LID is Used)

```
Iterations 1 to T₁:
    ├─ Train all clients
    ├─ Compute LID scores → Cumulative LID
    ├─ Update global model
    └─ (Optional) Identify and correct noisy samples

After T₁ iterations:
    ├─ GMM on cumulative LID → Identify noisy clients
    ├─ GMM on per-sample losses → Identify noisy samples
    └─ Estimate local noise levels → Adaptive regularization
```

### Stages 2 & 3: Correction and Retraining

LID is only used in Stage 1. Stages 2 and 3 use the identified client sets:
- Stage 2: Finetune on clean clients only
- Stage 3: Train on all clients with corrected labels

---

## Further Reading

**Original LID Papers:**
- Houle (2017): "Local Intrinsic Dimensionality I: An Extreme-Value-Theoretic Foundation"
- Amsaleg et al. (2015): "Estimating Local Intrinsic Dimensionality"

**LID for Label Noise:**
- Ma et al. (2018): "Dimensionality-Driven Learning with Noisy Labels" (ICML)
- Ma et al. (2018): "Characterizing Adversarial Subspaces Using LID" (ICLR)

**FedCorr Paper:**
- Xu et al. (2022): "FedCorr: Multi-Stage Federated Learning for Label Noise Correction" (CVPR)

---

*This document provides the correct technical details from the actual FedCorr paper. Previous versions incorrectly described SVD-based methods.*
