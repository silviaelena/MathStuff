# FedCorr: Multi-Stage Federated Learning for Label Noise Correction

> **A comprehensive guide to understanding FedCorr, a novel federated learning framework for handling heterogeneous label noise across distributed clients.**

---

## 📚 Table of Contents

- [Overview](#overview)
- [Paper Information](#paper-information)
- [Quick Start Guide](#quick-start-guide)
- [Documentation Structure](#documentation-structure)
- [Core Concepts](#core-concepts)
- [Technical Deep Dive](#technical-deep-dive)
- [Resources](#resources)

---

## Overview

This folder contains comprehensive documentation and analysis of **FedCorr**, a CVPR 2022 paper that addresses the critical challenge of handling heterogeneous label noise in federated learning environments.

**What is FedCorr?**  
FedCorr is a privacy-preserving, multi-stage framework that identifies and corrects mislabeled data across distributed clients without centralizing the data or making assumptions about noise models.

**Why is this important?**  
In real-world federated learning, different clients often have vastly different label quality due to varying annotator skills, biases, and hardware reliability. FedCorr solves this without violating privacy constraints.

---

## Paper Information

### Paper Metadata

**Authors:**
- Jingyi Xu¹* (jinyi_xu@mymail.sutd.edu.sg)
- Zihan Chen¹,²* (zihan_chen@mymail.sutd.edu.sg)
- Tony Q.S. Quek¹ (tonyquek@sutd.edu.sg)
- Kai Fong Ernest Chong¹† (ernest_chong@sutd.edu.sg)

*Equal contribution, †Corresponding author

**Affiliations:**
1. Singapore University of Technology and Design
2. National University of Singapore

**Publication:** CVPR 2022 (Computer Vision and Pattern Recognition)

**Links:**
- 📄 [arXiv Paper](https://arxiv.org/abs/2204.04677)
- 📄 [OpenReview](https://openreview.net/pdf?id=hIjPGwtWHAx)
- 💻 [GitHub Repository](https://github.com/Xu-Jingyi/FedCorr) (Official PyTorch Implementation)
- 📊 [IEEE Xplore](https://ieeexplore.ieee.org/document/9878965)

---

## Quick Start Guide

### New to This Topic?

If you're new to FedCorr or federated learning with label noise, we recommend reading in this order:

1. **Start here** → Read [Overview](#overview) and [Abstract](#abstract) (below) to understand the problem
2. **Understand the method** → Read [Core Concepts](#core-concepts) to grasp the three-stage approach
3. **Deep dive on LID** → Read [`local_intrinsic_dimensionality_LID_explained.md`](./local_intrinsic_dimensionality_LID_explained.md) to understand the key innovation
4. **Optional background** → Read [`principal_component_analysis_explained.md`](./principal_component_analysis_explained.md) for general dimensionality concepts (not used in FedCorr)
5. **Optional comparison** → Read [`pca_vs_lid_comparison.md`](./pca_vs_lid_comparison.md) to see how LID differs from PCA

### Already Familiar with FL?

Jump directly to:
- [Technical Deep Dive](#technical-deep-dive) - Detailed explanation of each stage
- [Implementation Considerations](#implementation-considerations) - What you need to build
- [GitHub Repository](https://github.com/Xu-Jingyi/FedCorr) - See the code

---

## Documentation Structure

This folder contains the following documents:

| Document | Description | Best For |
|----------|-------------|----------|
| **`README.md`** (this file) | Main overview and navigation | Getting started, quick reference |
| [`local_intrinsic_dimensionality_LID_explained.md`](./local_intrinsic_dimensionality_LID_explained.md) | Deep dive into LID with math | Understanding the core innovation |
| [`principal_component_analysis_explained.md`](./principal_component_analysis_explained.md) | Complete PCA tutorial (optional background) | Understanding dimensionality concepts |
| [`pca_vs_lid_comparison.md`](./pca_vs_lid_comparison.md) | Side-by-side comparison (optional) | Seeing how LID differs from PCA |
| [`corrections_after_reading_paper.md`](./corrections_after_reading_paper.md) | Documentation of errors fixed | Understanding what was wrong initially |

---

## Abstract

Federated learning (FL) is a privacy-preserving distributed learning paradigm that enables clients to jointly train a global model. In real-world FL implementations, client data could have label noise, and different clients could have vastly different label noise levels. 

Although there exist methods in centralized learning for tackling label noise, such methods do not perform well on heterogeneous label noise in FL settings, due to the typically smaller sizes of client datasets and data privacy requirements in FL.

In this paper, we propose **FedCorr**, a general multi-stage framework to tackle heterogeneous label noise in FL, without making any assumptions on the noise models of local clients, while still maintaining client data privacy.

### Key Approach

1. **Stage 1: Noisy Client and Label Identification**
   - FedCorr dynamically identifies noisy clients by exploiting the dimensionalities of the model prediction subspaces independently measured on all clients
   - Identifies incorrect labels on noisy clients based on per-sample losses
   - Proposes an adaptive local proximal regularization term based on estimated local noise levels to deal with data heterogeneity and increase training stability

2. **Stage 2: Finetuning and Label Correction**
   - Finetunes the global model on identified clean clients
   - Corrects the noisy labels for the remaining noisy clients after finetuning

3. **Stage 3: Full Training**
   - Applies the usual training on all clients to make full use of all local data

### Experimental Results

Experiments conducted on:
- CIFAR-10/100 with federated synthetic label noise
- Clothing1M (real-world noisy dataset)

**Results:** FedCorr is robust to label noise and substantially outperforms the state-of-the-art methods at multiple noise levels.

---

## Core Concepts

### The Three-Stage Framework

FedCorr operates in three distinct stages:

#### 🔍 Stage 1: Identification
**Goal:** Find noisy clients and their mislabeled samples

1. **Client-Level Detection**
   - Uses **Local Intrinsic Dimensionality (LID)** to identify noisy clients
   - Clean data → Low LID scores (predictions cluster tightly)
   - Noisy data → High LID scores (predictions spread diffusely)
   - Uses **cumulative LID** (sum across iterations) for robust separation
   - **Gaussian Mixture Model (GMM)** automatically separates clean vs noisy clients
   - 📖 *[Learn more about LID →](./local_intrinsic_dimensionality_LID_explained.md)*

2. **Sample-Level Detection**
   - Per-sample loss analysis on flagged clients only
   - High loss samples likely mislabeled
   - **Gaussian Mixture Model (GMM)** separates clean vs noisy samples

3. **Adaptive Regularization**
   - Adds **mixup data augmentation** during Stage 1
   - Adds proximal term based on estimated noise level
   - Stabilizes training on heterogeneous clients

#### 🔧 Stage 2: Correction
**Goal:** Refine model and fix wrong labels

1. **Selective Finetuning**
   - Train global model using ONLY clean clients
   - Produces more robust model without noise contamination

2. **Label Correction**
   - Use refined model to re-label noisy samples
   - Only correct samples identified in Stage 1

#### 🚀 Stage 3: Full Training
**Goal:** Maximize data utilization

- Train on ALL clients with corrected labels
- Standard federated learning (e.g., FedAvg)
- All data now contributes to learning

### Key Innovation: Local Intrinsic Dimensionality (LID)

> **Core Insight:** Models trained on clean data produce predictions with lower Local Intrinsic Dimensionality (LID) than models trained on noisy data. Clean predictions cluster tightly; noisy predictions spread diffusely.

**How it works:**
```
For each client:
  1. Make predictions P on local data (N × C matrix)
  2. For each prediction, find k=20 nearest neighbors in prediction space
  3. Compute LID: LID(x) = -1/k * Σᵢ log(rᵢ/rₘₐₓ)
  4. Average LID over all predictions → LID score
  5. Accumulate across T₁ iterations → Cumulative LID score
  6. Use GMM to separate clients: High cumulative LID → Noisy client
```

**Why this matters:**
- ✅ Privacy-preserving (only one scalar per round sent to server)
- ✅ No noise model assumptions needed
- ✅ Fast computation (k-NN search, k=20)
- ✅ Cumulative scores improve separation over training
- ✅ Theoretically grounded (prior work on LID and label noise)

📖 **[Complete mathematical explanation →](./local_intrinsic_dimensionality_LID_explained.md)**

---


## Technical Deep Dive

> **Note:** This section provides a high-level overview. For detailed mathematical formulations, see the individual topic documents.

### Stage 1: Identification (Detailed)

#### Noisy Client Detection via LID

**Mathematical Basis:**
- Prediction matrix P (N samples × C classes)
- For each prediction, find k=20 nearest neighbors in prediction space
- Local Intrinsic Dimensionality: `LID(x) = -1/k * Σᵢ log(rᵢ/rₘₐₓ)`
- LID score = average LID over all predictions
- **Cumulative LID** = sum of LID scores across T₁ iterations

**Algorithm:**
```python
for t in range(T1_iterations):
    for each client:
        predictions = model.predict(local_data)
        lid_score = compute_LID(predictions, k=20)
        cumulative_LID[client] += lid_score
        send lid_score to server

server:
    # Use Gaussian Mixture Model to separate
    gmm = GaussianMixture(n_components=2)
    gmm.fit(cumulative_LID)
    identify clients with higher mean → noisy clients
```

**Key Properties:**
- Privacy-preserving: Only scalar value shared per round
- Fast: O(N log N × C) with efficient k-NN
- Cumulative scores maintain separation even as training progresses
- GMM automatically finds threshold (no hyperparameter tuning)

📖 **[Complete mathematical treatment →](./local_intrinsic_dimensionality_LID_explained.md)**

#### Per-Sample Loss Analysis

Applied only to clients flagged as noisy:

1. Track loss for each sample during local training
2. Samples with high loss likely mislabeled
3. Use **Gaussian Mixture Model** to separate clean vs noisy samples

**Algorithm:**
```python
for each noisy_client k:
    # Track per-sample losses during training
    losses = [loss(model(xᵢ), yᵢ) for all samples]
    
    # Fit GMM with 2 components
    gmm = GaussianMixture(n_components=2)
    gmm.fit(losses.reshape(-1, 1))
    
    # High-loss component → noisy samples
    labels = gmm.predict(losses)
    identify component with higher mean → Dₖⁿ (noisy)
    other component → Dₖᶜ (clean)
    
    # Estimate noise level
    μₖ = |Dₖⁿ| / |Dₖ|
```

#### Adaptive Local Proximal Regularization

**Standard FedProx:**
```
Loss_client = Loss_task + (μ/2)||w_local - w_global||²
```

**FedCorr (Adaptive):**
```
Loss_client = LCE(f(X̃), Ỹ) + (μ̂ᵢ⁽ᵗ⁻¹⁾ * β / 2)||w_local - w_global||²

where:
  - LCE = Cross-entropy loss on mixup-augmented data
  - μ̂ᵢ⁽ᵗ⁻¹⁾ = estimated noise level of client i (from previous round)
  - β = 5 (base regularization strength)
  - X̃, Ỹ = mixup augmentation of batch
```

**Mixup Augmentation:**
```
For batch (X, Y):
  λ ~ Beta(α, α)  where α=1
  X̃ᵢ = λ·Xᵢ + (1-λ)·Xⱼ
  Ỹᵢ = λ·Yᵢ + (1-λ)·Yⱼ
```

**Effect:**
- Higher noise → Stronger regularization (stay closer to global)
- Lower noise → More freedom to adapt to local data
- Mixup reduces overfitting to noisy labels

### Stage 2: Correction (Detailed)

#### Selective Finetuning Process

```
1. Identify clean clients: C_clean = {clients with low dimensionality}
2. For T_finetune rounds:
     - Only clients in C_clean participate
     - Standard federated averaging
3. Result: Global model w_refined
```

**Why this works:**
- Clean data provides correct gradient signals
- No interference from noisy labels
- Model becomes more robust and accurate

#### Label Correction Mechanism

For each noisy client:

```python
for sample_i in noisy_samples:
    # Use refined model
    new_prediction = refined_model.predict(sample_i)
    new_label = argmax(new_prediction)
    
    # Correct if confident enough
    if max(new_prediction) > confidence_threshold:
        label[i] = new_label
```

**Safeguards:**
- Only correct high-confidence predictions
- Can use ensemble or majority voting
- Optionally keep original labels with low weight

### Stage 3: Full Training (Detailed)

```
1. All clients now have corrected labels
2. Run standard federated learning:
   - Local training on each client
   - Aggregate with FedAvg or FedProx
   - Repeat for remaining rounds
3. All data contributes to final model
```

**Benefits:**
- Maximizes data utilization
- Leverages corrected labels
- Simple and well-understood process

---

## Problem Statement & Motivation

### Challenges in Federated Learning with Label Noise

1. **Heterogeneous Label Noise**: Different clients can have vastly different label noise levels
2. **Small Client Datasets**: Typically smaller sizes of client datasets compared to centralized settings
3. **Data Privacy Requirements**: Cannot directly access or centralize client data
4. **Data Heterogeneity**: Non-IID data distribution across clients

### Why Centralized Methods Don't Work Well in FL

- Centralized label noise correction methods assume access to the entire dataset
- They require larger datasets for effective noise detection
- Privacy constraints prevent direct inspection of client data
- Heterogeneity across clients makes uniform noise handling ineffective

**The FedCorr Solution:**
- ✅ Client-level detection using dimensionality (no raw data needed)
- ✅ Works with small datasets per client
- ✅ Handles heterogeneous noise levels
- ✅ Privacy-preserving by design

---

## Key Contributions & Advantages

### Novel Contributions

1. **Local Intrinsic Dimensionality (LID) for Noisy Client Detection**
   - First to use LID of predictions for quality assessment in FL
   - Enables privacy-preserving identification of problematic clients

2. **Multi-Stage Framework**
   - Systematic approach: Identify → Correct → Retrain
   - Each stage builds on the previous one

3. **Adaptive Proximal Regularization**
   - Adjusts regularization strength based on estimated noise level
   - Balances local adaptation with global consistency

4. **No Noise Model Assumptions**
   - Works with any noise pattern (symmetric, asymmetric, class-dependent)
   - Doesn't require knowing noise rates

5. **Practical and Effective**
   - Tested on real-world noisy dataset (Clothing1M)
   - Substantially outperforms existing methods

### Advantages Over Existing Methods

| Feature | FedCorr | Centralized Methods | Standard FL |
|---------|---------|-------------------|-------------|
| **Privacy** | ✅ Preserved | ❌ Requires centralized data | ✅ Preserved |
| **Handles heterogeneous noise** | ✅ Yes | ❌ Assumes uniform noise | ❌ No noise handling |
| **Small client datasets** | ✅ Works well | ❌ Needs large datasets | ✅ Works well |
| **No noise assumptions** | ✅ Model-free | ❌ Often needs noise model | N/A |
| **Label correction** | ✅ Yes | ✅ Yes | ❌ No |

---

## Experimental Results

### Datasets Tested

1. **CIFAR-10**
   - 10 classes (airplanes, cars, birds, etc.)
   - Synthetic label noise at various levels (20%, 40%, 60%)
   - Federated setting with multiple clients

2. **CIFAR-100**
   - 100 classes
   - Synthetic label noise
   - Tests scalability to more complex classification

3. **Clothing1M**
   - Real-world noisy dataset
   - 1 million images with noisy labels from online shopping
   - 14 clothing categories
   - Realistic noise patterns from actual labeling errors

### Performance Highlights

**Key Results:**
- Substantially outperforms state-of-the-art FL methods at all noise levels
- Robust to heterogeneous noise (different clients with different noise rates)
- Maintains high accuracy even with 60% label noise
- Effective on real-world noisy data (Clothing1M)

**Compared Methods:**
- FedAvg (baseline)
- FedProx (proximal regularization)
- Other label noise methods adapted to FL setting

---

## Related Concepts & Background

### Federated Learning Basics

**Core Idea:**
```
Multiple clients train locally → Send updates to server → Server aggregates → Repeat
```

**Key Properties:**
- Data never leaves client devices
- Privacy-preserving by design
- Handles non-IID (heterogeneous) data

**Standard Algorithm (FedAvg):**
```
for each round t:
    server sends global model w_t to clients
    each client trains locally: w_t+1_i = w_t - η∇L(w_t)
    server aggregates: w_t+1 = Σ(n_i/n)·w_t+1_i
```

### Label Noise in Machine Learning

**Types of Label Noise:**

1. **Symmetric Noise:** Random flipping to any class
   - Example: 20% of labels randomly changed

2. **Asymmetric Noise:** Systematic errors
   - Example: "cat" often mislabeled as "dog", but not vice versa

3. **Instance-Dependent Noise:** Depends on features
   - Example: Ambiguous images more likely mislabeled

**Why It's Harmful:**
- Model learns incorrect patterns
- Overfits to noise
- Degraded generalization

### Local Intrinsic Dimensionality (LID)

**Important Clarification:**

FedCorr uses **Local Intrinsic Dimensionality (LID)**, NOT SVD-based dimensionality measures.

**Quick Summary:**
- Measures effective dimensionality using k-nearest neighbors in prediction space
- Clean data → Low LID (predictions cluster tightly around class prototypes)
- Noisy data → High LID (predictions spread diffusely)
- Formula: `LID(x) = -1/k * Σᵢ log(rᵢ/rₘₐₓ)`
- Uses **cumulative LID** across iterations for robust separation

📖 **[Full explanation with math →](./local_intrinsic_dimensionality_LID_explained.md)**

### Proximal Regularization (FedProx)

**Standard FedProx:**
```
min L(w) + (μ/2)||w - w_global||²
```

Keeps local model close to global model during training.

**FedCorr's Adaptation:**
Makes μ adaptive based on estimated noise level:
- High noise → High μ (stay close to global)
- Low noise → Low μ (adapt to local data)

---

## Implementation Considerations

### System Requirements

**For Clients:**
- Ability to compute predictions on local data
- k-NN computation capability (scipy, sklearn)
- Send single scalar to server

**For Server:**
- Aggregate dimensionality metrics
- Track which clients are clean vs noisy
- Coordinate multi-stage training

### Key Components to Implement

#### 1. LID Score Computation

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_lid_score(predictions, k=20):
    """
    Compute LID score for model predictions.
    
    Args:
        predictions: (N, C) array of softmax outputs
        k: number of neighbors (default 20 as in paper)
    
    Returns:
        lid_score: average LID over all predictions
    """
    N, C = predictions.shape
    
    # Find k nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(predictions)
    distances, indices = nbrs.kneighbors(predictions)
    
    # Remove self (distance 0)
    distances = distances[:, 1:]  # (N, k)
    
    # Compute LID for each point
    lids = np.zeros(N)
    for i in range(N):
        r = distances[i]
        r_max = r[-1]
        
        # Avoid log(0)
        epsilon = 1e-10
        r = np.maximum(r, epsilon)
        r_max = max(r_max, epsilon)
        
        # LID formula: -1/k * Σ log(rᵢ/rₘₐₓ)
        lids[i] = -np.mean(np.log(r / r_max))
    
    return np.mean(lids)

def identify_noisy_clients(cumulative_lids):
    """
    Identify noisy clients using Gaussian Mixture Model.
    
    Args:
        cumulative_lids: array of cumulative LID scores
    
    Returns:
        noisy_indices: indices of noisy clients
    """
    from sklearn.mixture import GaussianMixture
    
    X = cumulative_lids.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X)
    
    labels = gmm.predict(X)
    
    # Identify which cluster is "noisy" (higher mean)
    mean_0 = cumulative_lids[labels == 0].mean()
    mean_1 = cumulative_lids[labels == 1].mean()
    noisy_label = 0 if mean_0 > mean_1 else 1
    
    noisy = [i for i, l in enumerate(labels) if l == noisy_label]
    return noisy
```

#### 2. Per-Sample Loss Tracking

```python
def track_sample_losses_gmm(model, data, labels, num_epochs=5):
    """
    Track per-sample losses and identify noisy samples using GMM
    """
    from sklearn.mixture import GaussianMixture
    from collections import defaultdict
    
    loss_history = defaultdict(list)
    
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(zip(data, labels)):
            loss = compute_loss(model(x), y)
            loss_history[i].append(loss)
    
    # Average loss per sample
    avg_losses = np.array([np.mean(losses) 
                           for losses in loss_history.values()])
    
    # Fit GMM with 2 components
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(avg_losses.reshape(-1, 1))
    
    labels = gmm.predict(avg_losses.reshape(-1, 1))
    
    # High-loss component → noisy
    mean_0 = avg_losses[labels == 0].mean()
    mean_1 = avg_losses[labels == 1].mean()
    noisy_label = 0 if mean_0 > mean_1 else 1
    
    noisy_samples = [i for i, l in enumerate(labels) 
                     if l == noisy_label]
    
    return noisy_samples
```

#### 3. Adaptive Proximal Regularization with Mixup

```python
def mixup_data(x, y, alpha=1.0):
    """
    Apply mixup data augmentation.
    
    Args:
        x: batch of inputs
        y: batch of labels (one-hot)
        alpha: mixup parameter (default 1.0 as in paper)
    
    Returns:
        mixed_x, mixed_y: augmented batch
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.shape[0]
    index = np.random.permutation(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    
    return mixed_x, mixed_y

def adaptive_proximal_loss(local_model, global_model, 
                          x_batch, y_batch, noise_level, 
                          alpha=1.0, beta=5.0):
    """
    FedCorr loss with mixup and adaptive proximal term.
    
    Args:
        local_model: current local model parameters
        global_model: global model parameters
        x_batch, y_batch: training batch
        noise_level: estimated noise level (0-1)
        alpha: mixup parameter (default 1.0)
        beta: base regularization strength (default 5.0)
    """
    # Apply mixup
    x_mixed, y_mixed = mixup_data(x_batch, y_batch, alpha)
    
    # Cross-entropy loss on mixed data
    predictions = local_model(x_mixed)
    task_loss = cross_entropy(predictions, y_mixed)
    
    # Adaptive proximal term
    mu = noise_level * beta
    proximal_term = mu / 2 * torch.sum(
        (local_model.parameters() - global_model.parameters())**2
    )
    
    return task_loss + proximal_term
```

#### 4. Multi-Stage Training Loop

```python
def fedcorr_training(clients, model, config):
    """
    Complete FedCorr training loop with all three stages.
    """
    # ============ Stage 1: Identification ============
    print("Stage 1: Identification")
    cumulative_lids = np.zeros(len(clients))
    
    for iteration in range(config.T1):  # T1 = 5 for CIFAR-10
        # Sample clients without replacement (small fraction)
        client_order = np.random.permutation(len(clients))
        
        for client_id in client_order:
            client = clients[client_id]
            
            # Local training with mixup and adaptive proximal
            client.set_model(model)
            client.train_local(
                use_mixup=True, 
                alpha=1.0,
                beta=5.0,
                noise_level=client.estimated_noise
            )
            
            # Compute LID score
            predictions = client.predict_all()
            lid_score = compute_lid_score(predictions, k=20)
            cumulative_lids[client_id] += lid_score
            
            # Update global model (simplified FedAvg)
            model = aggregate_models([client.model])
    
    # Identify noisy clients using GMM
    noisy_clients = identify_noisy_clients(cumulative_lids)
    clean_clients = [i for i in range(len(clients)) 
                     if i not in noisy_clients]
    
    # Per-sample analysis on noisy clients
    for client_id in noisy_clients:
        client = clients[client_id]
        noisy_samples = track_sample_losses_gmm(
            model, client.data, client.labels
        )
        client.noisy_samples = noisy_samples
        client.estimated_noise = len(noisy_samples) / len(client.data)
    
    # ============ Stage 2: Finetuning & Correction ============
    print("Stage 2: Finetuning and Correction")
    
    # Identify very clean clients (noise < 10%)
    very_clean = [c for c in clean_clients 
                  if clients[c].estimated_noise < 0.1]
    
    # Finetune on clean clients only
    for round in range(config.T2):  # T2 = 500 for CIFAR-10
        model = federated_round([clients[i] for i in very_clean], model)
    
    # Correct labels on remaining noisy clients
    for client_id in noisy_clients:
        client = clients[client_id]
        for sample_id in client.noisy_samples:
            pred = model.predict_proba(client.data[sample_id])
            confidence = max(pred)
            
            # Only correct if confident (θ = 0.5 for CIFAR-10)
            if confidence > config.confidence_threshold:
                client.labels[sample_id] = np.argmax(pred)
    
    # ============ Stage 3: Full Training ============
    print("Stage 3: Full Training with Corrected Labels")
    
    for round in range(config.T3):  # T3 = 450 for CIFAR-10
        model = federated_round(clients, model)
    
    return model
```

#### 5. Label Correction Mechanism

```python
def correct_labels(model, data, labels, noisy_indices, 
                   confidence_threshold=0.8):
    """
    Correct labels using refined model
    """
    corrected_labels = labels.copy()
    
    for idx in noisy_indices:
        prediction = model.predict_proba(data[idx])
        confidence = max(prediction)
        
        if confidence > confidence_threshold:
            new_label = argmax(prediction)
            corrected_labels[idx] = new_label
    
    return corrected_labels
```

### Hyperparameters to Tune

From the paper (Table 7):

| Parameter | Description | CIFAR-10 | CIFAR-100 | Clothing1M |
|-----------|-------------|----------|-----------|------------|
| `T1` | Iterations in Stage 1 | 5 | 10 | 2 |
| `T2` | Rounds in Stage 2 (finetuning) | 500 | 450 | 50 |
| `T3` | Rounds in Stage 3 (full training) | 450 | 450 | 50 |
| `θ` | Confidence threshold for relabeling | 0.5 | 0.5 | 0.9 |
| `π` | Relabel ratio (top-π fraction) | 0.5 | 0.5 | 0.8 |
| `k` | Neighbors for LID | 20 | 20 | 20 |
| `α` | Mixup parameter | 1 | 1 | 1 |
| `β` | Base proximal regularization | 5 | 5 | 5 |
| Learning rate | | 0.03 | 0.01 | 0.001 |

**Note**: FedCorr uses the same total communication cost as baselines. Each iteration in Stage 1 involves all clients once (without replacement), which equals 10 communication rounds with fraction 0.1.

### Computational Complexity

**Per Client, Per Round:**

**LID Computation:**
- Find k nearest neighbors: O(N² × C) naive, or O(N log N × C) with KD-tree
- Compute LID for N points: O(N × k)
- **Total**: O(N log N × C) with efficient data structures

**For Typical Federated Settings:**
- CIFAR-10: N=500, C=10, k=20
  - With tree: 500 × log(500) × 10 ≈ 45K operations
- **Very fast!** (< 5% overhead per round)

**Server-Side:**

**GMM fitting:**
- Fit GMM on M clients: O(M × iterations × M)
- M typically 10-100 clients
- **Negligible** compared to model training

**Overall:**
- **Stage 1** (T₁ = 5 iterations): Standard FL cost + LID overhead (~5%)
- **Stage 2** (T₂ = 500 rounds): Reduced cost (only clean clients train)
- **Stage 3** (T₃ = 450 rounds): Standard FL cost

**Total overhead:** ~5-10% compared to standard FL

---

## Future Research Directions & Open Questions

### Potential Extensions

1. **Other Data Quality Issues**
   - Feature noise (corrupted inputs)
   - Missing data
   - Out-of-distribution samples
   - Backdoor attacks

2. **Theoretical Guarantees**
   - Convergence analysis
   - Sample complexity bounds
   - Privacy guarantees (differential privacy)
   - Robustness certificates

3. **Other FL Settings**
   - Cross-silo FL (few clients, large datasets)
   - Cross-device FL (many clients, tiny datasets)
   - Vertical FL (different features across clients)
   - Asynchronous FL

4. **Combination with Other Techniques**
   - FedNova (adaptive aggregation)
   - Personalized FL
   - Fair FL (fairness across clients)
   - Communication-efficient methods

5. **Scalability**
   - Very large numbers of clients (millions)
   - Continual learning / online adaptation
   - Dynamic client participation

### Open Questions

1. **Theoretical:**
   - Why exactly does dimensionality increase with label noise?
   - Can we prove convergence of the multi-stage approach?
   - What are the privacy implications of sharing dimensionality?

2. **Practical:**
   - How to choose hyperparameters automatically?
   - Can we detect other types of data quality issues?
   - How does it scale to very large models (LLMs)?

3. **Methodological:**
   - Can we make it work in one stage instead of three?
   - Alternative dimensionality measures?
   - Combining with semi-supervised or self-supervised learning?

4. **Comparison:**
   - How does it compare to robust aggregation methods?
   - Performance vs communication cost trade-offs?
   - When does FedCorr work better/worse than alternatives?

---

## Resources

### Official Resources

- 📄 **Paper (arXiv):** https://arxiv.org/abs/2204.04677
- 📄 **Paper (OpenReview):** https://openreview.net/pdf?id=hIjPGwtWHAx
- 💻 **Official Code:** https://github.com/Xu-Jingyi/FedCorr
- 📊 **IEEE Xplore:** https://ieeexplore.ieee.org/document/9878965
- 📑 **Supplementary Material:** Available on CVPR open access

### Documentation in This Folder

| Document | Lines | Description |
|----------|-------|-------------|
| `README.md` | ~1050 | This file - main overview and navigation |
| `local_intrinsic_dimensionality_LID_explained.md` | ~736 | Deep dive into LID concept with full math |
| `principal_component_analysis_explained.md` | ~795 | Optional: PCA tutorial for general dimensionality background |
| `pca_vs_lid_comparison.md` | ~576 | Optional: Comparison showing how LID differs from PCA |
| `corrections_after_reading_paper.md` | ~300 | Documentation of errors fixed after reading paper |

### Related Papers

**Federated Learning:**
- McMahan et al. (2017) - Communication-Efficient Learning (FedAvg)
- Li et al. (2020) - Federated Optimization (FedProx)

**Label Noise:**
- Zhang et al. (2018) - Generalized Cross Entropy
- Han et al. (2018) - Co-teaching
- Northcutt et al. (2021) - Confident Learning

**Dimensionality and Quality:**
- Roy & Vetterli (2007) - Effective Rank measures
- Vershynin (2018) - High-Dimensional Probability

### Suggested Reading Order

**For Beginners:**
1. This file (overview)
2. Section on "Core Concept" in `local_intrinsic_dimensionality_LID_explained.md`
3. GitHub repository examples

**For Deep Understanding:**
1. `local_intrinsic_dimensionality_LID_explained.md` (Core innovation - complete mathematical treatment)
2. `corrections_after_reading_paper.md` (Important errors to avoid)
3. `pca_vs_lid_comparison.md` (Optional: how LID differs from PCA)
4. `principal_component_analysis_explained.md` (Optional: general dimensionality background)
5. Original paper

**For Implementation:**
1. "Implementation Considerations" section (above)
2. Official GitHub repository
3. Experiment with CIFAR-10 example

---

## Citation

If you use FedCorr in your research, please cite:

```bibtex
@inproceedings{xu2022fedcorr,
  title={FedCorr: Multi-Stage Federated Learning for Label Noise Correction},
  author={Xu, Jingyi and Chen, Zihan and Quek, Tony QS and Chong, Kai Fong Ernest},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={10184--10193},
  year={2022}
}
```

---

## Quick Reference: Key Formulas

### Local Intrinsic Dimensionality (LID)

**Practical Formula (k-NN based):**
```
LID(x) = -1/k * Σᵢ₌₁ᵏ log(rᵢ(x) / rₘₐₓ(x))

where:
  - rᵢ(x) = distance to i-th nearest neighbor
  - rₘₐₓ(x) = distance to k-th neighbor
  - k = 20 (number of neighbors)
```

**Theoretical Definition (from paper):**

For a dataset drawn from distribution \( \mathcal{D} \) in \( \mathbb{R}^n \), let \( Y_x \) be the random variable representing the distance from point \( x \) to a randomly selected point \( y \) from \( \mathcal{D} \), with CDF \( F_{Y_x}(t) \).

```
LIDₓ(r) = lim_{ε→0} [log F_{Yₓ}((1+ε)r) - log F_{Yₓ}(r)] / log(1+ε)

LIDₓ = lim_{r→0} LIDₓ(r)
```

*Intuition:* LID measures how the volume of a neighborhood around \( x \) scales as the radius changes. Low LID means data lies on a lower-dimensional manifold (clean predictions cluster tightly).

### Cumulative LID Score
```
Cumulative_LID(client_k) = Σₜ₌₁ᵀ¹ LID_score(client_k, t)

Used for GMM-based client separation
```

**Why cumulative?** During training, LID overlap between clean/noisy clients increases due to (1) overfitting to noise and (2) label correction making noisy clients cleaner. Cumulative scores maintain better separation throughout Stage 1.

### Noise Model (ρ, τ)

FedCorr uses a federated noise model to simulate heterogeneous label noise:

```
For each client k:
  With probability ρ:
    μₖ ~ Uniform(τ, 1)  (client is noisy)
  With probability 1-ρ:
    μₖ = 0              (client is clean)

where:
  - ρ = system noise level (fraction of noisy clients)
  - τ = lower bound on noise level for noisy clients
  - μₖ = actual noise level (fraction of mislabeled samples)
```

When \( μₖ > 0 \), \( 100·μₖ\% \) of samples are randomly selected and assigned random incorrect labels.

**Example:** (ρ=0.6, τ=0.5) means 60% of clients are noisy, each with 50-100% label noise.

### Adaptive Proximal Loss (Stage 1)
```
L = LCE(f(X̃), Ỹ) + (μ̂ₖ⁽ᵗ⁻¹⁾ * β / 2) ||wₖ⁽ᵗ⁾ - wG⁽ᵗ⁻¹⁾||²

where:
  - LCE = cross-entropy on mixup data
  - μ̂ₖ⁽ᵗ⁻¹⁾ = estimated noise level of client k from previous round
  - β = 5 (base regularization strength)
  - wₖ⁽ᵗ⁾ = local model weights at round t
  - wG⁽ᵗ⁻¹⁾ = global model weights from previous round
```

**Effect:** Higher estimated noise → Stronger pull toward global model (more regularization).

### Mixup Augmentation

Mixup is applied during Stage 1 to reduce overfitting to noisy labels:

```
For each batch (Xb, Yb):
  λ ~ Beta(α, α)  where α = 1
  
  For pairs (xᵢ, yᵢ) and (xⱼ, yⱼ):
    x̃ᵢ = λ·xᵢ + (1-λ)·xⱼ
    ỹᵢ = λ·yᵢ + (1-λ)·yⱼ
  
  Train on (X̃b, Ỹb) instead of (Xb, Yb)
```

**Beta(1, 1) distribution:** This is the uniform distribution on [0,1], meaning λ is sampled uniformly between 0 and 1.

### Gaussian Mixture Model (GMM)

GMM is used twice in FedCorr for automatic threshold-free separation:

**1. Noisy Client Identification (at server):**
```
Input: cumulative_LID = [LID₁, LID₂, ..., LIDₙ]
Fit: GMM with 2 components (n_components=2)
Output: 
  - Component 0: mean μ₀, variance σ₀²
  - Component 1: mean μ₁, variance σ₁²
  
Noisy clients = component with higher mean
Clean clients = component with lower mean
```

**2. Noisy Sample Identification (at each noisy client):**
```
Input: per_sample_losses = [loss₁, loss₂, ..., lossₘ]
Fit: GMM with 2 components
Output:
  - Clean samples: component with lower mean loss
  - Noisy samples: component with higher mean loss
  
Estimated noise level: μ̂ₖ = |noisy_samples| / |total_samples|
```

**Mathematical formulation:**

GMM models data as a mixture of K Gaussian distributions:

```
p(x) = Σₖ₌₁ᴷ πₖ·𝒩(x | μₖ, Σₖ)

where:
  - πₖ = mixing coefficient (weight) for component k
  - 𝒩(x | μₖ, Σₖ) = Gaussian with mean μₖ and covariance Σₖ
  - Σₖ πₖ = 1, πₖ ≥ 0
```

Fitted using Expectation-Maximization (EM) algorithm. For FedCorr, K=2 always.

### Noisy Sample Relabeling
```
For noisy client k:
  1. Identify noisy samples Dₖⁿ using GMM on losses
  
  2. Select top-π·|Dₖⁿ| highest-loss samples as D̃ₖⁿ
  
  3. For each sample i in D̃ₖⁿ:
       prediction = model.predict(xᵢ)
       confidence = max(prediction)
       
       if confidence > θ:
         yᵢ = argmax(prediction)  // Relabel
  
where:
  - π = relabel ratio (0.5 for CIFAR-10/100, 0.8 for Clothing1M)
  - θ = confidence threshold (0.5 for CIFAR-10/100, 0.9 for Clothing1M)
```

---

## Summary: FedCorr in a Nutshell

**Problem:** Label noise in federated learning, especially when different clients have different noise levels.

**Solution:** Three-stage framework
1. **Identify** noisy clients (cumulative LID + GMM) and samples (per-sample loss + GMM)
2. **Correct** by finetuning on clean data and re-labeling noisy samples
3. **Retrain** on all corrected data

**Key Innovation:** Using **Local Intrinsic Dimensionality (LID)** of predictions as a privacy-preserving signal of data quality. Cumulative LID scores across iterations maintain clear separation between clean and noisy clients.

**Results:** Substantially outperforms existing methods on CIFAR-10/100 and Clothing1M. At extreme noise (ρ=0.8, τ=0.5), FedCorr achieves 90.59% accuracy vs. 72.00% for FedAvg.

**Why It Matters:** Enables robust federated learning in realistic scenarios with heterogeneous label noise while maintaining privacy.

---

*Last updated: February 1, 2026*  
*For questions or contributions, see the official [GitHub repository](https://github.com/Xu-Jingyi/FedCorr)*

