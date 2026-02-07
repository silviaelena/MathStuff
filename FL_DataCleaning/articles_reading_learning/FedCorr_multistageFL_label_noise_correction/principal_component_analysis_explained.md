# Principal Component Analysis (PCA) - Comprehensive Guide

## What is PCA?

**Principal Component Analysis (PCA)** is a dimensionality reduction technique that transforms data into a new coordinate system where the axes (principal components) capture the maximum variance in the data.

### The Core Idea

Given high-dimensional data, PCA finds a lower-dimensional representation that preserves as much information (variance) as possible.

**Example**: If you have 100 features, PCA might reveal that 95% of the variance can be captured by just 10 new features (principal components).

---

## The Problem PCA Solves

### Curse of Dimensionality

High-dimensional data causes problems:
- **Visualization**: Can't plot 100D data
- **Computation**: Algorithms slow down with many features
- **Overfitting**: Models struggle with too many features
- **Redundancy**: Features often correlate (contain duplicate information)

### PCA's Solution

Transform data from D dimensions to k dimensions (k < D) while:
- ✅ Retaining maximum variance
- ✅ Removing redundancy
- ✅ Finding uncorrelated features
- ✅ Revealing hidden structure

---

## Mathematical Foundation

### Setup

**Data Matrix X**: N samples × D features

```
        Feature 1  Feature 2  ...  Feature D
Sample 1    x₁₁        x₁₂            x₁D
Sample 2    x₂₁        x₂₂            x₂D
  ...
Sample N    xₙ₁        xₙ₂            xₙD
```

Each row is a data point in D-dimensional space.

### Step-by-Step Algorithm

#### Step 1: Center the Data

Subtract the mean from each feature:

```
X_centered = X - μ

where μ = [μ₁, μ₂, ..., μD] = mean of each column
```

**Why?** PCA finds directions of maximum variance around the origin. Centering ensures the origin is at the data's center.

#### Step 2: Compute the Covariance Matrix

```
C = (1/N) X_centered^T · X_centered
```

This is a D × D matrix where:
- Diagonal elements: Variance of each feature
- Off-diagonal elements: Covariance between feature pairs

```
C = [σ²₁      cov(1,2)  ...  cov(1,D)]
    [cov(2,1)  σ²₂      ...  cov(2,D)]
    [  ...      ...     ...     ...   ]
    [cov(D,1) cov(D,2)  ...   σ²D    ]
```

**Properties**:
- Symmetric: C = C^T
- Positive semi-definite: all eigenvalues ≥ 0

#### Step 3: Compute Eigenvalues and Eigenvectors

Solve the eigenvalue equation:

```
C · v = λ · v
```

Where:
- **v**: Eigenvector (principal component direction)
- **λ**: Eigenvalue (variance captured by this component)

**Computation**: Use eigendecomposition

```
C = V Λ V^T
```

Where:
- **V**: Matrix of eigenvectors (each column is one eigenvector)
- **Λ**: Diagonal matrix of eigenvalues

#### Step 4: Sort by Eigenvalue

Order eigenvalues (and corresponding eigenvectors) from largest to smallest:

```
λ₁ ≥ λ₂ ≥ λ₃ ≥ ... ≥ λD ≥ 0
```

Corresponding eigenvectors:
```
v₁, v₂, v₃, ..., vD
```

These are your **principal components**:
- **PC1** = v₁ (captures most variance, λ₁)
- **PC2** = v₂ (captures second-most variance, λ₂)
- etc.

#### Step 5: Select Top k Components

Choose how many components to keep:

**Method 1**: Variance explained threshold
```
Keep smallest k such that: (λ₁ + λ₂ + ... + λₖ) / (λ₁ + λ₂ + ... + λD) ≥ 0.95
```
(Keep components explaining 95% of variance)

**Method 2**: Scree plot (look for "elbow")

**Method 3**: Fixed number (e.g., reduce to k=2 for visualization)

#### Step 6: Project Data onto Principal Components

Create the projection matrix with top k eigenvectors:

```
V_k = [v₁, v₂, ..., vₖ]  (D × k matrix)
```

Transform data:

```
X_reduced = X_centered · V_k  (N × k matrix)
```

Each row is now a k-dimensional representation of the original D-dimensional data point.

---

## Alternative: PCA via SVD

Instead of eigendecomposition of covariance matrix, directly use SVD:

### SVD of Centered Data

```
X_centered = U Σ V^T
```

Where:
- **U** (N × N): Left singular vectors (sample space)
- **Σ** (N × D): Diagonal matrix of singular values σᵢ
- **V** (D × D): Right singular vectors (feature space)

### Key Relationships

**Principal Components**:
```
PC_i = V[:, i]  (i-th column of V)
```

**Eigenvalues from Singular Values**:
```
λᵢ = σᵢ² / N  (variance = squared singular value / N)
```

**Projected Data**:
```
X_reduced = U[:, :k] · Σ[:k, :k]
```

Or equivalently:
```
X_reduced = X_centered · V[:, :k]
```

### Why SVD is Better

1. **Numerical stability**: More robust than computing C = X^T·X explicitly
2. **Efficiency**: Can use truncated SVD for large data
3. **Direct**: No need to form covariance matrix
4. **Equivalence**: Gives same principal components

---

## Geometric Interpretation

### Original Data Space

Data points live in D-dimensional space:
```
[x₁, x₂, ..., xD]
```

### After PCA

New coordinate system defined by principal components:

```
Original axes: e₁=[1,0,0,...], e₂=[0,1,0,...], ...
New axes: PC₁, PC₂, PC₃, ...
```

**Properties of new axes**:
1. **PC1**: Direction of maximum variance
2. **PC2**: Direction of maximum variance orthogonal to PC1
3. **PC3**: Direction of maximum variance orthogonal to PC1 and PC2
4. etc.

### Visualization (2D → 1D example)

Original data (2D):
```
   Feature 2
       |
       |    •  •
       |  •  • •
       | • • •• •
       |• •••• •
       |••••• • 
       |•••• •
       | •• •
       |  •
       |__________ Feature 1
```

First principal component (diagonal line):
```
   Feature 2
       |
       |    •  •
       |  •  • •  ← Data elongated this way
       | • • •• • /
       |• •••• • /
       |••••• • / ← PC1 direction
       |•••• • /
       | •• • /
       |  • /
       |___/______ Feature 1
```

Projection onto PC1 (1D):
```
   PC1 axis: ••••••••••••••••
            (All variance captured in one dimension!)
```

---

## Concrete Example: Student Scores

### Dataset

5 students, 4 subjects:

```
        Math  Physics  Chemistry  Biology
Alice    90     88        75        70
Bob      75     72        85        88
Carol    85     83        78        72
David    60     58        82        90
Eve      95     92        70        65
```

### Step 1: Center the Data

Mean: [81, 78.6, 78, 77]

Centered:
```
        Math  Physics  Chemistry  Biology
Alice    9     9.4       -3        -7
Bob     -6    -6.6        7        11
Carol    4     4.4        0        -5
David  -21   -20.6        4        13
Eve     14    13.4       -8       -12
```

### Step 2: Covariance Matrix

```
C = [  147.5   142.55   -66.5   -88.5  ]
    [ 142.55   138.04   -64.3   -85.6  ]
    [  -66.5    -64.3     35      40   ]
    [  -88.5    -85.6     40      62.5 ]
```

Notice: Math and Physics highly correlated (large positive covariance)

### Step 3: Eigendecomposition

```
λ₁ = 358.2  (explains 93% of variance)
λ₂ = 24.8   (explains 6% of variance)
λ₃ = 3.5    (explains 1% of variance)
λ₄ = 0.5    (explains 0% of variance)
```

Total variance: 358.2 + 24.8 + 3.5 + 0.5 = 387

**Eigenvectors (Principal Components)**:

```
PC1 = [0.62, 0.60, -0.31, -0.41]
      → High weight on Math/Physics (positive)
      → Negative weight on Chemistry/Biology
      → Represents "Quantitative vs. Qualitative" sciences

PC2 = [0.36, 0.35, 0.73, 0.47]
      → Positive weight on all subjects
      → Represents "Overall ability"

PC3, PC4 = (less important, usually ignored)
```

### Step 4: Project Data

```
X_reduced = X_centered · [PC1, PC2]
```

New 2D representation:
```
         PC1    PC2
Alice   19.2   -3.1
Bob    -22.5    8.4
Carol    7.8   -1.5
David  -38.1   10.8
Eve     24.6   -5.2
```

**Interpretation**:
- Alice & Eve: High PC1 (strong in Math/Physics, weak in Biology/Chemistry)
- Bob & David: Low PC1 (opposite pattern)
- David: High PC2 (overall higher scores)

### Dimensionality Reduction

Original: 4 features  
After PCA: 2 components (explaining 99% of variance)  
**Reduction**: 50% fewer dimensions, 1% information loss!

---

## Applications of PCA

### 1. Visualization

Reduce high-dimensional data to 2D or 3D for plotting:
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
plt.scatter(X_2d[:, 0], X_2d[:, 1])
```

### 2. Noise Reduction

If noise is in lower-variance components, removing them filters noise:
```python
pca = PCA(n_components=50)  # Keep top 50 components
X_denoised = pca.inverse_transform(pca.fit_transform(X))
```

### 3. Feature Engineering

Use principal components as new features:
```python
pca = PCA(n_components=0.95)  # Keep 95% variance
X_features = pca.fit_transform(X_train)
model.fit(X_features, y_train)
```

### 4. Compression

Store data more efficiently:
- Images: JPEG compression uses similar ideas
- Databases: Reduce storage requirements

### 5. Preprocessing

Remove multicollinearity before regression:
```python
pca = PCA(n_components=20)
X_uncorrelated = pca.fit_transform(X)
# Now run linear regression on X_uncorrelated
```

### 6. Anomaly Detection

Points with large reconstruction error are anomalies:
```python
X_reconstructed = pca.inverse_transform(pca.transform(X))
reconstruction_error = np.sum((X - X_reconstructed)**2, axis=1)
anomalies = X[reconstruction_error > threshold]
```

---

## Properties of Principal Components

### 1. Orthogonality

All principal components are perpendicular:
```
PCᵢ · PCⱼ = 0  for i ≠ j
```

This means they capture independent sources of variation.

### 2. Unit Vectors

Each principal component has length 1:
```
||PCᵢ|| = 1
```

### 3. Ordered by Importance

```
var(PC₁) ≥ var(PC₂) ≥ var(PC₃) ≥ ... ≥ var(PCD)
```

### 4. Linear Transformation

PCA is a linear method - each PC is a weighted sum of original features:
```
PC₁ = w₁₁·feature₁ + w₁₂·feature₂ + ... + w₁D·featureD
```

Cannot capture nonlinear relationships (use kernel PCA for that).

### 5. Variance Preservation

Total variance is preserved:
```
∑ᵢ var(featureᵢ) = ∑ⱼ var(PCⱼ)
```

We just concentrate it in fewer dimensions.

---

## Choosing the Number of Components

### Method 1: Explained Variance Ratio

Plot cumulative variance explained:

```python
import numpy as np
import matplotlib.pyplot as plt

pca = PCA()
pca.fit(X)

cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumsum)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--')
```

Choose k where cumsum ≥ 0.95

### Method 2: Scree Plot

Plot eigenvalues vs. component number:

```python
plt.plot(pca.explained_variance_)
plt.xlabel('Component Number')
plt.ylabel('Eigenvalue (Variance)')
```

Look for the "elbow" where the curve flattens.

### Method 3: Kaiser Criterion

Keep components with eigenvalue > average:
```
Keep PCᵢ if λᵢ > (λ₁ + λ₂ + ... + λD) / D
```

For standardized data (mean=0, var=1), this means keep λᵢ > 1.

### Method 4: Domain Knowledge

If you need to visualize: k = 2 or 3  
If you need speed: Choose k based on computation budget  
If you need interpretability: Choose k based on meaning

---

## Standardization vs. Centering

### Centering Only (What We Did)

```
X_centered = X - mean(X)
```

**Use when**: Features have similar scales

### Standardization (Scaling + Centering)

```
X_standardized = (X - mean(X)) / std(X)
```

**Use when**: Features have very different scales

**Example**: If one feature is "age" (0-100) and another is "salary" (0-1000000), standardize!

### Effect on PCA

- Without standardization: Features with larger variance dominate
- With standardization: All features equally weighted

**Rule of thumb**: Standardize unless you have a good reason not to.

---

## Limitations of PCA

### 1. Linearity

PCA only finds linear relationships. 

**Limitation**: Can't capture nonlinear structure  
**Solution**: Kernel PCA, t-SNE, UMAP

### 2. Scale Sensitivity

Affected by feature scales.

**Limitation**: Features with larger variance dominate  
**Solution**: Standardize data first

### 3. Interpretability

Principal components are linear combinations of all features.

**Limitation**: Hard to interpret what PC1 "means"  
**Solution**: Examine component loadings carefully

### 4. Assumes Gaussian Structure

Works best when data is roughly Gaussian.

**Limitation**: May not work well for highly skewed data  
**Solution**: Transform data (log, box-cox) before PCA

### 5. Outliers

Sensitive to outliers.

**Limitation**: Outliers can skew principal components  
**Solution**: Remove outliers or use robust PCA

### 6. All Components Have Meaning

Even small components might be important for some tasks.

**Limitation**: Dimensionality reduction might remove signal  
**Solution**: Test different k values, validate on downstream task

---

## Implementation in Python

### Using NumPy (Manual)

```python
import numpy as np

def pca_manual(X, n_components):
    """
    Manual PCA implementation
    
    Args:
        X: (N, D) data matrix
        n_components: number of components to keep
    
    Returns:
        X_reduced: (N, n_components) transformed data
        components: (n_components, D) principal components
        explained_var: variance explained by each component
    """
    # 1. Center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    # 2. SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # 3. Components are rows of Vt (columns of V)
    components = Vt[:n_components, :]
    
    # 4. Explained variance
    explained_var = (S**2) / (X.shape[0] - 1)
    explained_var = explained_var[:n_components]
    
    # 5. Transform data
    X_reduced = X_centered @ components.T
    
    return X_reduced, components, explained_var

# Usage
X_reduced, components, var = pca_manual(X, n_components=2)
```

### Using scikit-learn (Easy)

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)  # or n_components=0.95 for variance threshold
X_reduced = pca.fit_transform(X_scaled)

# Inspect results
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Components shape: {pca.components_.shape}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

# Inverse transform (reconstruct)
X_reconstructed = pca.inverse_transform(X_reduced)
X_original_scale = scaler.inverse_transform(X_reconstructed)
```

### Incremental PCA (For Large Data)

```python
from sklearn.decomposition import IncrementalPCA

# Process data in batches
ipca = IncrementalPCA(n_components=50, batch_size=100)
for batch in data_batches:
    ipca.partial_fit(batch)

X_reduced = ipca.transform(X)
```

---

## Common Pitfalls

### 1. Forgetting to Center/Standardize

```python
# BAD
pca = PCA()
X_reduced = pca.fit_transform(X)  # Forgot to standardize!

# GOOD
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA()
X_reduced = pca.fit_transform(X_scaled)
```

### 2. Not Saving the PCA Object

```python
# BAD
pca = PCA(n_components=2)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = PCA(n_components=2).fit_transform(X_test)  # Wrong! New PCA!

# GOOD
pca = PCA(n_components=2)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)  # Use same PCA fitted on train
```

### 3. Applying PCA Before Train-Test Split

```python
# BAD - Data leakage!
X_reduced = pca.fit_transform(X)
X_train, X_test = train_test_split(X_reduced)

# GOOD
X_train, X_test = train_test_split(X)
pca = PCA()
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)
```

---

## Advanced Topics

### Kernel PCA

Extends PCA to nonlinear relationships using kernel trick:

```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_reduced = kpca.fit_transform(X)
```

### Sparse PCA

Principal components with many zero weights (more interpretable):

```python
from sklearn.decomposition import SparsePCA

spca = SparsePCA(n_components=10, alpha=0.1)
X_reduced = spca.fit_transform(X)
```

### Probabilistic PCA

Bayesian formulation of PCA:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=10, svd_solver='randomized')
X_reduced = pca.fit_transform(X)
```

---

## Summary

**PCA in a nutshell**:

1. **Input**: High-dimensional data (N samples × D features)
2. **Process**: 
   - Center (and optionally standardize) data
   - Find directions of maximum variance (eigenvectors)
   - Order by variance captured (eigenvalues)
   - Keep top k directions
3. **Output**: Low-dimensional representation (N samples × k components)
4. **Goal**: Preserve maximum variance with minimum dimensions

**Key Equations**:
```
Covariance: C = (1/N) X^T X
Eigendecomposition: C v = λ v
SVD: X = U Σ V^T
Transformation: X_reduced = X · V_k
```

**When to use PCA**:
- ✅ High-dimensional data
- ✅ Features are correlated
- ✅ Want to visualize
- ✅ Want to denoise
- ✅ Want to speed up algorithms
- ✅ Data is roughly Gaussian/continuous

**When NOT to use PCA**:
- ❌ Features are already uncorrelated
- ❌ Nonlinear structure (use kernel PCA/t-SNE)
- ❌ Categorical data
- ❌ Interpretability is critical
- ❌ Very few features already

---

*This document provides the foundation for understanding how PCA relates to FedCorr's prediction subspace dimensionality analysis. See companion document for detailed comparison.*

