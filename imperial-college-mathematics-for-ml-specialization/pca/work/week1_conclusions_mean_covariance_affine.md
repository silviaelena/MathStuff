# Mean, Variance, Covariance, and Affine Transformations

A comprehensive guide to understanding statistical properties and transformations in machine learning.

---

## Dataset Example

Consider a simple 2D dataset with 4 data points:

```python
X = np.array([
    [1, 10],
    [2, 20],
    [3, 30],
    [4, 40]
])
```

**Dataset Properties:**
- \( N = 4 \) (number of data points)
- \( D = 2 \) (dimensionality of each data point)
- Each row represents a sample
- Each column represents a feature

---

## 1. Mean Vector

**Definition:** The mean vector \( \boldsymbol{\mu} \) represents the center of the dataset in feature space.

**Mathematical Formula:**
\[
\boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{x}_i
\]

**Computation:**

```python
mean = np.mean(X, axis=0)  # Result: [2.5, 25.0]
```

**Key Properties:**
- Each component is the average of a feature across all data points
- Represents the centroid of the data cloud in \( \mathbb{R}^D \)
- Used for data centering: \( \mathbf{X}_{\text{centered}} = \mathbf{X} - \boldsymbol{\mu} \)

---

## 2. Variance and Covariance

### 2.1 Variance (Spread of Individual Features)

**Definition:** Variance measures how much a single feature fluctuates around its mean.

**Formula:**
\[
\text{Var}(X_j) = \frac{1}{N} \sum_{i=1}^{N} (X_{ij} - \mu_j)^2
\]

**Example:**

```python
Var(X[:, 0]) = 1.25   # Variance of first feature
Var(X[:, 1]) = 125.0  # Variance of second feature
```

**Interpretation:**
- Higher variance → feature values are more spread out
- Lower variance → feature values are clustered around the mean

### 2.2 Covariance (Joint Variability)

**Definition:** Covariance measures how two features vary together.

**Formula:**
\[
\text{Cov}(X_j, X_k) = \frac{1}{N} \sum_{i=1}^{N} (X_{ij} - \mu_j)(X_{ik} - \mu_k)
\]

**Example:**

```python
Cov(X[:, 0], X[:, 1]) = 12.5
```

**Interpretation:**
- **Positive covariance**: Features tend to increase together
- **Negative covariance**: One feature increases as the other decreases
- **Zero covariance**: No linear relationship between features

### 2.3 Covariance Matrix

**Definition:** The covariance matrix \( \boldsymbol{\Sigma} \) encodes all pairwise covariances.

**Formula:**
\[
\boldsymbol{\Sigma} = \frac{1}{N} (\mathbf{X} - \boldsymbol{\mu})^T (\mathbf{X} - \boldsymbol{\mu})
\]

**Computation:**

```python
Sigma = np.cov(X, rowvar=False)  # Shape: (2, 2)
# Result:
# [[  1.6667,  16.6667],
#  [ 16.6667, 166.6667]]
```

**Key Properties:**
- **Symmetric**: \( \Sigma_{ij} = \Sigma_{ji} \)
- **Positive semi-definite**: All eigenvalues \( \geq 0 \)
- **Diagonal elements**: Variances of individual features
- **Off-diagonal elements**: Covariances between feature pairs

---

## 3. Affine Transformations

### 3.1 Transformation Components

An affine transformation consists of:
1. **Linear transformation** \( \mathbf{A} \): Scaling, rotation, shearing
2. **Translation** \( \mathbf{b} \): Shifting

**Formula:**
\[
\mathbf{Y} = \mathbf{X} \mathbf{A}^T + \mathbf{b}
\]

**Example:**

```python
A = np.array([[2, 0],      # Scale x by 2, y by 0.5
              [0, 0.5]])
b = np.array([1, -2])       # Shift by (1, -2)
```

### 3.2 Transforming the Data

```python
X_transformed = X @ A.T + b
# Result:
# [[ 3.,   3.],
#  [ 5.,   8.],
#  [ 7.,  13.],
#  [ 9.,  18.]]
```

### 3.3 Transforming the Mean

**Formula:**
\[
\boldsymbol{\mu}_{\text{new}} = \mathbf{A} \boldsymbol{\mu} + \mathbf{b}
\]

**Computation:**

```python
mean_transformed = A @ mean + b
# Result: [6.0, 10.5]
```

**Key Insight:** Translation \( \mathbf{b} \) directly affects the mean.

### 3.4 Transforming the Covariance

**Formula:**
\[
\boldsymbol{\Sigma}_{\text{new}} = \mathbf{A} \boldsymbol{\Sigma} \mathbf{A}^T
\]

**Computation:**

```python
Sigma_transformed = A @ Sigma @ A.T
# Result:
# [[ 6.6667,  8.3335],
#  [ 8.3335, 41.6667]]
```

**Key Insights:**
- Translation \( \mathbf{b} \) does **not** affect covariance
- Only the linear transformation \( \mathbf{A} \) changes the covariance
- The transformation preserves the positive semi-definite property

---

## 4. Summary and Key Takeaways

| **Concept** | **Description** | **Effect of Affine Transform** |
|-------------|-----------------|-------------------------------|
| **Mean** | Center of the data cloud | Affected by both \( \mathbf{A} \) and \( \mathbf{b} \) |
| **Variance** | Spread of individual features | Affected only by \( \mathbf{A} \) |
| **Covariance** | Joint variability of features | Affected only by \( \mathbf{A} \) |
| **Covariance Matrix** | Shape, orientation, and correlations | Transforms as \( \mathbf{A} \boldsymbol{\Sigma} \mathbf{A}^T \) |

---

## 5. Geometric Interpretation

Understanding these concepts geometrically:

- **Mean** \( \boldsymbol{\mu} \) → **Center** of the data cloud
- **Covariance matrix** \( \boldsymbol{\Sigma} \) → **Shape and orientation** of the data cloud
- **Eigenvectors of** \( \boldsymbol{\Sigma} \) → **Principal directions** (axes of the ellipsoid)
- **Eigenvalues of** \( \boldsymbol{\Sigma} \) → **Spread** along those directions

**Visual Analogy:**
- Imagine the data as an elliptical cloud in space
- The mean is the center of the ellipse
- The eigenvectors point along the major and minor axes
- The eigenvalues determine how stretched the ellipse is along each axis

---

## 6. Applications

This framework is fundamental to:

- **Principal Component Analysis (PCA)**: Finding directions of maximum variance
- **Feature Normalization**: Standardizing data to have zero mean and unit variance
- **Whitening**: Transforming data to have identity covariance
- **Federated Learning**: Computing distributed statistics without sharing raw data
- **Gaussian Distributions**: Parameterizing multivariate normal distributions
- **Dimensionality Reduction**: Projecting data onto lower-dimensional subspaces

---

## 🔗 Further Reading

- **PCA**: Understanding how eigendecomposition of \( \boldsymbol{\Sigma} \) reveals data structure
- **Mahalanobis Distance**: Distance metric that accounts for covariance
- **Whitening Transformations**: Making features uncorrelated with unit variance

