# PCA - Imperial College ML Math Course

Week 1 work on mean, covariance, and affine transformations. This is the foundation for PCA.

---

## Week 1: Mean, Covariance, Affine Transformations

Started with the basics - mean, covariance, and how they change under transformations. The goal was to implement these using vectorized NumPy (not loops!) and understand how dataset statistics transform.

---

## Mean Vector

The mean is just the center of your data in feature space.

μ = (1/N) Σ x_i

**What I learned:**

I first wrote a naive loop version. It was slow. Then I used `np.mean(X, axis=0)` and it was like 100-1000× faster. Vectorization matters a lot.

---

## Covariance Matrix

The covariance matrix Σ encodes variances and correlations between features.

Σ = (1/N) (X - μ)ᵀ (X - μ)

**Properties:**
- Symmetric: Σ_ij = Σ_ji
- Positive semi-definite (all eigenvalues ≥ 0)
- Diagonal = variances of individual features
- Off-diagonal = covariances between feature pairs

**Implementation:**

My first version used triple nested loops. It was embarrassingly slow. The vectorized version is just `(X_centered.T @ X_centered) / N` - one matrix operation. Key insight: center the data first, then one matrix multiply.

Speedup was huge: naive version took ~500ms, vectorized took ~2ms on a 1000×20 dataset. That's 250× faster.

---

## Affine Transformations

How do statistics change when you transform data? If Y = X Aᵀ + b:

**Mean transforms as:**
μ_new = A μ + b

**Covariance transforms as:**
Σ_new = A Σ Aᵀ

**Important points:**
- Translation b only affects the mean, not covariance
- Linear transformation A affects both mean and covariance
- Covariance describes shape, independent of location
- You can compute transformed statistics without actually transforming the data

This is useful - you can predict what happens to statistics after transformations without doing the transformation.

---

## What I worked with

Used the Olivetti Faces dataset - 400 face images, each 64×64 pixels (4096 dimensions). High-dimensional data. The "mean face" is interesting - it's literally the average of all faces, and it looks like a blurry generic face.

---

## Why this matters

This stuff shows up everywhere:
1. Feature normalization - standardizing to zero mean, unit variance
2. Whitening - making features uncorrelated with equal variance
3. PCA foundation - covariance eigenvalues reveal data structure
4. Federated learning - computing distributed statistics without sharing raw data
5. Data augmentation - predicting statistics after rotations/scaling

---

## Vectorization is everything

Benchmarks on 1000×20 data:
- mean_naive(): ~50ms (Python loop)
- mean(): ~0.5ms (NumPy) → 100× faster

- cov_naive(): ~500ms (triple loop)
- cov(): ~2ms (matrix multiply) → 250× faster

In ML, loops are the enemy. Matrix operations use optimized BLAS libraries and can run on GPUs.

---

## Testing

I made sure to test everything:
- Edge cases
- Zero covariance
- Identity matrices
- Applied transformations twice to check consistency

Used `np.testing.assert_allclose` with rtol=1e-5. In ML, silent errors are bad - you need to catch them.

---

## Files

- `work/week1.ipynb` - Notebook with implementations
- `work/week1_conclusions_mean_covariance_affine.md` - More detailed notes
- Dataset: Olivetti faces (400 grayscale 64×64 images)

---

## What's next

This sets up PCA:
- Eigendecomposition of Σ gives principal components
- Eigenvectors = directions of maximum variance
- Eigenvalues = amount of variance along each direction
- Project onto top-k eigenvectors for dimensionality reduction

The main insight: understanding how covariance transforms under linear operations is the foundation for PCA, whitening, and basically all linear ML methods.

**Geometric view:**
- Mean μ = center of the data cloud
- Covariance Σ = shape and orientation of the ellipsoid
- Eigenvectors = principal axes of the ellipsoid
- Eigenvalues = how stretched it is along each axis

Think of your data as an elliptical cloud. The mean is the center, covariance encodes the shape, tilt, and stretch.
