# MathStuff

Mathematical computations and visualizations using Python.

---

## Understanding Differential Equations: A Conceptual Framework

### What Differential Equations Really Are

A differential equation doesn't describe **states** — it describes **laws of change**. Instead of saying "the position is X," it says "the way position changes follows this rule." This fundamental shift in thinking is why differential equations appear everywhere in science and engineering.

### Why the Universe Uses Differential Equations

**1. Local Evolution**
The universe evolves locally: systems change based on their current state and immediate surroundings, not distant futures. A falling object doesn't "know" where it will be in one second; it only responds to gravity **now**. This local behavior is exactly what derivatives capture.

**2. Physical Laws Are Rules of Change**
Almost every fundamental law describes change:
- Force causes acceleration (Newton's laws)
- Voltage creates current change (circuits)
- Temperature differences create heat flow
- Population size affects growth rate

**Algebra describes fixed relationships. Differential equations describe how things evolve.**

### What Solving a Differential Equation Means

You're not solving for a number — you're discovering the **entire family of behaviors** consistent with a law of change.

**Example:** dy/dt = y says "I grow proportional to my size"
**Solution:** y = Ce^t describes **all possible worlds** where this law holds

The equation describes the **law**. Solutions describe **worlds consistent with the law**.

### Where Differential Equations Are Essential

- **Physics**: Motion, waves, heat, electricity, quantum mechanics
- **Biology**: Population dynamics, disease spread, neural networks
- **Economics**: Growth models, market dynamics, option pricing
- **Engineering**: Control systems, circuits, fluid dynamics
- **Chemistry**: Reaction kinetics, diffusion
- **Machine Learning**: Optimization algorithms, Neural ODEs
- **Climate Science**: Weather prediction, ocean currents

**The Pattern:** Wherever something changes over time or space based on local rules, differential equations emerge naturally.

### The Core Insight

> **A differential equation is a rule telling you how a system changes now. The solution tells you how the system evolves over all time.**

This is not just mathematical technique — it's a fundamental way of understanding reality. We move from knowing **where things are** to understanding **how they change**, and from that, reconstruct their entire story.

### Real-World Examples: Why DEs Appear Naturally

**Physics - Motion (Newton's Second Law)**

Force doesn't give you position directly — it gives you **how quickly velocity changes**. Gravity doesn't hand you future positions; it only tells you your acceleration **right now**.

```
F = ma  →  m(d²x/dt²) = F
```

For a falling object: d²x/dt² = g → Solution: x(t) = ½gt² + v₀t + x₀

**Why unavoidable:** The universe evolves moment-by-moment through local rules, not by jumping to final states.

**Biology - Population Growth**

Observations show: more individuals → faster growth. The rate of growth is proportional to current population size.

```
dP/dt = kP  →  Solution: P(t) = Ce^(kt)
```

This single DE explains exponential curves in epidemics, bacterial growth, viral spread, and compound interest. Not because nature "loves exponentials," but because these systems **create themselves** — more rabbits produce more rabbits.

**Economics - Supply/Demand Adjustment**

Markets don't jump to equilibrium prices — they adjust gradually based on current imbalances.

```
dP/dt = k(Demand(P) - Supply(P))
```

You cannot write a simple formula for "tomorrow's price." Markets react through local-in-time behavior: traders respond to current conditions, creating feedback loops. This DE captures how fast equilibrium is reached and whether the system oscillates or stabilizes.

**The Pattern:**
Nature doesn't hand you final outcomes. It hands you **rules of change**. Differential equations are the mathematics of evolution, flow, feedback, and dynamic interaction.

---

## Differential Equations

Solves and plots various ordinary differential equations (ODEs) using symbolic mathematics.

### Equations Solved

1. **dx/dt = t³** with x(1) = 2
2. **dy/dx = xe^x**
3. **dy/dx = sin x** with y(π/2) = 3
4. **(1 + x)dy - dx = 0** (equivalent to dy/dx = 1/(1+x))
5. **dx/dt = 2t + 1/t** with x(1) = 2
6. **dy/dx = x ln x** with y(1) = 3

### Requirements

```bash
pip install sympy numpy matplotlib
```

### Usage

```bash
cd differential-equations
python plot-differential-equations.py
```

This will:
- Print analytical solutions to the console
- Display a 2×3 grid of plots showing all six equations

### How It Works

1. **Symbolic solving**: Uses SymPy to solve ODEs analytically
2. **Initial conditions**: Applies boundary conditions to find particular solutions
3. **Numerical evaluation**: Converts symbolic solutions to numerical data
4. **Visualization**: Plots all solutions using matplotlib

---

## Direction Fields (Slope Fields)

Visualizes direction fields (slope fields) with solution curves overlaid for two different ODEs side by side.

### Equations Visualized

#### 1. **dx/dt = 1 - sin(x)**
- Autonomous ODE with periodic equilibria at x = π/2 + 2πn
- Semi-stable equilibrium points
- Three solution curves with initial conditions: x(0) = 1.8, x(0) = π/2, and x(0) = -4.5

#### 2. **dy/dt = y(y - 2)(y + 1)**
- Cubic autonomous ODE with three equilibrium points
- **Equilibria**: y = -1 (unstable), y = 0 (stable), y = 2 (unstable)
- Four solution curves demonstrating stability behavior:
  - y(0) = -1.2: Below y = -1, escapes to -∞
  - y(0) = -0.5: Between -1 and 0, converges to 0
  - y(0) = 1: Between 0 and 2, converges to 0
  - y(0) = 2.2: Above y = 2, escapes to +∞

### Requirements

```bash
pip install scipy numpy matplotlib
```

### Usage

```bash
cd differential-equations
python direction_field.py
```

This will display:
- **Two side-by-side plots**: One for each differential equation
- **Direction field arrows**: Show the slope at each point in the (t, x/y) plane
- **Solution curves**: Numerical integration using `odeint` showing how solutions evolve over time
- **Equilibrium lines**: Dashed gray lines marking equilibrium points (second plot)

### Key Concepts

- **Direction field**: Normalized arrows showing the direction of solutions at grid points
- **Equilibrium points**: Where dy/dt = 0 (solutions remain constant)
- **Stability**: Stable equilibria attract nearby solutions; unstable equilibria repel them
- **Sensitive dependence**: Near y = 2, small changes in initial conditions lead to drastically different long-term behavior

### Picard-Lindelöf Theorem (Existence and Uniqueness)

The **Picard-Lindelöf theorem** is a fundamental result that guarantees the existence and uniqueness of solutions to initial value problems.

#### Theorem Statement

For a differential equation of the form:

```
dy/dt = f(t, y),  with initial condition y(t₀) = y₀
```

If the function **f(t, y)** is:
1. **Continuous** in a region containing (t₀, y₀)
2. **Lipschitz continuous** in y (i.e., |f(t, y₁) - f(t, y₂)| ≤ K|y₁ - y₂| for some constant K)

Then there exists:
- A **unique solution** y(t) that satisfies the initial value problem
- The solution exists in some interval around t₀

#### Consequences

**Why Solution Curves Never Intersect:**

If two different solution curves intersected at point (t*, y*), then both curves would pass through the same point with the same value. This would mean two different solutions exist for the initial condition y(t*) = y*, which contradicts uniqueness.

**Exception:** Equilibrium points (constant solutions) are special cases where y(t) = c for all t. Solutions can approach equilibria asymptotically as t → ∞, but never reach them in finite time.

#### Visual Evidence

In the direction field plots:
- Each arrow at point (t, y) shows the **unique direction** of the solution passing through that point
- Solution curves follow these arrows and cannot cross each other
- This creates a deterministic "flow" in the phase space
- Initial conditions completely determine the future behavior of solutions

This theorem is the mathematical foundation that makes differential equations **predictable and deterministic**.

---

## Interactive Gradient Descent: The Sandpit

An interactive Jupyter notebook application that visualizes gradient descent and optimization through the metaphor of finding a lost phone in sand.

### The Concept

A phone has rolled to the lowest point of a pit (global minimum), which is then filled with sand. Your tool: a "dip-stick" that measures the **gradient (Jacobian)** at any point, showing which direction is "downhill."

**Goal:** Find the phone using as few measurements as possible by following the gradient.

### Mathematical Foundation

**Gradient (Jacobian) in 2D:**
```
∇f(x, y) = [∂f/∂x, ∂f/∂y]ᵀ
```

The **negative gradient** -∇f points in the direction of steepest descent.

**Optimization Problem:**
```
x* = argmin f(x, y)
```

**Numerical Differentiation** (Central Difference):
```
∂f/∂x ≈ [f(x+h, y) - f(x-h, y)] / 2h    (h = 0.01)
```

### Implementation Highlights

**Core Algorithm:**
1. Click anywhere in the 6×6 sandpit
2. Compute gradient at clicked point using finite differences
3. Display arrow showing -∇f (direction of steepest descent)
4. Check if within tolerance of global minimum
5. Detect and warn about local minima

**Surface Generation:**
- Uses Fourier series (sums of cosines) to create smooth, random terrain
- Global minimum found using `scipy.optimize.differential_evolution`
- Gradient computed numerically (no symbolic differentiation needed)

**Interactive Features:**
- Click-based interface using matplotlib event handlers
- Real-time gradient arrow visualization
- Progressive difficulty: gradient mode → depth-only mode → auto-descent
- Contour reveal upon winning

### Educational Value

**Teaches Core ML Concepts:**
- **Gradient descent**: The foundation of neural network training
- **Local vs global minima**: Why optimization is hard
- **Steepest descent**: Following the negative gradient
- **Convergence**: How algorithms find optimal solutions

**Real-world Connection:**

In machine learning:
- Surface = Loss function L(θ)
- Phone = Optimal weights θ*
- Gradient = ∇L(θ) guides weight updates
- Clicking = Taking gradient descent steps

**This is literally visualizing what happens inside neural networks during training!**

---

## Principal Component Analysis (PCA) - Week 1

### Week 1: Mean, Covariance, and Affine Transformations

An in-depth exploration of foundational statistical concepts for machine learning, focusing on how datasets behave under transformations.

#### Learning Objectives

1. **Computational Skills**: Implement statistical functions using vectorized NumPy operations
2. **Statistical Understanding**: Master mean and covariance computation for datasets
3. **Transformation Theory**: Understand how affine transformations affect dataset statistics
4. **Testing Discipline**: Develop robust testing practices for ML implementations

#### Key Concepts Covered

##### 1. **Mean Vector**

The mean represents the center (centroid) of your dataset in feature space.

$$
\boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{x}_i
$$

**Implementation Lessons:**
- **Naive approach**: Loop over each data point (slow in Python)
- **Vectorized approach**: `np.mean(X, axis=0)` — dramatically faster
- **Speedup**: ~100-1000× faster on typical datasets

##### 2. **Covariance Matrix**

The covariance matrix $\boldsymbol{\Sigma}$ encodes both individual feature variances and pairwise correlations.

$$
\boldsymbol{\Sigma} = \frac{1}{N} (\mathbf{X} - \boldsymbol{\mu})^T (\mathbf{X} - \boldsymbol{\mu})
$$

**Key Properties:**
- **Symmetric**: $\Sigma_{ij} = \Sigma_{ji}$
- **Positive semi-definite**: All eigenvalues $\geq 0$
- **Diagonal elements**: Variances of individual features
- **Off-diagonal elements**: Covariances between feature pairs

**Implementation Lessons:**
- **Naive approach**: Triple nested loop over N×D×D (extremely slow)
- **Vectorized approach**: Matrix multiplication `(X_centered.T @ X_centered) / N`
- **Critical insight**: Center data first, then use one matrix operation

##### 3. **Affine Transformations**

Understanding how statistics transform under $\mathbf{Y} = \mathbf{X} \mathbf{A}^T + \mathbf{b}$

**Transformation Rules:**

Mean transformation:
$$
\boldsymbol{\mu}_{\text{new}} = \mathbf{A} \boldsymbol{\mu} + \mathbf{b}
$$

Covariance transformation:
$$
\boldsymbol{\Sigma}_{\text{new}} = \mathbf{A} \boldsymbol{\Sigma} \mathbf{A}^T
$$

**Critical Insights:**
- Translation $\mathbf{b}$ **only affects the mean**, not the covariance
- Linear transformation $\mathbf{A}$ affects **both mean and covariance**
- Covariance describes the **shape** of data, independent of location
- These rules enable computing statistics without explicitly transforming data

#### Practical Applications

**Dataset Used:**
- **Olivetti Faces Dataset**: 400 face images (64×64 pixels = 4096 dimensions)
- Demonstrates high-dimensional data analysis
- Visualizes "mean face" as the dataset centroid

**Real-World Relevance:**

1. **Feature Normalization**: Standardizing data to have zero mean and unit variance
2. **Whitening Transformations**: Making features uncorrelated and equal variance
3. **PCA Foundation**: Understanding how covariance eigenvalues reveal data structure
4. **Federated Learning**: Computing distributed statistics without sharing raw data
5. **Data Augmentation**: Predicting statistics after transformations (rotation, scaling)

#### Why Vectorization Matters

**Benchmark Results (1000×20 dataset):**
```
mean_naive():  ~50ms  (Python loop)
mean():        ~0.5ms (NumPy vectorized) → 100× speedup

cov_naive():   ~500ms  (Triple Python loop)
cov():         ~2ms    (Matrix multiplication) → 250× speedup
```

**Key Lesson:** In ML, loops are enemies. Matrix operations leverage optimized BLAS libraries and enable GPU acceleration.

#### Testing Philosophy

Every function includes:
- **Multiple test cases**: Edge cases, zero covariance, identity matrices
- **Numerical tolerance**: Using `np.testing.assert_allclose` with `rtol=1e-5`
- **Verification**: Applying transformations twice and checking consistency

**Critical principle:** In ML, silent errors are catastrophic. Test everything.

#### Files and Resources

- **`week1.ipynb`**: Interactive Jupyter notebook with complete implementation
- **`week1_conclusions_mean_covariance_affine.md`**: Comprehensive theoretical guide with examples
- **Dataset**: Olivetti faces (400 grayscale 64×64 images)

#### Mathematical Foundations for PCA

This week establishes the groundwork for Principal Component Analysis:

- **Next step**: Eigendecomposition of $\boldsymbol{\Sigma}$ reveals principal components
- **Eigenvectors**: Directions of maximum variance
- **Eigenvalues**: Amount of variance along each direction
- **Dimensionality reduction**: Project onto top-k eigenvectors

**The Core Insight:**

> Understanding how covariance transforms under linear operations is the foundation for understanding PCA, whitening, and essentially all linear methods in machine learning.

#### Geometric Interpretation

- **Mean** $\boldsymbol{\mu}$ → Center of the data cloud
- **Covariance** $\boldsymbol{\Sigma}$ → Shape and orientation of the ellipsoidal data distribution
- **Eigenvectors** → Principal axes of the ellipsoid
- **Eigenvalues** → Spread along those axes

**Visual Analogy:** Your data is an elliptical cloud in high-dimensional space. The mean is the center, and the covariance matrix encodes the exact shape, tilt, and stretch of that cloud.

---


