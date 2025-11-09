# MathStuff

Mathematical computations and visualizations using Python.

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



