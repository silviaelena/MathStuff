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



