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

Visualizes direction fields (slope fields) with solution curves overlaid.

### Equation Visualized

**dx/dt = 1 - sin(x)**

The script creates a direction field showing the behavior of the differential equation across a 2D space, then overlays three solution curves with different initial conditions.

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
- **Direction field arrows**: Show the slope at each point in the (t, x) plane
- **Solution curves**: Three trajectories starting from x(0) = 1.8, x(0) = π/2, and x(0) = -4.5

### Key Concepts

- **Direction field**: Normalized arrows showing the direction of solutions at grid points
- **Solution curves**: Numerical integration using `odeint` to show how solutions evolve over time
- The arrows are normalized to equal length to show direction only (not magnitude)



