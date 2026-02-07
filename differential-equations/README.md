# Differential Equations

Playing around with differential equations in Python. Mostly trying to understand what they actually mean and how to solve/visualize them.

---

## What are differential equations anyway?

I kept seeing DEs everywhere and realized I didn't really understand what they were. So I wrote down some thoughts:

Differential equations don't tell you where something is - they tell you how it changes. Instead of "the position is X", it's "the position changes like this". That shift in thinking explains why they're everywhere.

The universe seems to work locally - things change based on what's happening right now, not what will happen later. A falling object doesn't know where it'll be in a second, it just responds to gravity now. That's what derivatives capture.

Most physical laws are about change:
- Force → acceleration
- Voltage → current change  
- Temperature differences → heat flow
- Population size → growth rate

Algebra gives you fixed relationships. DEs give you how things evolve.

When you solve a DE, you're not finding one number - you're finding all the possible behaviors that follow a rule. Like dy/dt = y means "I grow proportional to my size", and the solution y = Ce^t describes all the worlds where that's true.

They show up in physics (motion, waves, quantum stuff), biology (populations, disease spread), economics (markets, growth), engineering (control systems, circuits), ML (optimization, Neural ODEs), and way more. Basically anywhere something changes based on local rules.

The key idea: A DE is a rule for how a system changes right now. The solution tells you how it evolves over time.

---

## Some examples

**Physics - falling objects**

Force doesn't give you position, it gives you how velocity changes. Gravity tells you your acceleration now, not where you'll be later.

F = ma → m(d²x/dt²) = F

For falling: d²x/dt² = g, solution is x(t) = ½gt² + v₀t + x₀

**Biology - population growth**

More individuals → faster growth. Growth rate proportional to current size.

dP/dt = kP → P(t) = Ce^(kt)

This explains exponentials in epidemics, bacteria, viruses, compound interest. Not because nature likes exponentials, but because these systems create themselves - more rabbits make more rabbits.

**Economics - markets**

Markets don't jump to equilibrium, they adjust gradually.

dP/dt = k(Demand(P) - Supply(P))

You can't write a simple formula for tomorrow's price. Markets react locally - traders respond to current conditions. This DE captures how fast equilibrium happens and whether things oscillate or stabilize.

Nature gives you rules of change, not final outcomes. DEs are the math of evolution, flow, and feedback.

---

## Code stuff

### Solving ODEs

I solved a few ODEs using SymPy:

1. dx/dt = t³ with x(1) = 2
2. dy/dx = xe^x
3. dy/dx = sin x with y(π/2) = 3
4. (1 + x)dy - dx = 0 (same as dy/dx = 1/(1+x))
5. dx/dt = 2t + 1/t with x(1) = 2
6. dy/dx = x ln x with y(1) = 3

To run:
```bash
pip install sympy numpy matplotlib
python plot-differential-equations.py
```

Prints solutions and shows plots of all six.

How it works: SymPy solves symbolically, applies initial conditions, converts to numbers, then matplotlib plots everything.

---

### Direction Fields

Made plots showing direction fields (slope fields) with solution curves. Two different ODEs side by side.

**First one: dx/dt = 1 - sin(x)**
- Autonomous ODE with periodic equilibria at x = π/2 + 2πn
- Semi-stable equilibrium points
- Plotted three solution curves starting from different points

**Second one: dy/dt = y(y - 2)(y + 1)**
- Cubic autonomous ODE with three equilibrium points
- Equilibria: y = -1 (unstable), y = 0 (stable), y = 2 (unstable)
- Four solution curves showing different behaviors:
  - y(0) = -1.2: goes to -∞
  - y(0) = -0.5: converges to 0
  - y(0) = 1: also converges to 0
  - y(0) = 2.2: goes to +∞

To run:
```bash
pip install scipy numpy matplotlib
python direction_field.py
```

Shows two plots side by side with arrows showing slopes, solution curves from numerical integration, and equilibrium lines.

Key concepts:
- Direction field = arrows showing solution direction at each point
- Equilibrium points = where dy/dt = 0 (solutions stay constant)
- Stability = stable equilibria attract nearby solutions, unstable ones repel
- Near y = 2, tiny changes in starting point lead to completely different outcomes

---

## Picard-Lindelöf Theorem

This theorem says when solutions exist and are unique.

If you have dy/dt = f(t, y) with y(t₀) = y₀, and f is:
1. Continuous near (t₀, y₀)
2. Lipschitz continuous in y (basically |f(t, y₁) - f(t, y₂)| ≤ K|y₁ - y₂|)

Then there's a unique solution in some interval around t₀.

**Why solution curves don't cross:**

If two curves crossed at (t*, y*), you'd have two different solutions for the same initial condition, which breaks uniqueness. Exception: equilibrium points are special - solutions can approach them but never reach them in finite time.

In the direction field plots, each arrow shows the unique direction at that point. Solution curves follow the arrows and can't cross. This makes DEs deterministic - initial conditions determine everything.
