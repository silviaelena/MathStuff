# Calculus - Imperial College ML Math Course

My work from the Calculus module. The sandpit exercise was actually really helpful for understanding gradient descent.

---

## The Sandpit

This is an interactive notebook where you find a phone that's buried in sand. The phone is at the lowest point of a pit (global minimum), and you have a "dip-stick" that measures the gradient at any point - basically tells you which way is downhill.

Goal: find the phone with as few measurements as possible by following the gradient.

**The math:**

Gradient in 2D: ∇f(x, y) = [∂f/∂x, ∂f/∂y]ᵀ

The negative gradient -∇f points downhill (steepest descent).

We're solving: x* = argmin f(x, y)

Gradient computed numerically using central differences: ∂f/∂x ≈ [f(x+h, y) - f(x-h, y)] / 2h with h = 0.01

**How it works:**

1. Click anywhere in the 6×6 sandpit
2. Compute gradient at that point using finite differences
3. Show an arrow pointing in the direction of steepest descent
4. Check if you're close enough to the minimum
5. Warns you if you hit a local minimum

The surface is generated using Fourier series (sums of cosines) to make random terrain. Global minimum found with scipy.optimize.differential_evolution. Gradient is computed numerically - no symbolic math needed.

**Features:**
- Click to measure gradient
- Arrow shows which way to go
- Three difficulty modes: gradient mode → depth-only mode → auto-descent
- Shows contour plot when you win

**Why this matters:**

This is basically what happens in neural network training:
- The surface = loss function L(θ)
- The phone = optimal weights θ*
- The gradient = ∇L(θ) that guides weight updates
- Each click = a gradient descent step

It really helped me visualize what's going on inside neural networks during training. The local vs global minima thing becomes obvious when you're clicking around trying to find the phone.

**Files:**
- `work/The Sandpit.ipynb` - Main interactive exercise
- `work/Gradient descent in a sandpit.ipynb` - Implementation exercise
- `work/The Sandpit - Part 2.ipynb` - More advanced stuff
- `work/readonly/` - Supporting code and resources
