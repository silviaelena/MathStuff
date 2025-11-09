# Import necessary libraries for plotting and numerical integration
import matplotlib.pyplot as plt
from scipy.integrate import odeint  # Solves ODEs numerically
import numpy as np
import warnings
from scipy.integrate import ODEintWarning

# Suppress expected warnings for stiff/unstable equations
warnings.filterwarnings('ignore', category=ODEintWarning)

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# 1 row, 2 columns of subplots
# ax1 = left plot, ax2 = right plot


# ===========================
# FIRST PLOT: dx/dt = 1 - sin(x)
# ===========================

# Define the vector field function (currently unused but kept for reference)
# Represents the system: dx/dt = 1, dy/dt = 1 - sin(x)
def vf(t, x):
    dx = np.zeros(2)
    dx[0] = 1  # Rate of change in x direction
    dx[1] = 1 - np.sin(x[0])  # Rate of change in y direction
    return dx


# ===== DIRECTION FIELD (SLOPE FIELD) =====
# Create a grid of 20x20 points from -10 to 10 in both directions
X, Y = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))

# Define the direction vectors at each grid point
U = 1  # Horizontal component (constant, moves right at unit speed)
V = 1 - np.sin(Y)  # Vertical component (depends on y value)

# Normalize arrow lengths so all arrows have the same size
# This shows direction only, not magnitude
N = np.sqrt(U ** 2 + V ** 2)  # Calculate magnitude of each vector
U2, V2 = U / N, V / N  # Divide by magnitude to normalize

# Draw the direction field as arrows
ax1.quiver(X, Y, U2, V2)

# Set axis limits and labels
ax1.set_xlim([-10, 10])
ax1.set_ylim([-10, 10])
ax1.set_xlabel(r"$t$")  # Using LaTeX formatting
ax1.set_ylabel(r"$x$")
ax1.set_title(r"$\frac{dx}{dt} = 1 - \sin(x)$", fontsize=16)


# ===== SOLUTION CURVES =====
# Define the ODE: dx/dt = 1 - sin(x)
def edo1(x, t):
    dx_dt = 1 - np.sin(x)  # The differential equation
    return dx_dt


# Create time array with 1000 points from -10 to 10
t = np.linspace(-10, 10, 1000)

# Solve and plot the ODE with different initial conditions
# Each solution curve shows how x evolves over time starting from a different x(0)

# Solution curve 1: Starting at x(0) = 1.8
sol_x = odeint(edo1, 1.8, t)  # odeint(function, initial_value, time_array)
ax1.plot(t, sol_x, linewidth=3)

# Solution curve 2: Starting at x(0) = π/2
sol_x = odeint(edo1, np.pi / 2, t)
ax1.plot(t, sol_x, linewidth=3)

# Solution curve 3: Starting at x(0) = -4.5
sol_x = odeint(edo1, -4.5, t)
ax1.plot(t, sol_x, linewidth=3)


# ===========================
# SECOND PLOT: dy/dt = y(y - 2)(y + 1)
# ===========================

# ===== DIRECTION FIELD (SLOPE FIELD) =====
# Create a grid of 20x20 points from -10 to 10 in both directions
X2, Y2 = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-3, 4, 20))

# Define the direction vectors at each grid point
U2_dir = 1  # Horizontal component (constant, time moves forward)
V2_dir = Y2 * (Y2 - 2) * (Y2 + 1)  # Vertical component: dy/dt = y(y - 2)(y + 1)

# Normalize arrow lengths so all arrows have the same size
N2 = np.sqrt(U2_dir ** 2 + V2_dir ** 2)  # Calculate magnitude
U2_norm, V2_norm = U2_dir / N2, V2_dir / N2  # Normalize

# Draw the direction field as arrows
ax2.quiver(X2, Y2, U2_norm, V2_norm)

# Set axis limits and labels
ax2.set_xlim([-10, 10])
ax2.set_ylim([-3, 4])
ax2.set_xlabel(r"$t$")
ax2.set_ylabel(r"$y$")
ax2.set_title(r"$\frac{dy}{dt} = y(y - 2)(y + 1)$", fontsize=16)

# Draw horizontal lines at equilibrium points
ax2.axhline(y=-1, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Equilibria')
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax2.axhline(y=2, color='gray', linestyle='--', linewidth=1, alpha=0.5)


# ===== SOLUTION CURVES =====
# Define the ODE: dy/dt = y(y - 2)(y + 1)
def edo2(y, t):
    dy_dt = y * (y - 2) * (y + 1)  # The differential equation
    return dy_dt


# Create time arrays for different solution curves
# Use very short time intervals for unstable solutions to avoid numerical blowup
t_unstable = np.linspace(-1, 1, 400)  # Very short interval for unstable solutions
t_stable = np.linspace(-10, 10, 1000)  # Longer interval for stable solutions

# Solve and plot the ODE with different initial conditions
# Equilibrium points are at y = -1, y = 0, y = 2

# Solution curve 1: Starting below y = -1 (goes to -infinity)
# Use very short time interval - solution blows up quickly
sol_y = odeint(edo2, -1.2, t_unstable)
ax2.plot(t_unstable, sol_y, linewidth=3, label='y(0) = -1.2')

# Solution curve 2: Starting between y = -1 and y = 0 (stable, converges to 0)
sol_y = odeint(edo2, -0.5, t_stable)
ax2.plot(t_stable, sol_y, linewidth=3, label='y(0) = -0.5')

# Solution curve 3: Starting between y = 0 and y = 2 (stable, converges to 0)
sol_y = odeint(edo2, 1, t_stable)
ax2.plot(t_stable, sol_y, linewidth=3, label='y(0) = 1')

# Solution curve 4: Starting above y = 2 (goes to +infinity)
# Use very short time interval - solution blows up quickly
sol_y = odeint(edo2, 2.2, t_unstable)
ax2.plot(t_unstable, sol_y, linewidth=3, label='y(0) = 2.2')

ax2.legend(loc='best')

# Display the complete figure with both direction fields and solution curves
plt.tight_layout()
plt.show()
