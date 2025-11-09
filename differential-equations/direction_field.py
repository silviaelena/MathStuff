# Import necessary libraries for plotting and numerical integration
import matplotlib.pyplot as plt
from scipy.integrate import odeint  # Solves ODEs numerically
import numpy as np

# Create a new figure and axis for plotting
fig = plt.figure(num=1)
ax = fig.add_subplot(111)  # Single subplot (1 row, 1 column, position 1)

# Resize window to 11x11 inches for better visibility
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 11  # Width
fig_size[1] = 11  # Height


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
ax.quiver(X, Y, U2, V2)

# Set axis limits and labels
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.xlabel(r"$x$")  # Using LaTeX formatting
plt.ylabel(r"$y$")


# ===== SOLUTION CURVES =====
# Define the ODE: dx/dt = 1 - sin(x)
def edo(x, t):
    dx_dt = 1 - np.sin(x)  # The differential equation
    return dx_dt


# Create time array with 1000 points from -10 to 10
t = np.linspace(-10, 10, 1000)

# Solve and plot the ODE with different initial conditions
# Each solution curve shows how x evolves over time starting from a different x(0)

# Solution curve 1: Starting at x(0) = 1.8
sol_x = odeint(edo, 1.8, t)  # odeint(function, initial_value, time_array)
ax.plot(t, sol_x, linewidth=3)

# Solution curve 2: Starting at x(0) = π/2
sol_x = odeint(edo, np.pi / 2, t)
ax.plot(t, sol_x, linewidth=3)

# Solution curve 3: Starting at x(0) = -4.5
sol_x = odeint(edo, -4.5, t)
ax.plot(t, sol_x, linewidth=3)

# Display the complete figure with direction field and solution curves
plt.show()
