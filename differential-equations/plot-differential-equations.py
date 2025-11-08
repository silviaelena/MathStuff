from sympy import symbols, Function, Eq, dsolve, solve, exp, sin, log
import numpy as np
import matplotlib.pyplot as plt

# Create subplots for all 6 equations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()


# (a) dx/dt = t³ and x(1) = 2
t = symbols('t')
x = Function('x')
eq = Eq(x(t).diff(t), t**3)
sol = dsolve(eq)
C1 = symbols('C1')
sol1 = sol.rhs.subs(t, 1)
# Don't forget to substitute the constants
C1_val = solve(Eq(sol1, 2), C1)[0]
sol_final = sol.subs(C1, C1_val)
print(f"(a) {sol_final}")

t_vals = np.linspace(0.1, 3, 100)
x_vals = [float(sol_final.rhs.subs(t, val)) for val in t_vals]
axes[0].plot(t_vals, x_vals)
axes[0].set_xlabel('t')
axes[0].set_ylabel('x(t)')
axes[0].set_title('(a) dx/dt = t³, x(1) = 2')
axes[0].grid(True)


# (b) dy/dx = xe^x
y = Function('y')
x = symbols('x')
eq = Eq(y(x).diff(x), x * exp(x))
sol = dsolve(eq)
C1 = symbols('C1')
sol_final = sol.subs(C1, 0)
print(f"(b) {sol}")

x_vals = np.linspace(-1, 2, 100)
y_vals = [float(sol_final.rhs.subs(x, val)) for val in x_vals]
axes[1].plot(x_vals, y_vals)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y(x)')
axes[1].set_title('(b) dy/dx = xe^x')
axes[1].grid(True)


# (c) dy/dx = sin x and y(π/2) = 3
x = symbols('x')
y = Function('y')
eq = Eq(y(x).diff(x), sin(x))
sol = dsolve(eq)
C1 = symbols('C1')
sol1 = sol.rhs.subs(x, np.pi/2)
C1_val = solve(Eq(sol1, 3), C1)[0]
sol_final = sol.subs(C1, C1_val)
print(f"(c) {sol_final}")

x_vals = np.linspace(0, 2*np.pi, 100)
y_vals = [float(sol_final.rhs.subs(x, val)) for val in x_vals]
axes[2].plot(x_vals, y_vals)
axes[2].set_xlabel('x')
axes[2].set_ylabel('y(x)')
axes[2].set_title('(c) dy/dx = sin x, y(π/2) = 3')
axes[2].grid(True)


# (d) (1 + x)dy - dx = 0  =>  dy/dx = 1/(1+x)
x = symbols('x')
y = Function('y')
eq = Eq(y(x).diff(x), 1/(1+x))
sol = dsolve(eq)
C1 = symbols('C1')
sol_final = sol.subs(C1, 0)
print(f"(d) {sol_final}")

x_vals = np.linspace(-0.9, 3, 100)
y_vals = [float(sol_final.rhs.subs(x, val)) for val in x_vals]
axes[3].plot(x_vals, y_vals)
axes[3].set_xlabel('x')
axes[3].set_ylabel('y(x)')
axes[3].set_title('(d) (1 + x)dy - dx = 0  =>  dy/dx = 1/(1+x)')
axes[3].grid(True)


# (e) dx/dt = 2t + 1/t and x(1) = 2
t = symbols('t')
x = Function('x')
eq = Eq(x(t).diff(t), 2*t + 1/t)
sol = dsolve(eq)
C1 = symbols('C1')
sol1 = sol.rhs.subs(t, 1)
C1_val = solve(Eq(sol1, 2), C1)[0]
sol_final = sol.subs(C1, C1_val)
print(f"(e) {sol_final}")

t_vals = np.linspace(0.1, 3, 100)
x_vals = [float(sol_final.rhs.subs(symbols('t'), val)) for val in t_vals]
axes[4].plot(t_vals, x_vals)
axes[4].set_xlabel('t')
axes[4].set_ylabel('x(t)')
axes[4].set_title('(e) dx/dt = 2t + 1/t, x(1) = 2')
axes[4].grid(True)

# (f) dy/dx = x ln x and y(1) = 3
x = symbols('x')
y = Function('y')
eq = Eq(y(x).diff(x), x*log(x))
sol = dsolve(eq)
C1 = symbols('C1')
sol1 = sol.rhs.subs(x, 1)
C1_val = solve(Eq(sol1, 3), C1)[0]
sol_final = sol.subs(C1, C1_val)
print(f"(f) {sol_final}")

x_vals = np.linspace(0.1, 3, 100)
y_vals = [float(sol_final.rhs.subs(x, val)) for val in x_vals]
axes[5].plot(x_vals, y_vals)
axes[5].set_xlabel('x')
axes[5].set_ylabel('y(x)')
axes[5].set_title('(f) dy/dx = x ln x and y(1) = 3')
axes[5].grid(True)

plt.tight_layout()
plt.show()