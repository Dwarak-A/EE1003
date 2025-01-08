import ctypes
import numpy as np
import matplotlib.pyplot as plt

# Loading the shared library
trapezoidal = ctypes.CDLL('./trapezoidal.so')

# Defining arguments and return types of the C function
trapezoidal.trapezoidal_rule.argtypes = [
    ctypes.c_double,	# x0
    ctypes.c_double,	# xn
    ctypes.c_double,	# a
    ctypes.c_double,	# x_init
    ctypes.c_double,	# y_init
    ctypes.c_double,	# h
    ctypes.POINTER(ctypes.c_double),  # y_theory
    ctypes.POINTER(ctypes.c_double),  # y_sim
    ctypes.POINTER(ctypes.c_double),  # x_vals
    ctypes.c_int       # num_points
]

# Parameters
x0 = -2.0
xn = 2.0
a = 0.5
x_init = 0.0
y_init = 1.0
h = 0.01
num_points = int((xn - x0) / h)

# Memory allocation for arrays
y_theory = (ctypes.c_double * num_points)()
y_sim = (ctypes.c_double * num_points)()
x_vals = (ctypes.c_double * num_points)()

# Calling the C function
trapezoidal.trapezoidal_rule(x0, xn, a, x_init, y_init, h, y_theory, y_sim, x_vals, num_points)

# Convert results to NumPy arrays for easier plotting
y_theory = np.array(y_theory)
y_sim = np.array(y_sim)
x_vals = np.array(x_vals)

# Plot the results
plt.plot(x_vals, y_theory, label='theory')
plt.plot(x_vals, y_sim, linestyle=':', color = 'r', label="sim")
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.legend()
plt.grid()
plt.savefig("../figs/plot.png")
plt.show()
