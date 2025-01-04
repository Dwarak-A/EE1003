import ctypes
import numpy as np
import matplotlib.pyplot as plt

# Loading the shared library
euler = ctypes.CDLL('./euler.so')

# Defining arguments and return types of the C function
euler.euler_method.argtypes = [
    ctypes.c_double,   # y0
    ctypes.c_double,   # yn
    ctypes.c_double,   # h
    ctypes.POINTER(ctypes.c_double),  # x_given
    ctypes.POINTER(ctypes.c_double),  # x_approx
    ctypes.POINTER(ctypes.c_double),  # y_values
    ctypes.c_int       # num_points
]

# Parameters
y0 = -4.0
yn = 4.0
h = 0.01
num_points = int((yn - y0) / h) + 1

# Memory allocation for arrays
x_theory = (ctypes.c_double * num_points)()
x_approx = (ctypes.c_double * num_points)()
y_values = (ctypes.c_double * num_points)()

# Calling the C function
euler.euler_method(y0, yn, h, x_theory, x_approx, y_values, num_points)

# Convert results to NumPy arrays for easier plotting
x_theory = np.array(x_theory)
x_approx = np.array(x_approx)
y_values = np.array(y_values)

# Plot the results
plt.plot(x_theory, y_values, label='theory')
plt.plot(x_approx, y_values, linestyle=':', color = 'r', label="sim")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.savefig("../figs/plot.png")
plt.show()
