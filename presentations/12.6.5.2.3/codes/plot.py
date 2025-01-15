import ctypes
import numpy as np
import matplotlib.pyplot as plt

# Load the shared object file
gradient_lib = ctypes.CDLL('./gradient.so')

# Define argument and return types for func and func_deriv
gradient_lib.func.argtypes = [ctypes.c_double]  # double func(double x)
gradient_lib.func.restype = ctypes.c_double

gradient_lib.func_deriv.argtypes = [ctypes.c_double]  # double func_deriv(double x)
gradient_lib.func_deriv.restype = ctypes.c_double

gradient_lib.points_gen.argtypes = [
    ctypes.c_double,  # double x_start
    ctypes.c_double,  # double x_end
    ctypes.POINTER(ctypes.c_double),  # double* x_vals
    ctypes.POINTER(ctypes.c_double),  # double* y_vals
    ctypes.c_double,  # double h
    ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)  # double (*func)(double)
]
gradient_lib.points_gen.restype = None

gradient_lib.gradient_method.argtypes = [
    ctypes.c_double,  # double x0
    ctypes.c_double,  # double learning_rate
    ctypes.c_double,  # double tol
    ctypes.POINTER(ctypes.c_double),  # double extremum[2]
    ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double),  # function pointer for func
    ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)   # function pointer for func_deriv
]
gradient_lib.gradient_method.restype = None

# Create function pointers using ctypes from the shared library functions directly
FuncType = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)

# Get function pointers from the shared library
func_ptr = FuncType(gradient_lib.func)
func_deriv_ptr = FuncType(gradient_lib.func_deriv)

# Parameters
x_start = -2.0       # Start of range
x_end = 2.0         # End of range
h = 0.01            # Step size for trapezoidal integration
num_points = int((x_end - x_start) / h) # Points for trapezoidal integration

x0 = 0.0
learning_rate = 0.1
tol = 1e-6

# Allocate memory for the output arrays
x_vals = (ctypes.c_double * num_points)()
y_vals = (ctypes.c_double * num_points)()

# Allocate memory for the extremum
min_coords = (ctypes.c_double * 2)()
max_coords = (ctypes.c_double * 2)()

# Generate points using func pointer
gradient_lib.points_gen(x_start, x_end, x_vals, y_vals, h, func_ptr)

# Gradient Descent
gradient_lib.gradient_method(x0, -learning_rate, tol, min_coords, func_ptr, func_deriv_ptr)

# Gradient Ascent
gradient_lib.gradient_method(x0, learning_rate, tol, max_coords, func_ptr, func_deriv_ptr)

# Convert the results to NumPy arrays
x_vals = np.array(x_vals)
y_vals = np.array(y_vals)

# Print the minimum found by gradient descent
print(f"Minimum found at x = {min_coords[0]}, y = {min_coords[1]}")
print(f"Maximum found at x = {max_coords[0]}, y = {max_coords[1]}")

# Plot the function and the gradient descent result
plt.plot(x_vals, y_vals, label='function')
plt.scatter(min_coords[0], min_coords[1], color='red', label='minimum')
plt.scatter(max_coords[0], max_coords[1], color='green', label='maximum')

plt.legend()
plt.axis('equal')
plt.grid()

plt.savefig("../figs/plot.png")
plt.show()

