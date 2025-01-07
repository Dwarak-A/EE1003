import ctypes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the shared object file
trapezoidal_lib = ctypes.CDLL('./trapezoidal.so')

# Define the function prototypes for `curve_gen` and `trapezoidal_method`
trapezoidal_lib.curve_gen.argtypes = [
    ctypes.c_double,                    # x0
    ctypes.c_double,                    # xn
    ctypes.c_double,                    # detail
    ctypes.POINTER(ctypes.c_double),    # x_vals
    ctypes.POINTER(ctypes.c_double),    # y_vals
    ctypes.c_int                        # num_points
]

trapezoidal_lib.trapezoidal_method.argtypes = [
    ctypes.c_double,                    # x0
    ctypes.c_double,                    # xn
    ctypes.c_double,                    # h
    ctypes.POINTER(ctypes.c_double),    # area
    ctypes.POINTER(ctypes.c_double),    # x_vals
    ctypes.POINTER(ctypes.c_double),    # y_vals
    ctypes.c_int                        # num_points
]

# Parameters
x0 = 1.0       # Start of range
xn = 5.0       # End of range
detail = 0.001  # Step size for curve generation
h = 0.005       # Step size for trapezoidal integration
num_points_curve = int((xn - x0) / detail) + 1  # Points for curve generation
num_points_trapezoid = int((xn - x0) / h) + 1   # Points for trapezoidal integration

# Allocate memory for the output arrays
x_vals_curve = (ctypes.c_double * num_points_curve)()
y_vals_curve = (ctypes.c_double * num_points_curve)()

x_vals_trapezoid = (ctypes.c_double * num_points_trapezoid)()
y_vals_trapezoid = (ctypes.c_double * num_points_trapezoid)()
area = ctypes.c_double(0.0)

# Call the C function to generate the curve
trapezoidal_lib.curve_gen(x0, xn, detail, x_vals_curve, y_vals_curve, num_points_curve)

# Call the C function to perform trapezoidal integration
trapezoidal_lib.trapezoidal_method(x0, xn, h, ctypes.byref(area), x_vals_trapezoid, y_vals_trapezoid, num_points_trapezoid)

# Convert the results to NumPy arrays
x_vals_curve = np.array(x_vals_curve)
y_vals_curve = np.array(y_vals_curve)

x_vals_trapezoid = np.array(x_vals_trapezoid)
y_vals_trapezoid = np.array(y_vals_trapezoid)

# Visualization using Matplotlib
fig, ax = plt.subplots()

# Plot the trapezoids
for i in range(1, len(x_vals_trapezoid)):
    coords = [
        (x_vals_trapezoid[i - 1], 0),
        (x_vals_trapezoid[i], 0),
        (x_vals_trapezoid[i], y_vals_trapezoid[i]),
        (x_vals_trapezoid[i - 1], y_vals_trapezoid[i - 1])
    ]
    trapezium = patches.Polygon(coords, closed=True, facecolor='blue', edgecolor='blue', alpha = 0.5)
    ax.add_patch(trapezium)

# Plot the curve
ax.plot(x_vals_curve, y_vals_curve, color='red', label=r'$y = x^4$')

# Add vertical lines at x=1 and x=5
ax.axvline(x=1, color='green', linestyle='--', linewidth=1, label=r'$x=1$')  # Vertical line at x=1
ax.axvline(x=5, color='purple', linestyle='--', linewidth=1, label=r'$x=5$')  # Vertical line at x=5

# Add the x-axis (y=0)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, label=r'$y=0$')  # x-axis (y=0)

# Set plot limits and labels
ax.set_xlim(min(x_vals_curve)-0.2, max(x_vals_curve)+0.2)
ax.set_ylim(-20, max(y_vals_curve)+20)
ax.legend()
plt.grid()

# Print the Area
print(f'Area = {area.value:.4f}')

# Show the plot
plt.savefig("../figs/plot.png")
plt.show()

