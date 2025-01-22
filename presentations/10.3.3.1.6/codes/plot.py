import ctypes
import numpy as np
import matplotlib.pyplot as plt

# Load the shared library
lib = ctypes.CDLL('./lu.so')

# Define argument and return types for the functions
lib.createMat.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
lib.createMat.argtypes = [ctypes.c_int, ctypes.c_int]

lib.freeMat.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_int]

lib.LU_decomp.restype = ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))
lib.LU_decomp.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_int]

lib.solveLU.restype = ctypes.c_int
lib.solveLU.argtypes = [
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # L
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # U
    ctypes.POINTER(ctypes.c_double),  # b
    ctypes.POINTER(ctypes.c_double),  # x
    ctypes.c_int,  # dim
]

# Define dimensions and matrices
dim = 2

a1, b1, a2, b2 = 9, -10, 2, 3
c1, c2 = -12, 13

matrix = np.array([[a1, b1], [a2, b2]], dtype=np.float64)
b = np.array([c1, c2], dtype=np.float64)

# Create matrix and vector in C
c_matrix = lib.createMat(dim, dim)
for i in range(dim):
    for j in range(dim):
        c_matrix[i][j] = matrix[i, j]

c_b = (ctypes.c_double * dim)(*b)
c_x = (ctypes.c_double * dim)()

# Perform LU decomposition
lu_result = lib.LU_decomp(c_matrix, dim)

# Extract L and U matrices
L = ctypes.cast(lu_result[0], ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))
U = ctypes.cast(lu_result[1], ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))

# Solve Ax = b
status = lib.solveLU(L, U, c_b, c_x, dim)

# Check the status and process the result
if status == 0:
    print("No solution exists.")
elif status == -1:
    print("Infinite solutions exist.")
else:
    solution = [c_x[i] for i in range(dim)]
    print("Solution vector x:", solution)

# Plot the two lines
x_vals = np.linspace(-5, 5, 500)
line1 = (c1 - a1 * x_vals) / b1
line2 = (c2 - a2 * x_vals) / b2

plt.figure(figsize=(8, 6))
plt.plot(x_vals, line1, label=f"{a1}x + {b1}y = {b[0]}", color='blue')
plt.plot(x_vals, line2, label=f"{a2}x + {b2}y = {b[1]}", color='red')

# Plot the solution point if unique
if status == 1:
    x, y = solution
    plt.plot(x, y, 'go', label=f'Solution: ({x:.2f}, {y:.2f})')
    plt.text(x, y, f"({x:.2f}, {y:.2f})", ha="center")

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('../figs/plot.png')
plt.show()

# Free allocated memory in C
lib.freeMat(c_matrix, dim)
lib.freeMat(L, dim)
lib.freeMat(U, dim)

