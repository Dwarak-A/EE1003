import ctypes
import numpy as np

# Load the compiled C shared library
lib = ctypes.CDLL('./eigen.so')

# Define the argument and return types for the C function
lib.solve_quadratic.restype = ctypes.POINTER(ctypes.c_double * 4)  # Complex numbers have real and imaginary parts
lib.solve_quadratic.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double, ctypes.c_double
]

# Define coefficients of the quadratic equation
a_real, a_imag = 1.0, 0.0
b_real, b_imag = -4.0, 0.0
c_real, c_imag = 6.0, 0.0

# Call the C function to solve the quadratic equation
result_ptr = lib.solve_quadratic(
    a_real, a_imag,
    b_real, b_imag,
    c_real, c_imag
)

# Extract the results as a list of complex numbers
result_array = result_ptr.contents
root1 = complex(result_array[0], result_array[1])
root2 = complex(result_array[2], result_array[3])

# Print the roots of the quadratic equation
print("Roots of the quadratic equation:")
print(f"Root 1: {root1.real} + {root1.imag}j")
print(f"Root 2: {root2.real} + {root2.imag}j")
