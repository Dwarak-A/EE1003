import ctypes

# Load the shared library
newton_lib = ctypes.CDLL('./newton_raphson.so')

# Define the argument and return types
newton_lib.newton_raphson_two_roots.argtypes = [
    ctypes.c_double,         # real part of initial guess 1
    ctypes.c_double,         # imaginary part of initial guess 1
    ctypes.c_double,         # real part of initial guess 2
    ctypes.c_double,         # imaginary part of initial guess 2
    ctypes.c_int,            # max_iterations
    ctypes.c_double,         # tolerance
    ctypes.POINTER(ctypes.c_double), # real part of root 1
    ctypes.POINTER(ctypes.c_double), # imaginary part of root 1
    ctypes.POINTER(ctypes.c_int),    # convergence flag for root 1
    ctypes.POINTER(ctypes.c_double), # real part of root 2
    ctypes.POINTER(ctypes.c_double), # imaginary part of root 2
    ctypes.POINTER(ctypes.c_int)     # convergence flag for root 2
]

# Variables
initial_guess1 = complex(1.0, 1.0)  # Initial guess for root 1
initial_guess2 = complex(-1.0, -1.0)  # Initial guess for root 2
max_iterations = 100
tolerance = 1e-6

# Outputs
real_root1 = ctypes.c_double()
imag_root1 = ctypes.c_double()
converged1 = ctypes.c_int()
real_root2 = ctypes.c_double()
imag_root2 = ctypes.c_double()
converged2 = ctypes.c_int()

# Call the function
newton_lib.newton_raphson_two_roots(
    initial_guess1.real,
    initial_guess1.imag,
    initial_guess2.real,
    initial_guess2.imag,
    max_iterations,
    tolerance,
    ctypes.byref(real_root1),
    ctypes.byref(imag_root1),
    ctypes.byref(converged1),
    ctypes.byref(real_root2),
    ctypes.byref(imag_root2),
    ctypes.byref(converged2)
)

# Print results
if converged1.value:
    print(f"Root 1: {real_root1.value} + {imag_root1.value}i")
else:
    print("Root 1: Did not converge")

if converged2.value:
    print(f"Root 2: {real_root2.value} + {imag_root2.value}i")
else:
    print("Root 2: Did not converge")

