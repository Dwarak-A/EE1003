import cmath
import time

start_time = time.time()

# Helper Functions
def multiply_matrices(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    assert cols_A == rows_B, "Incompatible dimensions for matrix multiplication."

    C = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]

    return C


def transpose_matrix(A):
    """Returns the transpose of matrix A."""
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


def compute_norm(v):
    """Calculates the Euclidean (or Frobenius) norm of a vector v."""
    return sum(x.real**2 + x.imag**2 for x in v)**0.5


# Givens Rotation Helper
def givens_rotation(a, b):
    # Calculates the cosine (c) and sine (s) for a Givens rotation for complex values
    if b == 0:
        return 1, 0  # c=1, s=0
    elif abs(b) > abs(a):
        t = -a / b
        s = 1 / cmath.sqrt(1 + t**2)
        c = s * t
    else:
        t = -b / a
        c = 1 / cmath.sqrt(1 + t**2)
        s = c * t
    return c, s


def apply_rotation(A, c, s, i, k, transpose=False):
    # Givens rotation to rows or columns of matrix A
    n = len(A[0]) if not transpose else len(A)
    for j in range(n):
        if not transpose:
            temp = c * A[i][j] + s * A[k][j]
            A[k][j] = -s * A[i][j] + c * A[k][j]
            A[i][j] = temp
        else:
            temp = c * A[j][i] + s * A[j][k]
            A[j][k] = -s * A[j][i] + c * A[j][k]
            A[j][i] = temp


def qr_decompose(A):
    # QR decomposition of matrix A using Givens rotations
    m, n = len(A), len(A[0])
    
    # Initialize Q as an identity matrix and R as A
    Q = [[1 if i == j else 0 for j in range(m)] for i in range(m)]
    R = [row[:] for row in A]  # Make a copy of A to store R
    
    # Apply Givens rotations to R
    for j in range(n):
        for i in range(j+1, m):
            if R[i][j] != 0:
                c, s = givens_rotation(R[j][j], R[i][j])
                
                # Rotate the rows of R
                for k in range(j, n):
                    R[j][k], R[i][k] = c * R[j][k] - s * R[i][k], s * R[j][k] + c * R[i][k]
                
                # Rotate the columns of Q
                for k in range(m):
                    Q[k][j], Q[k][i] = c * Q[k][j] - s * Q[k][i], s * Q[k][j] + c * Q[k][i]
    
    return Q, R


def shift_matrix(A, shift):
    # Apply a shift to matrix A
    n = len(A)
    return [[A[i][j] - (shift if i == j else 0) for j in range(n)] for i in range(n)]


def compute_wilkinson_shift(A):
    # Computes the Wilkinson shift for a matrix
    n = len(A)
    if n < 2:
        return A[0][0]

    a = A[n-2][n-2]
    b = A[n-2][n-1]
    c = A[n-1][n-2]
    d = A[n-1][n-1]

    tr = a + d
    det = a * d - b * c
    disc = cmath.sqrt(tr * tr - 4 * det)

    sigma1 = (tr + disc) / 2
    sigma2 = (tr - disc) / 2

    return sigma1 if abs(sigma1 - d) < abs(sigma2 - d) else sigma2


# Hessenberg Reduction
def reduce_to_hessenberg(A):
    # Reduces matrix A to upper Hessenberg form
    n = len(A)

    for k in range(n - 2):
        for i in range(k + 2, n):
            if abs(A[i][k]) > 1e-12:
                c, s = givens_rotation(A[k+1][k], A[i][k])

                for j in range(k, n):
                    A[k + 1][j], A[i][j] = (
                        c * A[k + 1][j] - s * A[i][j],
                        s * A[k + 1][j] + c * A[i][j],
                    )

                for j in range(n):
                    A[j][k + 1], A[j][i] = (
                        c * A[j][k + 1] - s * A[j][i],
                        s * A[j][k + 1] + c * A[j][i],
                    )

    return A


# Eigenvalue Computation
def format_complex_number(z):
    # Formats a complex number for display.
    return f"{z.real:.8f}{'+' if z.imag >= 0 else ''}{z.imag:.8f}j"


def compute_eigenvalues(A, tol=1e-6, max_iters=10000):
    # Finds eigenvalues of A using the QR algorithm with Wilkinson shift.
    A = reduce_to_hessenberg(A)
    n = len(A)
    eigenvalues = []
    total_iterations = 0

    while n > 1:
        iterations = 0
        while iterations < max_iters:
            if abs(A[n-1][n-2]) < tol:
                eigenval = A[n-1][n-1]
                eigenvalues.append(eigenval)
                n -= 1
                A = [[A[i][j] for j in range(n)] for i in range(n)]
                break

            shift = compute_wilkinson_shift(A)
            A_shifted = shift_matrix(A, shift)

            Q, R = qr_decompose(A_shifted)

            A = multiply_matrices(R, Q)
            A = shift_matrix(A, -shift)

            iterations += 1
            total_iterations += 1

            if iterations == max_iters:
                print(f"\nWarning: Maximum iterations ({max_iters}) reached for current eigenvalue")

    if n == 1:
        eigenval = A[0][0]
        eigenvalues.append(eigenval)

    print(f"Total iterations for convergence : {total_iterations}\n")
    return list(reversed(eigenvalues))

b = -4
c = 6

A = [
    [0, -c],
    [1, -b]
    ]
    
eigenvalues = compute_eigenvalues(A)
for i, eigenvalue in enumerate(eigenvalues, start=1):
    print(f"Eigenvalue {i}: {format_complex_number(eigenvalue)}")
    
end_time = time.time()
run_time = end_time - start_time
print(f"\nCalculation time: {run_time:.6f} seconds")

