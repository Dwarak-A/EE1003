#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include "functions.h"

double complex *eigenvalues(double complex **A, int dim) {
    double complex **H = makehessberg(A, dim);
    double complex sigma;
    double tolerance = 1e-10;

    for (int m = dim; m > 1; m--) {
        int iterations = 0;

        while (iterations < 20*dim) {
            iterations++;
            sigma = H[m-1][m-1];
            double complex **sigmaI = Matscale(identity(dim), dim, dim, sigma);
            double complex **H_shifted = Matsub(H, sigmaI, dim, dim);

            double complex **Q_T = identity(dim);

            for (int p = 1; p < m; p++) {
                double complex xi = H_shifted[p-1][p-1];
                double complex xj = H_shifted[p][p-1];
                if (cabs(xj) < tolerance) {
                    continue;
                }

                double complex c = conj(xi)/(csqrt(cabs(xi)*cabs(xi)+cabs(xj)*cabs(xj)));
		double complex s = conj(xj)/(csqrt(cabs(xi)*cabs(xi)+cabs(xj)*cabs(xj)));
		
                double complex **Gi = identity(dim);
                Gi[p-1][p-1] = c;
                Gi[p-1][p] = s;
                Gi[p][p-1] = -conj(s);
                Gi[p][p] = conj(c);
                H_shifted = Matmul(Gi, H_shifted, dim, dim, dim);
                Q_T = Matmul(Q_T, transposeMat(Gi, dim, dim), dim, dim, dim);

                freeMat(Gi, dim);
            }
            H = Matmul(H_shifted, Q_T, dim, dim, dim);
            H = Matadd(H, sigmaI, dim, dim);
            freeMat(sigmaI, dim);
            freeMat(H_shifted, dim);
            freeMat(Q_T, dim);

                if (cabs(H[m-1][m-2]) > tolerance) {
                    continue;
                }
                else{
                	break;
                }
        }
    }

    return calcuppereig(H,dim);
}

// Function to solve a polynomial of a given degree and return its roots
double complex *solve_polynomial(int degree, double complex A[degree+1]) {
    // Check if the leading coefficient is zero
    if (cabs(A[0]) < 1e-12) {
        fprintf(stderr, "Error: Coefficient 'an' cannot be zero.\n");
        return NULL;  // Return NULL to indicate failure
    }

    // Create a companion matrix to find polynomial roots
    double complex **matrix = createMat(degree, degree);

    // Populate the companion matrix
    for (int i = 0; i < degree - 1; i++) {
        matrix[i][i + 1] = 1;  // Set subdiagonal elements to 1
    }
    for (int i = 0; i < degree; i++) {
        matrix[degree - 1][i] = -A[degree - i] / A[0];  // Set last row based on coefficients
    }

    // Find and return the eigenvalues of the companion matrix (roots of the polynomial)
    return eigenvalues(matrix, degree);
}

// Function to solve a quadratic equation ax^2 + bx + c = 0
double complex *solve_quadratic(double complex a, double complex b, double complex c) {
    // Create an array of coefficients for the quadratic polynomial
    double complex coeffs[3] = {a, b, c};

    // Compute the roots using the general polynomial solver
    double complex *result = solve_polynomial(2, coeffs);

    return result;  // Return the roots
}
