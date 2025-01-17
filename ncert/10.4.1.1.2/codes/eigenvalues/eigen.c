#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include "functions.h"

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
