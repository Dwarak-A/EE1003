#include <stdio.h>
#include <stdlib.h>
#include "functions.h"

// Perform LU decomposition
double*** LU_decomp(double **matrix, int dim) {
    double **L = identity(dim);  // Start L as an identity matrix
    double **U = createMat(dim, dim); // Zero matrix for U

    for (int i = 0; i < dim; i++) {
        for (int j = i; j < dim; j++) {
            U[i][j] = matrix[i][j];
            for (int k = 0; k < i; k++) {
                U[i][j] -= L[i][k] * U[k][j];
            }
        }

        for (int j = i + 1; j < dim; j++) {
            L[j][i] = matrix[j][i];
            for (int k = 0; k < i; k++) {
                L[j][i] -= L[j][k] * U[k][i];
            }
            L[j][i] /= U[i][i];
        }
    }

    double*** toreturn = (double***)malloc(2 * sizeof(double**));
    toreturn[0] = L;
    toreturn[1] = U;
    printf("L is\n");
    printMat(L,dim,dim);
    printf("\nU is\n");
    printMat(U,dim,dim);
    return toreturn;
}

// Solve system of equations Ax = b using LU decomposition
int solveLU(double **L, double **U, double *b, double *x, int dim) {
    double *y = (double*)malloc(dim * sizeof(double)); // Solution for Ly = b

    // Forward substitution: Solve Ly = b
    for (int i = 0; i < dim; i++) {
        y[i] = b[i];
        for (int j = 0; j < i; j++) {
            y[i] -= L[i][j] * y[j];
        }
    }

    // Backward substitution: Solve Ux = y
    for (int i = dim - 1; i >= 0; i--) {
        if (U[i][i] == 0) {
            if (y[i] != 0) {
                free(y);
                return 0; // No solution
            }
            // Infinite solutions detected
            free(y);
            return -1; // Infinite solutions
        }
        x[i] = y[i];
        for (int j = i + 1; j < dim; j++) {
            x[i] -= U[i][j] * x[j];
        }
        x[i] /= U[i][i];
    }

    free(y);
    return 1; // Unique solution
}

int main() {
    int dim = 2;

    // Initialize matrix
    double **matrix = createMat(dim, dim);
    matrix[0][0] = 1; matrix[0][1] = -1;
    matrix[1][0] = 4; matrix[1][1] = -4;

    // Initialize right-hand side vector b
    double b[] = {1, 5};

    // Perform LU decomposition
    double*** result = LU_decomp(matrix, dim);
    
    // Allocate space for the solution vector
    double *x = (double*)malloc(dim * sizeof(double));

    // Solve Ax = b using LU decomposition
    int status = solveLU(result[0], result[1], b, x, dim);

    if (status == 0) {
        printf("No solution exists.\n");
    } 
    else if (status == -1) {
        printf("Infinite solutions exist.\n");
    } 
    else {
        // Print the solution
        printf("\nSolution vector x:\n");
        for (int i = 0; i < dim; i++) {
             printf("%lf\n", x[i]);
        }
    }

    // Free allocated memory
    free(x);
    freeMat(result[0], dim);
    freeMat(result[1], dim);
    free(result);

    // Free matrix memory
    freeMat(matrix, dim);

    return 0;
}

