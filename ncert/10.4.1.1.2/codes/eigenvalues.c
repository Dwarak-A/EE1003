#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <stdbool.h>

#define MAX_ITER 1000
#define EPSILON 1e-10

// Matrix structure
typedef struct {
    int n;              // Dimension of matrix
    double* data;       // Data stored in row-major order
} Matrix;

// Matrix utility functions
Matrix* create_matrix(int n) {
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    mat->n = n;
    mat->data = (double*)calloc(n * n, sizeof(double));
    return mat;
}

void free_matrix(Matrix* mat) {
    free(mat->data);
    free(mat);
}

// Get element at position (i,j)
double get(const Matrix* mat, int i, int j) {
    return mat->data[i * mat->n + j];
}

// Set element at position (i,j)
void set(Matrix* mat, int i, int j, double value) {
    mat->data[i * mat->n + j] = value;
}

// Copy matrix source to destination
Matrix* copy_matrix(const Matrix* source) {
    Matrix* dest = create_matrix(source->n);
    memcpy(dest->data, source->data, source->n * source->n * sizeof(double));
    return dest;
}

// Print matrix
void print_matrix(const Matrix* mat) {
    for (int i = 0; i < mat->n; i++) {
        for (int j = 0; j < mat->n; j++) {
            printf("%8.4f ", get(mat, i, j));
        }
        printf("\n");
    }
    printf("\n");
}

// Matrix multiplication: result = A * B
Matrix* multiply_matrices(const Matrix* A, const Matrix* B) {
    if (A->n != B->n) return NULL;
    
    Matrix* result = create_matrix(A->n);
    for (int i = 0; i < A->n; i++) {
        for (int j = 0; j < A->n; j++) {
            double sum = 0;
            for (int k = 0; k < A->n; k++) {
                sum += get(A, i, k) * get(B, k, j);
            }
            set(result, i, j, sum);
        }
    }
    return result;
}

// Compute Frobenius norm of the off-diagonal elements
double off_diagonal_norm(const Matrix* A) {
    double sum = 0;
    for (int i = 0; i < A->n; i++) {
        for (int j = 0; j < A->n; j++) {
            if (i != j) {
                sum += get(A, i, j) * get(A, i, j);
            }
        }
    }
    return sqrt(sum);
}

// Create identity matrix
Matrix* create_identity(int n) {
    Matrix* I = create_matrix(n);
    for (int i = 0; i < n; i++) {
        set(I, i, i, 1.0);
    }
    return I;
}

// Compute QR decomposition using Householder reflections
void qr_decomposition(const Matrix* A, Matrix* Q, Matrix* R) {
    int n = A->n;
    Matrix* H = create_identity(n);
    Matrix* temp_R = copy_matrix(A);
    
    for (int k = 0; k < n - 1; k++) {
        // Compute Householder vector
        double norm = 0;
        for (int i = k; i < n; i++) {
            norm += get(temp_R, i, k) * get(temp_R, i, k);
        }
        norm = sqrt(norm);
        
        if (norm > 0) {
            // First component of v should be norm + x1
            double v1 = get(temp_R, k, k) + (get(temp_R, k, k) >= 0 ? norm : -norm);
            double s = sqrt(v1 * v1 + norm * norm);
            
            // Create Householder matrix
            Matrix* H_k = create_identity(n);
            for (int i = k; i < n; i++) {
                for (int j = k; j < n; j++) {
                    double v_i = (i == k) ? v1 : get(temp_R, i, k);
                    double v_j = (j == k) ? v1 : get(temp_R, j, k);
                    set(H_k, i, j, get(H_k, i, j) - (2 * v_i * v_j) / (s * s));
                }
            }
            
            // Update R
            Matrix* new_R = multiply_matrices(H_k, temp_R);
            free_matrix(temp_R);
            temp_R = new_R;
            
            // Update Q
            Matrix* new_H = multiply_matrices(H_k, H);
            free_matrix(H);
            H = new_H;
            
            free_matrix(H_k);
        }
    }
    
    // Copy results to Q and R
    memcpy(Q->data, H->data, n * n * sizeof(double));
    memcpy(R->data, temp_R->data, n * n * sizeof(double));
    
    free_matrix(H);
    free_matrix(temp_R);
}

// Find eigenvalues using QR algorithm with shifts
void find_eigenvalues(const Matrix* A, double complex* eigenvalues) {
    it n = A->n;
    Matrix* H = copy_matrix(A);
    Matrix* Q = create_matrix(n);
    Matrix* R = create_matrix(n);
    
    int iter = 0;
    while (iter < MAX_ITER && off_diagonal_norm(H) > EPSILON) {
        // Compute shift using Wilkinson shift strategy
        double s = get(H, n-1, n-1);
        
        // Apply shift
        for (int i = 0; i < n; i++) {
            set(H, i, i, get(H, i, i) - s);
        }
        
        // QR decomposition
        qr_decomposition(H, Q, R);
        
        // Form RQ and add shift back
        Matrix* temp = multiply_matrices(R, Q);
        free_matrix(H);
        H = temp;
        
        for (int i = 0; i < n; i++) {
            set(H, i, i, get(H, i, i) + s);
        }
        
        iter++;
    }
    
    // Extract eigenvalues from diagonal
    for (int i = 0; i < n; i++) {
        eigenvalues[i] = get(H, i, i) + 0.0 * I;  // Convert to complex
    }
    
    // Clean up
    free_matrix(H);
    free_matrix(Q);
    free_matrix(R);
}

// Example usage
int main() {
    int n = 4;  // Size of matrix
    
    // Create a test matrix
    Matrix* A = create_matrix(n);
    // Fill with some test values
    double test_values[][4] = {
        {4, -1, 0, -1},
        {-1, 4, -1, 0},
        {0, -1, 4, -1},
        {-1, 0, -1, 4}
    };
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            set(A, i, j, test_values[i][j]);
        }
    }
    
    printf("Original matrix:\n");
    print_matrix(A);
    
    // Calculate eigenvalues
    double complex* eigenvalues = (double complex*)malloc(n * sizeof(double complex));
    find_eigenvalues(A, eigenvalues);
    
    // Print eigenvalues
    printf("Eigenvalues:\n");
    for (int i = 0; i < n; i++) {
        if (cimag(eigenvalues[i]) == 0) {
            printf("λ%d = %.6f\n", i+1, creal(eigenvalues[i]));
        } else {
            printf("λ%d = %.6f %c %.6fi\n", i+1, 
                   creal(eigenvalues[i]),
                   cimag(eigenvalues[i]) > 0 ? '+' : '-',
                   fabs(cimag(eigenvalues[i])));
        }
    }
    
    // Clean up
    free(eigenvalues);
    free_matrix(A);
    
    return 0;
}n
