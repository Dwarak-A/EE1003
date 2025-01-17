#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>

double complex **createMat(int m, int n);
double complex **Matadd(double complex **a, double complex **b, int m, int n);
double complex **Matscale(double complex **a, int m, int n, double complex k);
double Matnorm(double complex **a, int m);
void printMat(double complex **p, int m, int n);
double complex **transposeMat(double complex **a, int m, int n);
double complex Matdot(double complex **a, double complex **b, int m);
double complex **Matunit(double complex **a, int m);
double complex **Matmul(double complex **a, double complex **b, int m, int n, int p);
double complex **Matcol(double complex **a, int m, int col);
double complex **Matsub(double complex **a, double complex **b, int m, int n);
double complex **identity(int dim);
double complex **copyMat(double complex **src, int m, int n);
void freeMat(double complex **a, int m);
double complex **makehessberg(double complex **A, int dim);

double complex **createMat(int m, int n) {
    double complex **a = (double complex **)malloc(m * sizeof(*a));
    for (int i = 0; i < m; i++) {
        a[i] = (double complex *)malloc(n * sizeof(*a[i]));
    }
    for(int i=0;i<m;i++){
    	for(int j=0;j<n;j++){
    		a[i][j] = 0;
    	}
    }
    return a;
}

double complex **Matadd(double complex **a, double complex **b, int m, int n) {
    double complex **c = createMat(m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    return c;
}

double complex **Matscale(double complex **a, int m, int n, double complex k) {
    double complex **c = createMat(m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            c[i][j] = k * a[i][j];
        }
    }
    return c;
}

double Matnorm(double complex **a, int m) {
    double norm = 0;
    for (int i = 0; i < m; i++) {
        norm += creal(a[i][0]) * creal(a[i][0]) + cimag(a[i][0]) * cimag(a[i][0]);
    }
    return sqrt(norm);
}

void printMat(double complex **p, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("(%.9lf + %.9lfi) ", creal(p[i][j]), cimag(p[i][j]));
        }
        printf("\n");
    }
}

double complex **transposeMat(double complex **a, int m, int n) {
    double complex **c = createMat(n, m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            c[i][j] = conj(a[j][i]);
        }
    }
    return c;
}

double complex Matdot(double complex **a, double complex **b, int m) {
    double complex result = 0;
    for (int i = 0; i < m; i++) {
        result += a[i][0] * conj(b[i][0]);
    }
    return result;
}

double complex **Matunit(double complex **a, int m) {
    double complex **c = createMat(m, 1);
    double norm = Matnorm(a, m);
    for (int i = 0; i < m; i++) {
        c[i][0] = a[i][0] / norm;
    }
    return c;
}

double complex **Matmul(double complex **a, double complex **b, int m, int n, int p) {
    double complex **c = createMat(m, p);
    for (int i = 0; i < m; i++) {
        for (int k = 0; k < p; k++) {
            c[i][k] = 0;
            for (int j = 0; j < n; j++) {
                c[i][k] += a[i][j] * b[j][k];
            }
        }
    }
    return c;
}

double complex **Matcol(double complex **a, int m, int col) {
    double complex **b = createMat(m, 1);
    for (int i = 0; i < m; i++) {
        b[i][0] = a[i][col];
    }
    return b;
}

double complex **Matsub(double complex **a, double complex **b, int m, int n) {
    double complex **c = createMat(m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
    return c;
}

double complex **identity(int dim) {
    double complex **A = createMat(dim, dim);
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            A[i][j] = (i == j) ? 1 + 0 * I : 0 + 0 * I;
        }
    }
    return A;
}

double complex **copyMat(double complex **src, int m, int n) {
   double complex **dest = createMat(m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            dest[i][j] = src[i][j];
        }
    }
    return dest;
}

void freeMat(double complex **a, int m) {
    for (int i = 0; i < m; i++) {
        free(a[i]);
    }
    free(a);
}

double complex **makehessberg(double complex **A, int dim) {
    // Copy the original matrix to prevent modifying A directly
    double complex **toreturn = copyMat(A, dim, dim);

    for (int k = 0; k < dim - 2; k++) {
        int u_dim = dim - k - 1;
        double complex **u = createMat(u_dim, 1);

        // Extract subvector from toreturn (column below the diagonal element)
        for (int i = 0; i < u_dim; i++) {
            u[i][0] = toreturn[k + 1 + i][k];
        }
        
        // Apply sign to make u's first element larger for stability
        if(cabs(u[0][0])!=0){
        	double complex rho = - u[0][0] / cabs(u[0][0]);
        	u[0][0] += -1 * rho * Matnorm(u, u_dim);
        }
        
        // Normalize u
        double norm_u = Matnorm(u, u_dim);
        if (norm_u != 0) {
            for (int i = 0; i < u_dim; i++) {
                u[i][0] /= norm_u;
            }
        }

        // Construct Householder reflector Hk
        double complex **Hk = identity(dim);
        double complex **u_transpose = transposeMat(u, u_dim, 1);
        double complex **u_ut = Matmul(u, u_transpose, u_dim, 1, u_dim);

        // Update Hk within the relevant submatrix
        for (int i = 0; i < u_dim; i++) {
            for (int j = 0; j < u_dim; j++) {
                Hk[k + 1 + i][k + 1 + j] -= 2 * u_ut[i][j];
            }
        }
        
        // Apply similarity transformation Hk * toreturn * Hk
        double complex **temp1 = Matmul(Hk, toreturn, dim, dim, dim);
        double complex **temp2 = Matmul(temp1, Hk, dim, dim, dim);

        // Replace toreturn with the updated matrix
        freeMat(toreturn, dim);
        toreturn = temp2;
        
        // Free intermediate matrices
        freeMat(temp1, dim);
        freeMat(u, u_dim);
        freeMat(u_transpose, 1);
        freeMat(u_ut, u_dim);
        freeMat(Hk, dim);
    }

    return toreturn;
}

double complex *calcuppereig(double complex **A, int dim) {
    // Allocate memory to store eigenvalues
    double complex *eigenvalues = (double complex *)malloc(dim * sizeof(double complex));
    int count = 0;

    for (int i = 0; i < dim; i++) {
        // Check if the element below the diagonal is zero
        if (i < dim - 1 && cabs(A[i + 1][i]) > 1e-12) { // Non-zero entry, handle 2x2 block
            double complex x1 = A[i][i];
            double complex x2 = A[i + 1][i + 1];
            double complex y1 = A[i + 1][i];
            double complex y2 = A[i][i + 1];

            // Coefficients of the quadratic equation: x^2 - (x1+x2)x + (x1x2-y1y2) = 0
            double complex a = 1.0;
            double complex b = -(x1 + x2);
            double complex c = x1 * x2 - y1 * y2;

            // Compute the roots
            double complex eigenvalue1 = (-b + csqrt(b * b - 4.0 * a * c)) / (2.0 * a);
            double complex eigenvalue2 = (-b - csqrt(b * b - 4.0 * a * c)) / (2.0 * a);

            // Store the eigenvalues
            eigenvalues[count++] = (cabs(eigenvalue1) > 1e-8) ? eigenvalue1 : 0.0 + 0.0 * I;
            eigenvalues[count++] = (cabs(eigenvalue2) > 1e-8) ? eigenvalue2 : 0.0 + 0.0 * I;

            // Skip the next row since it's part of the 2x2 block
            i++;
        } else { // Diagonal element is an eigenvalue
            eigenvalues[count++] = (cabs(A[i][i]) > 1e-12) ? A[i][i] : 0.0 + 0.0 * I;
        }
    }

    return eigenvalues;
}
