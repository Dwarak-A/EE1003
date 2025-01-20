#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>

double **createMat(int m, int n);
double **Matadd(double **a, double **b, int m, int n);
double **Matscale(double **a, int m, int n, double k);
double Matnorm(double **a, int m);
void printMat(double **p, int m, int n);
double **transposeMat(double **a, int m, int n);
double Matdot(double **a, double **b, int m);
double **Matunit(double **a, int m);
double **Matmul(double **a, double **b, int m, int n, int p);
double **Matcol(double **a, int m, int col);
double **Matsub(double **a, double **b, int m, int n);
double **identity(int dim);
double **copyMat(double **src, int m, int n);
void freeMat(double **a, int m);

double **createMat(int m, int n) {
    double **a = (double **)malloc(m * sizeof(*a));
    for (int i = 0; i < m; i++) {
        a[i] = (double *)malloc(n * sizeof(*a[i]));
    }
    for(int i=0;i<m;i++){
    	for(int j=0;j<n;j++){
    		a[i][j] = 0;
    	}
    }
    return a;
}

double **Matadd(double **a, double **b, int m, int n) {
    double **c = createMat(m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    return c;
}

double **Matscale(double **a, int m, int n, double k) {
    double **c = createMat(m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            c[i][j] = k * a[i][j];
        }
    }
    return c;
}

double Matnorm(double **a, int m) {
    double norm = 0;
    for (int i = 0; i < m; i++) {
        norm += creal(a[i][0]) * creal(a[i][0]) + cimag(a[i][0]) * cimag(a[i][0]);
    }
    return sqrt(norm);
}

void printMat(double **p, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.9lf ",p[i][j]);
        }
        printf("\n");
    }
}

double **transposeMat(double **a, int m, int n) {
    double **c = createMat(n, m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            c[i][j] = conj(a[j][i]);
        }
    }
    return c;
}

double Matdot(double **a, double **b, int m) {
    double result = 0;
    for (int i = 0; i < m; i++) {
        result += a[i][0] * conj(b[i][0]);
    }
    return result;
}

double **Matunit(double **a, int m) {
    double **c = createMat(m, 1);
    double norm = Matnorm(a, m);
    for (int i = 0; i < m; i++) {
        c[i][0] = a[i][0] / norm;
    }
    return c;
}

double **Matmul(double **a, double **b, int m, int n, int p) {
    double **c = createMat(m, p);
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

double **Matcol(double **a, int m, int col) {
    double **b = createMat(m, 1);
    for (int i = 0; i < m; i++) {
        b[i][0] = a[i][col];
    }
    return b;
}

double **Matsub(double **a, double **b, int m, int n) {
    double **c = createMat(m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
    return c;
}

double **identity(int dim) {
    double **A = createMat(dim, dim);
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            A[i][j] = (i == j) ? 1 + 0 * I : 0 + 0 * I;
        }
    }
    return A;
}

double **copyMat(double **src, int m, int n) {
    double **dest = createMat(m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            dest[i][j] = src[i][j];
        }
    }
    return dest;
}

void freeMat(double **a, int m) {
    for (int i = 0; i < m; i++) {
        free(a[i]);
    }
    free(a);
}
