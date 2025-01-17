#include <stdio.h>
#include <complex.h>

// Define the function f(x) = x^2 + 4
double complex f(double complex x) {
    return x * x - 4 * x + 6;
}

// Define the derivative f'(x) = 2x
double complex f_prime(double complex x) {
    return 2 * x - 4;
}

// Newton-Raphson method for finding complex roots
void newton_raphson_two_roots(
    double real_guess1, double imag_guess1,
    double real_guess2, double imag_guess2,
    int max_iterations, double tolerance,
    double* real_root1, double* imag_root1, int* converged1,
    double* real_root2, double* imag_root2, int* converged2) {

    double complex guesses[2] = { real_guess1 + imag_guess1 * I, real_guess2 + imag_guess2 * I };
    double complex roots[2];
    int converged[2] = {0, 0};

    for (int i = 0; i < 2; i++) {
        double complex x = guesses[i];
        for (int j = 0; j < max_iterations; j++) {
            double complex fx = f(x);
            double complex fpx = f_prime(x);

            // Check for small derivative
            if (cabs(fpx) < 1e-10) {
                converged[i] = 0;
                break;
            }

            // Update step
            double complex x_new = x - fx / fpx;

            // Check convergence
            if (cabs(x_new - x) < tolerance) {
                roots[i] = x_new;
                converged[i] = 1;
                break;
            }

            x = x_new;
        }

        // If not converged within max_iterations
        if (!converged[i]) {
            roots[i] = x;
        }
    }

    // Set results for root 1
    *real_root1 = creal(roots[0]);
    *imag_root1 = cimag(roots[0]);
    *converged1 = converged[0];

    // Set results for root 2
    *real_root2 = creal(roots[1]);
    *imag_root2 = cimag(roots[1]);
    *converged2 = converged[1];
}

