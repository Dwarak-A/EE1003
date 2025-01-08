#include <math.h>
#include <stdlib.h>

void trapezoidal_rule(double x0, double xn, double a, double x_init, double y_init, double h, double* y_theory, double* y_sim, double* x_vals, int num_points) {
	// Finding integration constant
    double integralconst(double x, double y, double a) {
		return y - x * acos(a);
	}

    // Function to compute
    double func(double x, double a, double c) {
        return x * acos(a) + c;
    }

    // Difference Equation
    double diff_eqn(double x, double y, double a, double h) {
        return y + h * acos(a);
    }

    // Initialize values
    x_vals[0] = x0;
    
    double c = integralconst(x_init, y_init, a);
    
    y_theory[0] = func(x0, a, c);
    y_sim[0] = func(x0, a, c);

    double xi = x0 + h;
    int i = 1;

    // Compute values using Euler's method
    while (xi <= xn) {
        x_vals[i] = xi;
        y_theory[i] = func(xi, a, c);

       	y_sim[i] = diff_eqn(x_vals[i-1], y_sim[i-1], a, h);

        xi += h;
        i++;
    }
}

