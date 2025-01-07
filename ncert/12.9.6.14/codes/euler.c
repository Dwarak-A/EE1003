#include <math.h>
#include <stdlib.h>

void euler_method(double x0, double xn, double x_init, double y_init, double h, double* y_theory, double* y_sim, double* x_vals, int num_points) {
	// Finding integration constant
    double integralconst(double x, double y) {
		return y*(1+pow(x, 2)) - atan(x);
	}

    // Function to compute
    double func(double x, double c) {
        return (atan(x))/(1+pow(x,2));
    }

    // Derivative function
    double deriv(double x, double y) { 
        return (-2*x*y)/(1+pow(x,2)) + 1/(pow(1+pow(x,2),2));
    }

    // Euler computation
    double eulerfunc(double x, double y, double h) {
        return y + h * deriv(x, y);
    }

    // Initialize values
    x_vals[0] = x0;
    
    double c = integralconst(x_init, y_init);
    
    y_theory[0] = func(x0, c);
    y_sim[0] = func(x0, c);

    double xi = x0 + h;
    int i = 1;

    // Compute values using Euler's method
    while (xi <= xn) {
        x_vals[i] = xi;
        y_theory[i] = func(xi, c);

       	y_sim[i] = eulerfunc(x_vals[i-1], y_sim[i-1], h);

        xi += h;
        i++;
    }
}

