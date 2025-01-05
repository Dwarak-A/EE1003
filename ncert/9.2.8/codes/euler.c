#include <math.h>
#include <stdlib.h>

void euler_method(double y0, double yn, double h, double* x_given, double* x_approx, double* y_values, int num_points) {
    // Function to compute f(y) = y - cos(y)
    double func(double y) {
        return y - cos(y);
    }

    // Derivative function
    double deriv(double x, double y) { 
        return sin(y) + ((cos(y) + x) / y);
    }

    // Euler computation
    double eulerfunc(double x, double y, double h) {
        return x + h * deriv(x, y);
    }

    // Initialize values
    y_values[0] = y0;
    x_given[0] = func(y0);
    x_approx[0] = func(y0);

    double yi = y0 + h;
    int i = 1;

    // Compute values using Euler's method
    while (yi <= yn) {
        y_values[i] = yi;
        x_given[i] = func(yi);

        if (fabs(yi) < 1e-10) {
            x_approx[i] = eulerfunc(func(yi), yi, h);
        } else {
            x_approx[i] = eulerfunc(x_approx[i - 1], yi, h);
        }

        yi += h;
        i++;
    }
}

