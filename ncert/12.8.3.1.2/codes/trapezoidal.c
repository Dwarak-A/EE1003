#include <math.h>
#include <stdlib.h>

// Generate curve points for y = x^4
void curve_gen(double x0, double xn, double detail, double* x_vals, double* y_vals, int num_points) {
    // Curve function: y = x^4
    double curve(double x) {
        return pow(x, 4);
    }

    // Generate x and y points
    double xi = x0;
    for (int i = 0; i < num_points; i++) {
        x_vals[i] = xi;
        y_vals[i] = curve(xi);
        xi += detail;
    }
}

// Trapezoidal integration using generated curve points
void trapezoidal_method(double x0, double xn, double h, double* area, double* x_vals, double* y_vals, int num_points) {
    // Curve function: y = x^4
    double curve(double x) {
        return pow(x, 4);
    }

    double curve_deriv(double x) {
        return 4 * pow(x, 3);
    }

    // Trapezoid area calculation
    /*
    double trapezoid(double xi, double xj, double h) {
        return (h / 2.0) * (curve(xi) + curve(xj));
    }*/

    double trapezoid(double area, double x, double h) {
        return area + h * pow(x, 4) + 2 * pow(h, 2) * pow(x, 3);
    }

    // Initialize the first point
    x_vals[0] = x0;
    y_vals[0] = curve(x0);
    *area = 0.0;

    double xi = x0 + h;
    int i = 1;

    // Perform trapezoidal integration
    while (xi <= xn && i < num_points) {
        x_vals[i] = xi;
        y_vals[i] = curve(xi);

        //*area += trapezoid(x_vals[i - 1], x_vals[i], h);
        *area = trapezoid(*area, xi, h);

        xi += h;
        i++;
    }
}

