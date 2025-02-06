#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <openssl/rand.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function to generate a random indicator (0 or 1) based on probability p */
int random_number(double p) {
    unsigned char random_byte;
    if (RAND_bytes(&random_byte, 1) != 1) {
        fprintf(stderr, "Error generating random byte\n");
        exit(1);
    }
    double random_value = (double)random_byte / 255.0;
    return random_value < p ? 1 : 0;
}

/* Function to compute probabilities using Monte Carlo simulation */
float* compute_probabilities(float p_A, float p_B, float p_AB, int n) {
    static float arr[4];  // Array to store computed probabilities
    int count_A = 0, count_B = 0, count_AB = 0, count_AcBc = 0;

    for (int i = 0; i < n; i++) {
        int A = random_number(p_A);
        int B = random_number(p_B);
        int AB = random_number(p_AB); // Independent sampling for P(AB)
        
        int Ac = 1 - A;
        int Bc = 1 - B;

        count_A += A;
        count_B += B;
        count_AB += AB;
        count_AcBc += (Ac & Bc); // Counting occurrences where neither event occurs
    }

    // Storing computed probabilities
    arr[0] = (float)count_A / n;    // P(A)
    arr[1] = (float)count_B / n;    // P(B)
    arr[2] = (float)count_AB / n;   // P(A ∩ B)
    arr[3] = (float)count_AcBc / n; // P(A' ∩ B')

    return arr;
}

/* Function to free allocated memory (not needed here, included for compatibility) */
void freeMemory(float* arr) {
    // No dynamic memory allocation is used, so nothing to free.
}

#ifdef __cplusplus
}
#endif

