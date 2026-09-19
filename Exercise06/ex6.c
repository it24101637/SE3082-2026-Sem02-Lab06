#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000
#define STRIP_SIZE 256 // Aligns with AVX/SIMD vector widths

int main() {
    // Dynamically allocate large arrays to avoid stack overflow
    double *A = (double*)malloc(N * sizeof(double));
    double *B = (double*)malloc(N * sizeof(double));
    double *C = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        A[i] = 1.5;
        B[i] = 2.0;
    }

    double tstart = omp_get_wtime();

    // Distribute the strips across OpenMP threads
    #pragma omp parallel for
    for (int i = 0; i < N; i += STRIP_SIZE) {
        
        // Inner loop processes the strip
        for (int j = i; j < i + STRIP_SIZE && j < N; j++) {
            C[j] = A[j] * B[j];
        }
    }

    double tcalc = (omp_get_wtime() - tstart) * 1000;
    
    printf("Successfully processed %d elements.\n", N);
    printf("Verification: C[0] = %.2f, C[N-1] = %.2f\n", C[0], C[N-1]);
    printf("Calculation Time: %f ms\n", tcalc);

    free(A); free(B); free(C);
    return 0;
}
