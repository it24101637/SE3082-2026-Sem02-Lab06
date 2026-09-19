#include <omp.h>
#include <stdio.h>

int fib(int n) {
    int i, j;
    if (n < 2)
        return n;
    else {
        #pragma omp task shared(i)
        i = fib(n - 1);
        
        #pragma omp task shared(j)
        j = fib(n - 2);
        
        #pragma omp taskwait
        return i + j;
    }
}

int main() {
    int n = 20; // Example value
    int result;
    
    double tstart = omp_get_wtime();
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            result = fib(n);
        }
    }
    
    double tcalc = omp_get_wtime() - tstart;
    
    printf("Fibonacci(%d) = %d\n", n, result);
    printf("Time: %f ms\n", tcalc * 1000);
    return 0;
}
