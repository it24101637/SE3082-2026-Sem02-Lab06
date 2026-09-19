#include <stdio.h>
#include <omp.h>

#define NPOINTS 1000
#define MAXITER 2000

int main() {
    int numoutside = 0;
    double area;
    double tstart, tcalc;

    tstart = omp_get_wtime();

    /* 
     * 1. Start parallel region before the main loop.
     * 2. Distribute outermost loop across threads.
     * 3. numoutside is a reduction variable.
     */
    #pragma omp parallel for reduction(+:numoutside) schedule(dynamic)
    for (int i = 0; i < NPOINTS; i++) {
        for (int j = 0; j < NPOINTS; j++) {
            // Complex plane bounds containing the upper half of the set
            double c_real = -2.0 + 2.5 * (double)i / (double)NPOINTS;
            double c_imag = 1.125 * (double)j / (double)NPOINTS; 
            
            double z_real = c_real;
            double z_imag = c_imag;
            double temp;
            int iter;

            // Iterate z = z^2 + c
            for (iter = 0; iter < MAXITER; iter++) {
                double z2_real = z_real * z_real;
                double z2_imag = z_imag * z_imag;
                
                // If |z| > 2 (or |z|^2 > 4), it's outside the set
                if (z2_real + z2_imag > 4.0) {
                    numoutside++;
                    break;
                }
                
                temp = z2_real - z2_imag + c_real;
                z_imag = 2.0 * z_real * z_imag + c_imag;
                z_real = temp;
            }
        }
    }

    tcalc = (omp_get_wtime() - tstart) * 1000.0; // Time in ms

    // Calculate Area: 
    // Area of our sample box (2.5 width * 1.125 height). 
    // Multiply by 2 because we only calculated the symmetric upper half.
    double total_points = (double)NPOINTS * (double)NPOINTS;
    double area_of_box = 2.5 * 1.125 * 2.0;
    double points_inside = total_points - numoutside;
    area = area_of_box * (points_inside / total_points);

    printf("Points outside: %d\n", numoutside);
    printf("Estimated Area of Mandelbrot set = %f\n", area);
    printf("Calculation Time = %f ms\n", tcalc);

    return 0;
}

