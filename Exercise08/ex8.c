#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 1000 // Number of particles

// Global variables to simulate the shared arrays in the zip file
double x[N], y[N], z[N];
double f_x[N], f_y[N], f_z[N];
double epot, vir; // The 2 reduction variables

// Initialize particles with random positions
void init_particles() {
    for (int i = 0; i < N; i++) {
        x[i] = (double)rand() / RAND_MAX * 10.0;
        y[i] = (double)rand() / RAND_MAX * 10.0;
        z[i] = (double)rand() / RAND_MAX * 10.0;
        f_x[i] = f_y[i] = f_z[i] = 0.0;
    }
}

// This is the function the lab instructs you to parallelize
void forces() {
    double rx, ry, rz, rsq, rsqinv, r6inv, r12inv, force_mag, vij;
    epot = 0.0;
    vir = 0.0;

    /* 
     * 1. Start PARALLEL FOR for the outer loop.
     * 2. Identify 2 reduction variables: epot and vir.
     * 3. Declare distance and math variables as private to each thread.
     */
    #pragma omp parallel for reduction(+:epot, vir) \
        private(rx, ry, rz, rsq, rsqinv, r6inv, r12inv, force_mag, vij) schedule(static, 50)
    for (int i = 0; i < N - 1; i++) {
        for (int j = i + 1; j < N; j++) {
            
            rx = x[i] - x[j];
            ry = y[i] - y[j];
            rz = z[i] - z[j];
            rsq = rx*rx + ry*ry + rz*rz;

            // Calculate Lennard-Jones potential if within cutoff distance
            if (rsq < 25.0) { 
                rsqinv = 1.0 / rsq;
                r6inv = rsqinv * rsqinv * rsqinv;
                r12inv = r6inv * r6inv;
                
                // Potential energy accumulation
                vij = 4.0 * (r12inv - r6inv);
                epot += vij;
                
                // Force magnitude
                force_mag = 24.0 * (2.0 * r12inv - r6inv) * rsqinv;
                
                // Virial energy accumulation
                vir += force_mag * rsq; 

                /* 
                 * Identify variables updated atomically.
                 * Because threads are updating the forces on particle 'j' 
                 * which belongs to the inner loop, we MUST use a critical section 
                 * to prevent data race conditions.
                 */
                #pragma omp critical
                {
                    f_x[i] += rx * force_mag;
                    f_y[i] += ry * force_mag;
                    f_z[i] += rz * force_mag;
                    
                    f_x[j] -= rx * force_mag;
                    f_y[j] -= ry * force_mag;
                    f_z[j] -= rz * force_mag;
                }
            }
        }
    }
}

int main() {
    init_particles();
    
    double tstart = omp_get_wtime();
    
    forces();
    
    double tcalc = (omp_get_wtime() - tstart) * 1000.0; // Time in ms
    
    printf("Successfully calculated forces for %d particles.\n", N);
    printf("Potential Energy (epot) = %f\n", epot);
    printf("Virial Energy (vir) = %f\n", vir);
    printf("Calculation Time = %f ms\n", tcalc);
    
    return 0;
}
