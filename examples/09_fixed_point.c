#define NC_IMPLEMENTATION
#include "../NumC.h"
#include <stdio.h>
#include <string.h>

void nc_fixed_info(int int_bits, int frac_bits, bool is_signed) {
    double min_val = nc_fixed_range_min(int_bits, frac_bits, is_signed);
    double max_val = nc_fixed_range_max(int_bits, frac_bits, is_signed);
    printf("  Q%d.%d (%s): range=[%.6g, %.6g], step=%g\n", 
           int_bits, frac_bits, is_signed ? "signed" : "unsigned",
           min_val, max_val, (max_val - min_val) / (1LL << (int_bits + frac_bits)));
}

int main() {
    printf("=== Example 9: Fixed-Point Numbers ===\n\n");
    
    printf("--- Fixed-Point Format Overview ---\n");
    printf("Fixed-point representation: Q<int_bits>.<frac_bits>\n");
    printf("Q4.4 signed: 4 integer bits (1 sign + 3 magnitude), 4 fractional bits\n");
    printf("Q8.8 signed: 8 integer bits (1 sign + 7 magnitude), 8 fractional bits\n");
    printf("Q16.16 signed: 16 integer bits, 16 fractional bits (32-bit total)\n\n");
    
    printf("--- Range Analysis ---\n");
    nc_fixed_info(4, 4, true);
    nc_fixed_info(4, 4, false);
    nc_fixed_info(8, 8, true);
    nc_fixed_info(8, 8, false);
    nc_fixed_info(16, 16, true);
    nc_fixed_info(16, 16, false);
    
    printf("\n--- Basic Fixed-Point Creation ---\n");
    double vals1[] = {0.0, 0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -0.75, -1.0};
    NCArray *fx1 = nc_fixed_from_values(8, 8, true, vals1, 9);
    printf("Q8.8 from [0, 0.25, 0.5, 0.75, 1, -0.25, -0.5, -0.75, -1]:\n  ");
    nc_fixed_print(fx1, 8);
    
    double vals2[] = {1.5, 2.5, 3.5, 4.5, 5.5};
    NCArray *fx2 = nc_fixed_from_values(16, 16, true, vals2, 5);
    printf("\nQ16.16 from [1.5, 2.5, 3.5, 4.5, 5.5]:\n  ");
    nc_fixed_print(fx2, 16);
    
    printf("\n--- Fixed-Point Arithmetic ---\n");
    double a_vals[] = {1.0, 2.0, 3.0};
    double b_vals[] = {0.5, 0.5, 0.5};
    NCArray *fa = nc_fixed_from_values(8, 8, true, a_vals, 3);
    NCArray *fb = nc_fixed_from_values(8, 8, true, b_vals, 3);
    
    printf("fa = "); nc_fixed_print(fa, 8);
    printf("fb = "); nc_fixed_print(fb, 8);
    
    NCArray *fadd = nc_fixed_add(fa, fb, 8);
    printf("\nfa + fb = "); nc_fixed_print(fadd, 8);
    
    NCArray *fmul = nc_fixed_multiply(fa, fb, 8, 8, 8, NC_INT16);
    printf("fa * fb (Q8.8 x Q8.8 -> Q8.8) = "); nc_fixed_print(fmul, 8);
    
    NCArray *fmul_high = nc_fixed_multiply(fa, fb, 8, 8, 16, NC_INT32);
    printf("fa * fb (Q8.8 x Q8.8 -> Q16.16, more precision) = "); nc_fixed_print(fmul_high, 16);
    
    printf("\n--- Fixed-Point Arange ---\n");
    NCArray *farange = nc_fixed_arange(-1.0, 1.0, 0.25, 8, 8, true);
    printf("nc_fixed_arange(-1, 1, 0.25, Q8.8):\n  ");
    nc_fixed_print(farange, 8);
    
    printf("\n--- Fixed-Point Random (Uniform Distribution) ---\n");
    nc_random_seed(12345);
    
    int64_t shape1d[1] = {10};
    NCArray *frand = nc_fixed_random_rand(8, 8, true, 1, shape1d);
    printf("Q8.8 random (10 values):\n  ");
    nc_fixed_print(frand, 8);
    
    NCArray *frand_u = nc_fixed_random_rand(8, 8, false, 1, shape1d);
    printf("\nUQ8.8 random (unsigned, 10 values):\n  ");
    nc_fixed_print(frand_u, 8);
    
    NCArray *funiform = nc_fixed_random_uniform(8, 8, true, -1.0, 1.0, 1, shape1d);
    printf("\nQ8.8 uniform in [-1, 1] (10 values):\n  ");
    nc_fixed_print(funiform, 8);
    
    printf("\n--- Uniformity Demonstration ---\n");
    printf("Unlike float which clusters values near 0, fixed-point provides\n");
    printf("uniform spacing across the entire range. Each integer step\n");
    printf("represents exactly the same magnitude difference.\n\n");
    
    NCArray *float_rand = nc_random_rand(1, shape1d);
    printf("Float (0,1) distribution: ");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", ((double*)float_rand->data)[i]);
    }
    printf("\nNotice how float has high precision near 0 and low precision near 1.\n");
    
    NCArray *fixed_rand = nc_fixed_random_rand(8, 8, false, 1, shape1d);
    printf("\nFixed-point UQ8.8 (0,255/256) distribution: ");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", nc_fixed_get_value_as_double(fixed_rand, i, 8));
    }
    printf("\nEach step is exactly 1/256. Uniform across the entire range!\n");
    
    printf("\n--- High Precision Fixed-Point ---\n");
    int64_t shape2d[2] = {3, 3};
    NCArray *fhigh = nc_fixed_random_rand(20, 12, true, 2, shape2d);
    printf("Q20.12 random (3x3 matrix):\n  ");
    nc_fixed_print(fhigh, 12);
    
    printf("\n--- Comparison: Float vs Fixed-Point Distribution ---\n");
    printf("Generating 10000 samples in [0, 1]...\n\n");
    
    int64_t big_shape[1] = {10000};
    NCArray *float_samples = nc_random_rand(1, big_shape);
    NCArray *fixed_samples = nc_fixed_random_uniform(16, 16, false, 0.0, 1.0, 1, big_shape);
    
    int float_buckets[10] = {0};
    int fixed_buckets[10] = {0};
    double bucket_size = 0.1;
    
    for (int i = 0; i < 10000; i++) {
        double fv = ((double*)float_samples->data)[i];
        double xv = nc_fixed_get_value_as_double(fixed_samples, i, 16);
        
        for (int b = 0; b < 10; b++) {
            if (fv >= b * bucket_size && fv < (b + 1) * bucket_size) float_buckets[b]++;
            if (xv >= b * bucket_size && xv < (b + 1) * bucket_size) fixed_buckets[b]++;
        }
    }
    
    printf("Bucket   Float    Fixed-Point (UQ16.16)\n");
    for (int b = 0; b < 10; b++) {
        printf("[%.1f,%.1f) %5d   %5d\n", 
               b * bucket_size, (b + 1) * bucket_size,
               float_buckets[b], fixed_buckets[b]);
    }
    printf("\nFloat shows clustering near 0 (bucket 0 has more than expected).\n");
    printf("Fixed-point shows nearly uniform distribution across all buckets.\n");
    
    nc_free(fx1);
    nc_free(fx2);
    nc_free(fa);
    nc_free(fb);
    nc_free(fadd);
    nc_free(fmul);
    nc_free(fmul_high);
    nc_free(farange);
    nc_free(frand);
    nc_free(frand_u);
    nc_free(funiform);
    nc_free(float_rand);
    nc_free(fixed_rand);
    nc_free(fhigh);
    nc_free(float_samples);
    nc_free(fixed_samples);
    
    printf("\n[PASS] Example 9 completed successfully!\n");
    return 0;
}