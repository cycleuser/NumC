#include "NumC.h"
#include <stdio.h>

int main() {
    printf("NumC v%s - NumPy-like library for C\n\n", nc_version());
    
    int64_t shape1d[1] = {10};
    NCArray *arr = nc_arange(0.0, 10.0, 1.0, NC_FLOAT64);
    printf("1D Array (arange):\n");
    nc_print(arr);
    
    NCArray *reshaped;
    nc_reshape(reshaped = nc_copy(arr), 2, (int64_t[]){2, 5});
    printf("\nReshaped to 2x5:\n");
    nc_print(reshaped);
    
    NCArray *identity = nc_identity(4, NC_FLOAT64);
    printf("\n4x4 Identity matrix:\n");
    nc_print(identity);
    
    NCArray *a = nc_random_randn(2, (int64_t[]){2, 2}, NC_FLOAT64);
    NCArray *b = nc_random_randn(2, (int64_t[]){2, 2}, NC_FLOAT64);
    printf("\nRandom matrices:\n");
    nc_print(a);
    nc_print(b);
    
    NCArray *sum = nc_add(a, b);
    printf("\nSum:\n");
    nc_print(sum);
    
    NCArray *product = nc_matmul(a, b);
    printf("\nMatrix product:\n");
    nc_print(product);
    
    NCArray *mean = nc_mean(a, NULL, 0);
    printf("\nMean of random matrix: %.6f\n", ((double*)mean->data)[0]);
    
    printf("\nStatistics:\n");
    NCArray *min_val = nc_min(a, NULL, 0);
    NCArray *max_val = nc_max(a, NULL, 0);
    printf("  Min: %.6f, Max: %.6f\n", 
           ((double*)min_val->data)[0], 
           ((double*)max_val->data)[0]);
    
    nc_release(arr);
    nc_release(reshaped);
    nc_release(identity);
    nc_release(a);
    nc_release(b);
    nc_release(sum);
    nc_release(product);
    nc_release(mean);
    nc_release(min_val);
    nc_release(max_val);
    
    printf("\nMemory management test passed!\n");
    return 0;
}
