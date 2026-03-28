#define NC_IMPLEMENTATION
#include "NumC.h"
#include <stdio.h>

int main() {
    printf("=== Example 6: Array Creation Functions ===\n\n");
    
    int64_t shape1[1] = {5};
    NCArray *zeros = nc_zeros(1, shape1, NC_FLOAT64);
    printf("nc_zeros(5): "); nc_print(zeros);
    
    NCArray *ones = nc_ones(1, shape1, NC_INT32);
    printf("\nnc_ones(5): "); nc_print(ones);
    
    int64_t fill_val = 42;
    NCArray *full = nc_full(1, shape1, &fill_val, NC_INT64);
    printf("\nnc_full(5, 42): "); nc_print(full);
    
    NCArray *arange = nc_arange(0.0, 10.0, 2.0, NC_FLOAT64);
    printf("\nnc_arange(0, 10, 2): "); nc_print(arange);
    
    NCArray *linspace = nc_linspace(0.0, 1.0, 5, true, NC_FLOAT64);
    printf("\nnc_linspace(0, 1, 5): "); nc_print(linspace);
    
    NCArray *identity = nc_identity(4, NC_FLOAT64);
    printf("\nnc_identity(4):\n"); nc_print(identity);
    
    NCArray *eye = nc_eye(3, 4, 0, NC_FLOAT64);
    printf("\nnc_eye(3, 4):\n"); nc_print(eye);
    
    NCArray *diag = nc_diag(NC_INT(1, 2, 3, 4), 0);
    printf("\nnc_diag([1,2,3,4]):\n"); nc_print(diag);
    
    nc_free(zeros);
    nc_free(ones);
    nc_free(full);
    nc_free(arange);
    nc_free(linspace);
    nc_free(identity);
    nc_free(eye);
    nc_free(diag);
    
    printf("\n[PASS] Example 6 completed successfully!\n");
    return 0;
}