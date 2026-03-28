#define NC_IMPLEMENTATION
#include "NumC.h"
#include <stdio.h>

int main() {
    printf("=== Example 3: Matrix Operations ===\n\n");
    
    NCArray *A = NC_INT2D(2, 3, 1, 2, 3, 4, 5, 6);
    NCArray *B = NC_INT2D(2, 3, 6, 5, 4, 3, 2, 1);
    
    printf("Matrix A (2x3):\n");
    nc_print(A);
    printf("Matrix B (2x3):\n");
    nc_print(B);
    
    NCArray *C = NC_INT2D(3, 2, 1, 2, 3, 4, 5, 6);
    printf("Matrix C (3x2):\n");
    nc_print(C);
    
    NCArray *AB = nc_matmul(A, C);
    printf("\nA @ C (matrix multiply, result 2x2):\n");
    nc_print(AB);
    
    NCArray *dot_result = nc_dot(A, C);
    printf("\nnc_dot(A, C):\n");
    nc_print(dot_result);
    
    NCArray *transposeA = nctranspose(A, NULL);
    printf("\nTranspose of A (3x2):\n");
    nc_print(transposeA);
    
    int64_t shape3d[3] = {2, 3, 4};
    NCArray *three_d = nc_zeros(3, shape3d, NC_INT64);
    printf("\n3D array shape: (%lld, %lld, %lld)\n", 
           three_d->shape[0], three_d->shape[1], three_d->shape[2]);
    
    nc_free(A);
    nc_free(B);
    nc_free(C);
    nc_free(AB);
    nc_free(dot_result);
    nc_free(transposeA);
    nc_free(three_d);
    
    printf("\n[PASS] Example 3 completed successfully!\n");
    return 0;
}