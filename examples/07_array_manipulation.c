#define NC_IMPLEMENTATION
#include "NumC.h"
#include <stdio.h>

int main() {
    printf("=== Example 7: Array Manipulation ===\n\n");
    
    NCArray *a = NC_INT(1, 2, 3, 4, 5, 6);
    printf("Original a = "); nc_print(a);
    
    int64_t new_shape[2] = {2, 3};
    NCArray *reshaped = nc_empty(2, new_shape, NC_UINT8);
    for (int i = 0; i < 6; i++) {
        nc_set_value_from_double(reshaped, i, i + 1);
    }
    printf("\nreshape to (2, 3):\n"); nc_print(reshaped);
    
    NCArray *flat = ncflatten(reshaped);
    printf("\nflatten() = "); nc_print(flat);
    
    NCArray *transposed = nctranspose(reshaped, NULL);
    printf("\ntranspose() (3x2):\n"); nc_print(transposed);
    
    NCArray *a2 = NC_INT(7, 8, 9);
    NCArray *arrays[2] = {a, a2};
    NCArray *concat = nc_concatenate(arrays, 2, 0);
    printf("\nconcatenate([1,2,3,4,5,6], [7,8,9]) = "); nc_print(concat);
    
    nc_free(a);
    nc_free(reshaped);
    nc_free(flat);
    nc_free(transposed);
    nc_free(a2);
    nc_free(concat);
    
    printf("\n[PASS] Example 7 completed successfully!\n");
    return 0;
}