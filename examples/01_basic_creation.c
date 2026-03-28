#define NC_IMPLEMENTATION
#include "NumC.h"
#include <stdio.h>

int main() {
    printf("=== Example 1: Basic Array Creation ===\n\n");
    
    NCArray *a = NC_INT(1, 2, 3, 4, 5);
    NCArray *b = NC_FLOAT(1.0, 2.0, 3.0, 4.0, 5.0);
    
    printf("NC_INT(1, 2, 3, 4, 5):\n  dtype: %s\n  shape: (%lld,)\n  data: ", nc_dtype_name(a->dtype), a->shape[0]);
    nc_print(a);
    
    printf("\nNC_FLOAT(1.0, 2.0, 3.0, 4.0, 5.0):\n  dtype: %s\n  shape: (%lld,)\n  data: ", nc_dtype_name(b->dtype), b->shape[0]);
    nc_print(b);
    
    NCArray *c = NC_INT2D(2, 3, 1, 2, 3, 4, 5, 6);
    printf("\nNC_INT2D(2, 3, 1, 2, 3, 4, 5, 6):\n  dtype: %s\n  shape: (%lld, %lld)\n", 
           nc_dtype_name(c->dtype), c->shape[0], c->shape[1]);
    nc_print(c);
    
    nc_free(a);
    nc_free(b);
    nc_free(c);
    
    printf("\n[PASS] Example 1 completed successfully!\n");
    return 0;
}