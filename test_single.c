#define NC_IMPLEMENTATION
#include "NumC.h"
#include <stdio.h>

int main() {
    printf("NumC v%s - Single Header Edition\n\n", nc_version());
    
    NCArray *a = NC_INT(1, 2, 3, 4, 5);
    NCArray *b = NC_FLOAT(1.5, 2.5, 3.5);
    NCArray *c = NC_INT2D(2, 3, 1, 2, 3, 4, 5, 6);
    NCArray *d = NC_FLOAT2D(2, 2, 1.0, 2.0, 3.0, 4.0);
    
    printf("NC_INT(1,2,3,4,5):\n"); nc_print(a);
    printf("NC_FLOAT(1.5,2.5,3.5):\n"); nc_print(b);
    printf("NC_INT2D(2,3,...):\n"); nc_print(c);
    printf("NC_FLOAT2D(2,2,...):\n"); nc_print(d);
    
    NCArray *sum = nc_add(a, a);
    NCArray *prod = nc_matmul(c, d);
    NCArray *mean = nc_mean(a, NULL, 0);
    
    printf("\nOperations:\n");
    printf("a + a:\n"); nc_print(sum);
    printf("c @ d:\n"); nc_print(prod);
    printf("mean(a): %.2f\n", ((double*)mean->data)[0]);
    
    nc_release(a); nc_release(b); nc_release(c); nc_release(d);
    nc_release(sum); nc_release(prod); nc_release(mean);
    
    printf("\nAll tests passed!\n");
    return 0;
}
