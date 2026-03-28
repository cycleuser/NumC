#define NC_IMPLEMENTATION
#include "../NumC.h"
#include <stdio.h>

int main() {
    printf("=== Example 2: Arithmetic Operations ===\n\n");
    
    NCArray *a = NC_INT(1, 2, 3, 4, 5);
    NCArray *b = NC_INT(10, 20, 30, 40, 50);
    
    NCArray *sum = nc_add(a, b);
    NCArray *diff = nc_subtract(b, a);
    NCArray *prod = nc_multiply(a, b);
    NCArray *quot = nc_divide(b, a);
    
    printf("a = "); nc_print(a);
    printf("b = "); nc_print(b);
    printf("\na + b = "); nc_print(sum);
    printf("b - a = "); nc_print(diff);
    printf("a * b = "); nc_print(prod);
    printf("b / a = "); nc_print(quot);
    
    NCArray *pow_result = nc_power(a, NC_INT(1, 2, 3, 4, 5));
    printf("\na ^ [1,2,3,4,5] = "); nc_print(pow_result);
    
    nc_free(a);
    nc_free(b);
    nc_free(sum);
    nc_free(diff);
    nc_free(prod);
    nc_free(quot);
    nc_free(pow_result);
    
    printf("\n[PASS] Example 2 completed successfully!\n");
    return 0;
}