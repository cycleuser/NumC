#define NC_IMPLEMENTATION
#include "NumC.h"
#include <stdio.h>

int main() {
    printf("=== Example 8: Random and Comparison ===\n\n");
    
    nc_random_seed(42);
    
    int64_t shape1[1] = {5};
    NCArray *unif = nc_random_rand(1, shape1);
    printf("nc_random_rand(5) - uniform [0,1]: "); nc_print(unif);
    
    NCArray *norm = nc_random_randn(1, shape1, NC_FLOAT64);
    printf("\nnc_random_randn(5) - normal: "); nc_print(norm);
    
    int64_t shape10[1] = {10};
    NCArray *randi = nc_random_randint(0, 100, 1, shape10);
    printf("\nnc_random_randint(0, 100, 10) - integers: "); nc_print(randi);
    
    NCArray *a = NC_INT(1, 2, 3, 4, 5);
    NCArray *b = NC_INT(1, 2, 6, 4, 5);
    NCArray *eq = nc_equal(a, b);
    printf("\na = "); nc_print(a);
    printf("b = "); nc_print(b);
    printf("equal(a, b) = "); nc_print(eq);
    
    NCArray *ne = nc_not_equal(a, b);
    printf("\nnot_equal(a, b) = "); nc_print(ne);
    
    NCArray *lt = nc_less(a, b);
    printf("\nless(a, b) = "); nc_print(lt);
    
    NCArray *gt = nc_greater(a, b);
    printf("\ngreater(a, b) = "); nc_print(gt);
    
    NCArray *logical_and = nc_logical_and(
        NC_INT(0, 1, 1, 0),
        NC_INT(1, 1, 0, 0)
    );
    printf("\nlogical_and([0,1,1,0], [1,1,0,0]) = "); nc_print(logical_and);
    
    NCArray *logical_or = nc_logical_or(
        NC_INT(0, 0, 1, 0),
        NC_INT(0, 1, 0, 0)
    );
    printf("\nlogical_or([0,0,1,0], [0,1,0,0]) = "); nc_print(logical_or);
    
    NCArray *logical_not = nc_logical_not(NC_INT(0, 1, 0, 1));
    printf("\nlogical_not([0,1,0,1]) = "); nc_print(logical_not);
    
    nc_free(unif);
    nc_free(norm);
    nc_free(randi);
    nc_free(a);
    nc_free(b);
    nc_free(eq);
    nc_free(ne);
    nc_free(lt);
    nc_free(gt);
    nc_free(logical_and);
    nc_free(logical_or);
    nc_free(logical_not);
    
    printf("\n[PASS] Example 8 completed successfully!\n");
    return 0;
}