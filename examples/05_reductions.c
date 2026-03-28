#define NC_IMPLEMENTATION
#include "../NumC.h"
#include <stdio.h>

int main() {
    printf("=== Example 5: Reduction Operations ===\n\n");
    
    NCArray *a = NC_INT(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    
    printf("Array a = "); nc_print(a);
    
    NCArray *sum = nc_sum(a, NULL, 0);
    printf("\nsum(a) = %.0f", nc_get_value_as_double(sum, 0));
    
    NCArray *mean = nc_mean(a, NULL, 0);
    printf("\nmean(a) = %.1f", nc_get_value_as_double(mean, 0));
    
    NCArray *var = nc_var(a, NULL, 0);
    printf("\nvar(a) = %.1f", nc_get_value_as_double(var, 0));
    
    NCArray *std = nc_std(a, NULL, 0);
    printf("\nstd(a) = %.1f", nc_get_value_as_double(std, 0));
    
    NCArray *min = nc_min(a, NULL, 0);
    printf("\nmin(a) = %.0f", nc_get_value_as_double(min, 0));
    
    NCArray *max = nc_max(a, NULL, 0);
    printf("\nmax(a) = %.0f", nc_get_value_as_double(max, 0));
    
    int axis = 0;
    NCArray *prod = nc_prod(a, &axis, 1);
    printf("\nprod(a) = %.0f", nc_get_value_as_double(prod, 0));
    
    NCArray *cumsum = nc_cumsum(a, 0);
    printf("\ncumsum(a) = "); nc_print(cumsum);
    
    NCArray *argsort_idx = nc_argmax(a, 0);
    printf("\nargmax(a) = %.0f", nc_get_value_as_double(argsort_idx, 0));
    
    NCArray *any_result = nc_any(NC_INT(0, 0, 1, 0), NULL, 0);
    printf("\nany([0,0,1,0]) = %s", nc_get_value_as_double(any_result, 0) ? "true" : "false");
    
    NCArray *all_result = nc_all(NC_INT(1, 1, 1, 1), NULL, 0);
    printf("\nall([1,1,1,1]) = %s", nc_get_value_as_double(all_result, 0) ? "true" : "false");
    
    nc_free(a);
    nc_free(sum);
    nc_free(mean);
    nc_free(var);
    nc_free(std);
    nc_free(min);
    nc_free(max);
    nc_free(prod);
    nc_free(cumsum);
    nc_free(argsort_idx);
    nc_free(any_result);
    nc_free(all_result);
    
    printf("\n\n[PASS] Example 5 completed successfully!\n");
    return 0;
}