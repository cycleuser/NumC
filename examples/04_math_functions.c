#define NC_IMPLEMENTATION
#include "../NumC.h"
#include <stdio.h>
#include <math.h>

int main() {
    printf("=== Example 4: Mathematical Functions ===\n\n");
    
    NCArray *x = NC_FLOAT(0.0, M_PI/6.0, M_PI/4.0, M_PI/3.0, M_PI/2.0);
    printf("x = "); nc_print(x);
    
    NCArray *sin_x = nc_sin(x);
    printf("\nsin(x) = "); nc_print(sin_x);
    
    NCArray *cos_x = nc_cos(x);
    printf("\ncos(x) = "); nc_print(cos_x);
    
    NCArray *tan_x = nc_tan(x);
    printf("\ntan(x) = "); nc_print(tan_x);
    
    NCArray *exp_x = nc_exp(x);
    printf("\nexp(x) = "); nc_print(exp_x);
    
    NCArray *log_x = nc_log(NC_FLOAT(1.0, 2.0, 3.0, 4.0, 5.0));
    printf("\nlog([1,2,3,4,5]) = "); nc_print(log_x);
    
    NCArray *sqrt_x = nc_sqrt(NC_FLOAT(1.0, 4.0, 9.0, 16.0, 25.0));
    printf("\nsqrt([1,4,9,16,25]) = "); nc_print(sqrt_x);
    
    NCArray *abs_x = nc_abs(NC_FLOAT(-1.0, -2.0, -3.0, 4.0, 5.0));
    printf("\nabs([-1,-2,-3,4,5]) = "); nc_print(abs_x);
    
    NCArray *pow_x = nc_power(NC_FLOAT(2.0, 2.0, 2.0, 2.0, 2.0), 
                              NC_FLOAT(0.0, 1.0, 2.0, 3.0, 4.0));
    printf("\n2^[0,1,2,3,4] = "); nc_print(pow_x);
    
    nc_free(x);
    nc_free(sin_x);
    nc_free(cos_x);
    nc_free(tan_x);
    nc_free(exp_x);
    nc_free(log_x);
    nc_free(sqrt_x);
    nc_free(abs_x);
    nc_free(pow_x);
    
    printf("\n[PASS] Example 4 completed successfully!\n");
    return 0;
}