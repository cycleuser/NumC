#define NC_IMPLEMENTATION
#include "NumC.h"
#include <stdio.h>

int main() {
    printf("=== 使用NumC库实现一维数组 ===\n\n");
    
    NCArray *arr1 = NC_INT(1, 2, 3, 4, 5);
    NCArray *arr2 = NC_INT(2, 4, 6, 8, 10);
    
    printf("数组1: ");
    nc_print(arr1);
    
    printf("数组2: ");
    nc_print(arr2);
    
    NCArray *sum = nc_add(arr1, arr2);
    printf("相加: ");
    nc_print(sum);
    
    NCArray *prod = nc_multiply(arr1, arr2);
    printf("相乘: ");
    nc_print(prod);
    
    NCArray *sum_result = nc_sum(arr1, NULL, 0);
    NCArray *mean_result = nc_mean(arr1, NULL, 0);
    
    printf("数组1的和: %.0f\n", ((double*)sum_result->data)[0]);
    printf("数组1的平均值: %.2f\n", ((double*)mean_result->data)[0]);
    
    nc_release(arr1);
    nc_release(arr2);
    nc_release(sum);
    nc_release(prod);
    nc_release(sum_result);
    nc_release(mean_result);
    
    return 0;
}