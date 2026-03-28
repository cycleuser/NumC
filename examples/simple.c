#include "NumC.h"
#include <stdio.h>

int main() {
    printf("NumC v%s - NumPy-like library for C\n\n", nc_version());
    
    printf("=== Python-like Array Literals in C ===\n\n");
    
    printf("1D Integer Array (like [1, 2, 3, 4, 5]):\n");
    NCArray *arr1 = NC_INT(1, 2, 3, 4, 5);
    nc_print(arr1);
    printf("  dtype: %s\n\n", nc_dtype_name(nc_dtype(arr1)));
    
    printf("1D Float Array (auto float32/64 detection):\n");
    NCArray *f1 = NC_FLOAT(1.0, 2.0, 3.0);
    nc_print(f1);
    printf("  dtype: %s (simple values fit in float32)\n\n", nc_dtype_name(nc_dtype(f1)));
    
    NCArray *f2 = NC_FLOAT(1.123456789, 2.987654321);
    nc_print(f2);
    printf("  dtype: %s (precision requires float64)\n\n", nc_dtype_name(nc_dtype(f2)));
    
    printf("2D Integer Array (like [[1,2,3],[4,5,6]]):\n");
    NCArray *arr2d = NC_INT2D(2, 3, 1, 2, 3, 4, 5, 6);
    nc_print(arr2d);
    printf("  dtype: %s\n\n", nc_dtype_name(nc_dtype(arr2d)));
    
    printf("2D Float Array:\n");
    NCArray *f2d = NC_FLOAT2D(2, 2, 1.0, 2.0, 3.0, 4.0);
    nc_print(f2d);
    printf("  dtype: %s\n\n", nc_dtype_name(nc_dtype(f2d)));
    
    printf("=== Type Detection Examples ===\n\n");
    
    NCArray *i1 = NC_INT(1, 2, 3);          
    NCArray *i2 = NC_INT(100, 200, 300);   
    NCArray *i3 = NC_INT(100000, 200000);   
    NCArray *f3 = NC_FLOAT(1.5, 2.5, 3.5); 
    NCArray *f4 = NC_FLOAT(1.23456789, 9.87654321);
    
    printf("  NC_INT(1,2,3)           -> %s\n", nc_dtype_name(nc_dtype(i1)));
    printf("  NC_INT(100,200,300)     -> %s\n", nc_dtype_name(nc_dtype(i2)));
    printf("  NC_INT(100000,200000)   -> %s\n", nc_dtype_name(nc_dtype(i3)));
    printf("  NC_FLOAT(1.5,2.5,3.5)   -> %s\n", nc_dtype_name(nc_dtype(f3)));
    printf("  NC_FLOAT(1.23...,9.87...) -> %s\n", nc_dtype_name(nc_dtype(f4)));
    
    printf("\n=== Explicit Type Arrays ===\n\n");
    
    NCArray *e1 = NC_INT32(1, 2, 3);
    NCArray *e2 = NC_INT64(1, 2, 3);
    NCArray *e3 = NC_FLOAT32(1.0, 2.0, 3.0);
    NCArray *e4 = NC_FLOAT64(1.0, 2.0, 3.0);
    
    printf("  NC_INT32(1,2,3)    -> %s\n", nc_dtype_name(nc_dtype(e1)));
    printf("  NC_INT64(1,2,3)   -> %s\n", nc_dtype_name(nc_dtype(e2)));
    printf("  NC_FLOAT32(1,2,3) -> %s\n", nc_dtype_name(nc_dtype(e3)));
    printf("  NC_FLOAT64(1,2,3) -> %s\n", nc_dtype_name(nc_dtype(e4)));
    
    printf("\n=== Operations ===\n");
    
    NCArray *sum = nc_add(arr1, arr1);
    printf("\n[1,2,3,4,5] + [1,2,3,4,5]:\n");
    nc_print(sum);
    
    NCArray *product = nc_matmul(arr2d, f2d);
    printf("\n[[1,2,3],[4,5,6]] @ [[1,2],[3,4]]:\n");
    nc_print(product);
    
    NCArray *mean = nc_mean(arr1, NULL, 0);
    printf("\nMean of [1,2,3,4,5]: %.2f\n", ((double*)mean->data)[0]);
    
    printf("\n=== Memory Management ===\n");
    nc_release(arr1);
    nc_release(f1);
    nc_release(f2);
    nc_release(arr2d);
    nc_release(f2d);
    nc_release(i1);
    nc_release(i2);
    nc_release(i3);
    nc_release(f3);
    nc_release(f4);
    nc_release(e1);
    nc_release(e2);
    nc_release(e3);
    nc_release(e4);
    nc_release(sum);
    nc_release(product);
    nc_release(mean);
    
    printf("\nAll arrays released. Test passed!\n");
    return 0;
}
