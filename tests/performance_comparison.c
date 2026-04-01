#define NC_IMPLEMENTATION
#include "NumC.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ROWS1 50
#define COLS1 2000
#define ROWS2 2000
#define COLS2 50

// 传统方法：二维数组结构
typedef struct {
    double *data;
    int rows;
    int cols;
} TraditionalArray;

TraditionalArray* trad_create(int rows, int cols) {
    TraditionalArray *arr = (TraditionalArray*)malloc(sizeof(TraditionalArray));
    arr->data = (double*)malloc(rows * cols * sizeof(double));
    arr->rows = rows;
    arr->cols = cols;
    return arr;
}

void trad_free(TraditionalArray *arr) {
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

void trad_set(TraditionalArray *arr, int row, int col, double value) {
    arr->data[row * arr->cols + col] = value;
}

double trad_get(TraditionalArray *arr, int row, int col) {
    return arr->data[row * arr->cols + col];
}

// 传统方法：按行遍历（缓存友好）
double trad_sum_row_major(TraditionalArray *arr) {
    double sum = 0.0;
    for (int i = 0; i < arr->rows; i++) {
        for (int j = 0; j < arr->cols; j++) {
            sum += trad_get(arr, i, j);
        }
    }
    return sum;
}

// 传统方法：按列遍历（缓存不友好）
double trad_sum_col_major(TraditionalArray *arr) {
    double sum = 0.0;
    for (int j = 0; j < arr->cols; j++) {
        for (int i = 0; i < arr->rows; i++) {
            sum += trad_get(arr, i, j);
        }
    }
    return sum;
}

// 传统方法：逐元素乘法
void trad_multiply(TraditionalArray *a, TraditionalArray *b, TraditionalArray *result) {
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            double val = trad_get(a, i, j) * trad_get(b, i, j);
            trad_set(result, i, j, val);
        }
    }
}

// NumC方法：按行遍历
double numc_sum_row_major(NCArray *arr) {
    double sum = 0.0;
    int64_t rows = arr->shape[0];
    int64_t cols = arr->shape[1];
    double *data = (double*)arr->data;
    
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            sum += data[i * cols + j];
        }
    }
    return sum;
}

// NumC方法：按列遍历
double numc_sum_col_major(NCArray *arr) {
    double sum = 0.0;
    int64_t rows = arr->shape[0];
    int64_t cols = arr->shape[1];
    double *data = (double*)arr->data;
    
    for (int64_t j = 0; j < cols; j++) {
        for (int64_t i = 0; i < rows; i++) {
            sum += data[i * cols + j];
        }
    }
    return sum;
}

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int main() {
    printf("========================================\n");
    printf("二维数组遍历性能对比测试\n");
    printf("========================================\n\n");
    
    printf("测试数组：\n");
    printf("  数组A: %d行 × %d列 (共%d个元素)\n", ROWS1, COLS1, ROWS1*COLS1);
    printf("  数组B: %d行 × %d列 (共%d个元素)\n", ROWS2, COLS2, ROWS2*COLS2);
    printf("  元素总数相同，但形状不同\n\n");
    
    // ==================== 传统方法测试 ====================
    printf("【一、传统C语言方法】\n\n");
    
    // 创建数组
    TraditionalArray *trad_arr1 = trad_create(ROWS1, COLS1);
    TraditionalArray *trad_arr2 = trad_create(ROWS2, COLS2);
    TraditionalArray *trad_result1 = trad_create(ROWS1, COLS1);
    TraditionalArray *trad_result2 = trad_create(ROWS2, COLS2);
    
    // 初始化数据
    for (int i = 0; i < ROWS1; i++) {
        for (int j = 0; j < COLS1; j++) {
            trad_set(trad_arr1, i, j, (i + j) * 0.1);
        }
    }
    for (int i = 0; i < ROWS2; i++) {
        for (int j = 0; j < COLS2; j++) {
            trad_set(trad_arr2, i, j, (i + j) * 0.1);
        }
    }
    
    double start, end, duration;
    double sum;
    int iterations = 100;
    
    // 测试 50x2000 数组
    printf("1. 数组A (%d×%d) 测试结果：\n", ROWS1, COLS1);
    
    start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        sum = trad_sum_row_major(trad_arr1);
    }
    end = get_time_ms();
    duration = (end - start) / iterations;
    printf("   按行遍历求和: %.3f 毫秒 (sum=%.2f)\n", duration, sum);
    
    start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        sum = trad_sum_col_major(trad_arr1);
    }
    end = get_time_ms();
    duration = (end - start) / iterations;
    printf("   按列遍历求和: %.3f 毫秒 (sum=%.2f)\n", duration, sum);
    
    start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        trad_multiply(trad_arr1, trad_arr1, trad_result1);
    }
    end = get_time_ms();
    duration = (end - start) / iterations;
    printf("   逐元素乘法:   %.3f 毫秒\n\n", duration);
    
    // 测试 2000x50 数组
    printf("2. 数组B (%d×%d) 测试结果：\n", ROWS2, COLS2);
    
    start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        sum = trad_sum_row_major(trad_arr2);
    }
    end = get_time_ms();
    duration = (end - start) / iterations;
    printf("   按行遍历求和: %.3f 毫秒 (sum=%.2f)\n", duration, sum);
    
    start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        sum = trad_sum_col_major(trad_arr2);
    }
    end = get_time_ms();
    duration = (end - start) / iterations;
    printf("   按列遍历求和: %.3f 毫秒 (sum=%.2f)\n", duration, sum);
    
    start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        trad_multiply(trad_arr2, trad_arr2, trad_result2);
    }
    end = get_time_ms();
    duration = (end - start) / iterations;
    printf("   逐元素乘法:   %.3f 毫秒\n\n", duration);
    
    // 清理
    trad_free(trad_arr1);
    trad_free(trad_arr2);
    trad_free(trad_result1);
    trad_free(trad_result2);
    
    // ==================== NumC方法测试 ====================
    printf("【二、NumC库方法】\n\n");
    
    // 创建数组
    int64_t shape1[] = {ROWS1, COLS1};
    int64_t shape2[] = {ROWS2, COLS2};
    
    NCArray *numc_arr1 = nc_zeros(2, shape1, NC_FLOAT64);
    NCArray *numc_arr2 = nc_zeros(2, shape2, NC_FLOAT64);
    
    // 初始化数据
    double *data1 = (double*)numc_arr1->data;
    double *data2 = (double*)numc_arr2->data;
    
    for (int i = 0; i < ROWS1; i++) {
        for (int j = 0; j < COLS1; j++) {
            data1[i * COLS1 + j] = (i + j) * 0.1;
        }
    }
    for (int i = 0; i < ROWS2; i++) {
        for (int j = 0; j < COLS2; j++) {
            data2[i * COLS2 + j] = (i + j) * 0.1;
        }
    }
    
    // 测试 50x2000 数组
    printf("1. 数组A (%d×%d) 测试结果：\n", ROWS1, COLS1);
    
    start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        sum = numc_sum_row_major(numc_arr1);
    }
    end = get_time_ms();
    duration = (end - start) / iterations;
    printf("   按行遍历求和: %.3f 毫秒 (sum=%.2f)\n", duration, sum);
    
    start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        sum = numc_sum_col_major(numc_arr1);
    }
    end = get_time_ms();
    duration = (end - start) / iterations;
    printf("   按列遍历求和: %.3f 毫秒 (sum=%.2f)\n", duration, sum);
    
    start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        NCArray *result = nc_multiply(numc_arr1, numc_arr1);
        nc_release(result);
    }
    end = get_time_ms();
    duration = (end - start) / iterations;
    printf("   逐元素乘法:   %.3f 毫秒 (使用nc_multiply)\n\n", duration);
    
    // 测试 2000x50 数组
    printf("2. 数组B (%d×%d) 测试结果：\n", ROWS2, COLS2);
    
    start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        sum = numc_sum_row_major(numc_arr2);
    }
    end = get_time_ms();
    duration = (end - start) / iterations;
    printf("   按行遍历求和: %.3f 毫秒 (sum=%.2f)\n", duration, sum);
    
    start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        sum = numc_sum_col_major(numc_arr2);
    }
    end = get_time_ms();
    duration = (end - start) / iterations;
    printf("   按列遍历求和: %.3f 毫秒 (sum=%.2f)\n", duration, sum);
    
    start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        NCArray *result = nc_multiply(numc_arr2, numc_arr2);
        nc_release(result);
    }
    end = get_time_ms();
    duration = (end - start) / iterations;
    printf("   逐元素乘法:   %.3f 毫秒 (使用nc_multiply)\n\n", duration);
    
    // 清理
    nc_release(numc_arr1);
    nc_release(numc_arr2);
    
    printf("========================================\n");
    printf("测试完成！\n");
    printf("========================================\n");
    
    return 0;
}