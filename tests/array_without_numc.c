#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int *data;
    size_t size;
} IntArray;

IntArray* array_create(size_t size) {
    IntArray *arr = (IntArray*)malloc(sizeof(IntArray));
    if (!arr) return NULL;
    
    arr->data = (int*)malloc(size * sizeof(int));
    if (!arr->data) {
        free(arr);
        return NULL;
    }
    arr->size = size;
    return arr;
}

void array_free(IntArray *arr) {
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

void array_set(IntArray *arr, size_t index, int value) {
    if (arr && index < arr->size) {
        arr->data[index] = value;
    }
}

int array_get(IntArray *arr, size_t index) {
    if (arr && index < arr->size) {
        return arr->data[index];
    }
    return 0;
}

void array_print(IntArray *arr) {
    if (!arr) return;
    printf("[");
    for (size_t i = 0; i < arr->size; i++) {
        printf("%d", arr->data[i]);
        if (i < arr->size - 1) printf(", ");
    }
    printf("]\n");
}

IntArray* array_add(IntArray *a, IntArray *b) {
    if (!a || !b || a->size != b->size) return NULL;
    
    IntArray *result = array_create(a->size);
    if (!result) return NULL;
    
    for (size_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    return result;
}

IntArray* array_multiply(IntArray *a, IntArray *b) {
    if (!a || !b || a->size != b->size) return NULL;
    
    IntArray *result = array_create(a->size);
    if (!result) return NULL;
    
    for (size_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
    return result;
}

int array_sum(IntArray *arr) {
    if (!arr) return 0;
    int sum = 0;
    for (size_t i = 0; i < arr->size; i++) {
        sum += arr->data[i];
    }
    return sum;
}

double array_mean(IntArray *arr) {
    if (!arr || arr->size == 0) return 0.0;
    return (double)array_sum(arr) / arr->size;
}

int main() {
    printf("=== 传统C语言实现一维数组 ===\n\n");
    
    IntArray *arr1 = array_create(5);
    IntArray *arr2 = array_create(5);
    
    for (size_t i = 0; i < 5; i++) {
        array_set(arr1, i, i + 1);
        array_set(arr2, i, (i + 1) * 2);
    }
    
    printf("数组1: ");
    array_print(arr1);
    
    printf("数组2: ");
    array_print(arr2);
    
    IntArray *sum = array_add(arr1, arr2);
    printf("相加: ");
    array_print(sum);
    
    IntArray *prod = array_multiply(arr1, arr2);
    printf("相乘: ");
    array_print(prod);
    
    printf("数组1的和: %d\n", array_sum(arr1));
    printf("数组1的平均值: %.2f\n", array_mean(arr1));
    
    array_free(arr1);
    array_free(arr2);
    array_free(sum);
    array_free(prod);
    
    return 0;
}