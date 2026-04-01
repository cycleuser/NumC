# 两种方式实现一维数组：传统C语言 vs NumC库

最近在用C语言做数组运算时，发现代码写得又麻烦又长。于是就想着封装一个库，让它能像Python的NumPy那样简单好用。

NumC参考了NumPy的设计，如果你会用NumPy，上手NumC几乎不需要学习成本。对于需要大量数组运算的C语言项目，用NumC真的能省下大量的时间和精力。


NumC库的GitHub地址：https://github.com/cycleuser/NumC

单头文件库，下载NumC.h就能用：
```bash
wget https://blog.cycleuser.org/NumC.h
```

编译时加上`-lm`参数链接数学库就行：
```bash
gcc your_program.c -o program -lm
```

今天就来对比一下，用传统C语言和使用NumC库实现一维数组，到底有多大差别。

先说结论：**同样的功能，传统C语言要写120行，用NumC只要30行。**

## 先看传统C语言怎么写

用C语言实现一个简单的一维数组，我们至少需要以下功能：
- 创建和销毁数组
- 设置和获取元素
- 数组加减乘除
- 统计功能（求和、平均值）

把这些功能都实现出来，代码大概是这样的：

```c
#include <stdio.h>
#include <stdlib.h>

// 定义数组结构体
typedef struct {
    int *data;
    size_t size;
} IntArray;

// 创建数组
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

// 释放数组
void array_free(IntArray *arr) {
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

// 设置元素
void array_set(IntArray *arr, size_t index, int value) {
    if (arr && index < arr->size) {
        arr->data[index] = value;
    }
}

// 获取元素
int array_get(IntArray *arr, size_t index) {
    if (arr && index < arr->size) {
        return arr->data[index];
    }
    return 0;
}

// 打印数组
void array_print(IntArray *arr) {
    if (!arr) return;
    printf("[");
    for (size_t i = 0; i < arr->size; i++) {
        printf("%d", arr->data[i]);
        if (i < arr->size - 1) printf(", ");
    }
    printf("]\n");
}

// 数组加法
IntArray* array_add(IntArray *a, IntArray *b) {
    if (!a || !b || a->size != b->size) return NULL;
    
    IntArray *result = array_create(a->size);
    if (!result) return NULL;
    
    for (size_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    return result;
}

// 数组乘法（逐元素相乘）
IntArray* array_multiply(IntArray *a, IntArray *b) {
    if (!a || !b || a->size != b->size) return NULL;
    
    IntArray *result = array_create(a->size);
    if (!result) return NULL;
    
    for (size_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
    return result;
}

// 求和
int array_sum(IntArray *arr) {
    if (!arr) return 0;
    int sum = 0;
    for (size_t i = 0; i < arr->size; i++) {
        sum += arr->data[i];
    }
    return sum;
}

// 求平均值
double array_mean(IntArray *arr) {
    if (!arr || arr->size == 0) return 0.0;
    return (double)array_sum(arr) / arr->size;
}

int main() {
    // 创建两个数组
    IntArray *arr1 = array_create(5);
    IntArray *arr2 = array_create(5);
    
    // 手动赋值
    for (size_t i = 0; i < 5; i++) {
        array_set(arr1, i, i + 1);
        array_set(arr2, i, (i + 1) * 2);
    }
    
    // 打印
    printf("数组1: ");
    array_print(arr1);
    printf("数组2: ");
    array_print(arr2);
    
    // 运算
    IntArray *sum = array_add(arr1, arr2);
    printf("相加: ");
    array_print(sum);
    
    IntArray *prod = array_multiply(arr1, arr2);
    printf("相乘: ");
    array_print(prod);
    
    // 统计
    printf("数组1的和: %d\n", array_sum(arr1));
    printf("数组1的平均值: %.2f\n", array_mean(arr1));
    
    // 清理内存
    array_free(arr1);
    array_free(arr2);
    array_free(sum);
    array_free(prod);
    
    return 0;
}
```

数一数，大概**120行代码**。而且这还只是最基本的功能，如果要做浮点数数组、多维数组、矩阵运算，代码量会成倍增加。每增加一种数据类型，就要重写一套函数。如果想要支持int、float、double三种类型，代码量至少要翻三倍。

更麻烦的是，这些代码都需要自己维护。每次发现bug或者要优化性能，都得去翻这一大堆代码。

## 再看看用NumC库怎么写

同样的功能，用NumC库来实现：

```c
#define NC_IMPLEMENTATION
#include "NumC.h"
#include <stdio.h>

int main() {
    // 创建两个数组，一行搞定
    NCArray *arr1 = NC_INT(1, 2, 3, 4, 5);
    NCArray *arr2 = NC_INT(2, 4, 6, 8, 10);
    
    // 打印
    printf("数组1: ");
    nc_print(arr1);
    printf("数组2: ");
    nc_print(arr2);
    
    // 直接调用库函数
    NCArray *sum = nc_add(arr1, arr2);
    printf("相加: ");
    nc_print(sum);
    
    NCArray *prod = nc_multiply(arr1, arr2);
    printf("相乘: ");
    nc_print(prod);
    
    // 统计功能也是一行搞定
    NCArray *sum_result = nc_sum(arr1, NULL, 0);
    NCArray *mean_result = nc_mean(arr1, NULL, 0);
    
    printf("数组1的和: %.0f\n", ((double*)sum_result->data)[0]);
    printf("数组1的平均值: %.2f\n", ((double*)mean_result->data)[0]);
    
    // 释放内存
    nc_release(arr1);
    nc_release(arr2);
    nc_release(sum);
    nc_release(prod);
    nc_release(sum_result);
    nc_release(mean_result);
    
    return 0;
}
```

只有**30行代码**！而且功能更强大，还支持更多数据类型。

## 详细对比一下代码复杂度

### 1. 数组创建

**传统方式（需要7行）：**
```c
IntArray *arr1 = array_create(5);
for (size_t i = 0; i < 5; i++) {
    array_set(arr1, i, i + 1);
}
```
先创建数组，然后循环赋值。如果要创建多个数组，这个过程要重复多次。

**NumC方式（只需1行）：**
```c
NCArray *arr1 = NC_INT(1, 2, 3, 4, 5);
```
直接用宏定义，把值写进去就行。直观、简洁、不易出错。

### 2. 数组运算

**传统方式：**
```c
// 需要先写一个加法函数
IntArray* array_add(IntArray *a, IntArray *b) {
    if (!a || !b || a->size != b->size) return NULL;
    IntArray *result = array_create(a->size);
    if (!result) return NULL;
    for (size_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    return result;
}
// 然后才能调用
IntArray *sum = array_add(arr1, arr2);
```

**NumC方式：**
```c
NCArray *sum = nc_add(arr1, arr2);
```
不用自己实现，直接调用。减法、乘法、除法也一样简单。

### 3. 统计功能

**传统方式：**
```c
// 求和函数
int array_sum(IntArray *arr) {
    if (!arr) return 0;
    int sum = 0;
    for (size_t i = 0; i < arr->size; i++) {
        sum += arr->data[i];
    }
    return sum;
}

// 平均值函数
double array_mean(IntArray *arr) {
    if (!arr || arr->size == 0) return 0.0;
    return (double)array_sum(arr) / arr->size;
}

// 调用
int s = array_sum(arr1);
double m = array_mean(arr1);
```

**NumC方式：**
```c
NCArray *s = nc_sum(arr1, NULL, 0);
NCArray *m = nc_mean(arr1, NULL, 0);
```
而且NumC还提供了标准差、方差、最大值、最小值等一大堆统计函数，都不用自己写。

### 4. 数据类型支持

**传统方式：**
如果想要支持float类型，就得重写所有函数：
```c
typedef struct {
    float *data;    // 改成float
    size_t size;
} FloatArray;

FloatArray* float_array_create(size_t size);
void float_array_set(FloatArray *arr, size_t index, float value);
FloatArray* float_array_add(FloatArray *a, FloatArray *b);
// ... 所有函数都要重写一遍
```

如果要支持int、float、double三种类型，代码量至少翻三倍，变成360行以上。

**NumC方式：**
```c
// 整数数组
NCArray *int_arr = NC_INT(1, 2, 3);

// 浮点数组
NCArray *float_arr = NC_FLOAT(1.5, 2.5, 3.5);

// 双精度数组
NCArray *double_arr = NC_DOUBLE(1.5, 2.5, 3.5);
```
一套API，支持多种数据类型，完全不用重写代码。


## 粗略对比

| 功能 | 传统C语言 | NumC库 |
|-----|---------|--------|
| 数组创建 | 7行代码 + 循环赋值 | 1行宏定义 |
| 加减乘除 | 每个运算需要写10-15行函数 | 1行函数调用 |
| 统计功能 | 每个功能需要写5-10行 | 1行函数调用 |
| 数据类型 | 每种类型需要重写所有函数 | 自动支持多种类型 |
| 多维数组 | 需要大量额外代码 | 直接支持 |
| 数学函数 | 需要逐个实现 | 内置数十个函数 |
| 总代码量 | 120+行（基础功能） | 30行 |
| 维护成本 | 自己维护所有代码 | 库已经维护好 |


## 那么代价是什么？

用传统C或者更强的专门的库肯定能有更好的性能，而这个NumC只能算是用来做教学示范的一个小玩具了。

