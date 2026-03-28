# NumC 教程

## 目录

1. [入门指南](#入门指南)
2. [安装方法](#安装方法)
3. [快速开始](#快速开始)
4. [一维数组](#一维数组)
5. [二维数组](#二维数组)
6. [定点数](#定点数)
7. [矩阵乘法](#矩阵乘法)
8. [常用操作](#常用操作)

---

## 入门指南

### 什么是 NumC？

NumC 是一个 NumPy 风格的 C 语言库，提供以下特性：

- **单头文件**：只需一个 `NumC.h` 文件，无需复杂的构建系统
- **Python 风格语法**：`NC_INT(1, 2, 3)` 轻松创建数组
- **自动类型检测**：自动选择最小适用的数据类型
- **定点数支持**：整个范围内精度均匀
- **无依赖**：纯 C 语言，仅需要标准库

### 下载方式

**方式一：克隆仓库**

```bash
git clone https://github.com/cycleuser/NumC.git
cd NumC
```

**方式二：直接下载单个文件**

```bash
# 使用 curl
curl -O https://raw.githubusercontent.com/cycleuser/NumC/main/NumC.h

# 使用 wget
wget https://raw.githubusercontent.com/cycleuser/NumC/main/NumC.h
```

**方式三：从 GitHub 网页下载**

访问：https://github.com/cycleuser/NumC

点击 "NumC.h" → 点击 "Raw" 按钮 → 另存为 `NumC.h`

---

## 安装方法

### 系统要求

- C 编译器（gcc、clang 或任何 C99 兼容编译器）
- 标准 C 库
- 数学库（所有系统通常都已预装）

### 目录结构

下载后，你应该有：

```
你的项目/
├── NumC.h          # 库文件（你只需要这一个文件）
└── main.c          # 你的代码
```

### 引入项目

```c
#define NC_IMPLEMENTATION
#include "NumC.h"
```

**重要**：你必须且只能在**一个**源文件中定义 `NC_IMPLEMENTATION`，然后包含 `NumC.h`。这会引入实现代码。在其他源文件中，只需使用 `#include "NumC.h"` 而不需要定义。

---

## 快速开始

### 第一个程序

创建 `main.c`：

```c
#define NC_IMPLEMENTATION
#include "NumC.h"
#include <stdio.h>

int main() {
    // 创建数组
    NCArray *a = NC_INT(1, 2, 3, 4, 5);
    
    // 打印数组
    nc_print(a);
    
    // 释放内存
    nc_free(a);
    
    return 0;
}
```

### 编译和运行

**使用 GCC（Linux/macOS/Windows MinGW）：**

```bash
gcc -o main main.c -lm
./main
```

**使用 Clang（macOS 默认）：**

```bash
clang -o main main.c -lm
./main
```

**使用 MSVC（Windows Visual Studio）：**

```cmd
cl main.c
main.exe
```

### `-lm` 是什么？

`-lm` 是链接数学库（`libm`）的选项。NumC 使用了 `sin()`、`cos()`、`sqrt()` 等数学函数，所以需要这个库。

- **Linux**：必须加 `-lm`
- **macOS**：不是必须的（数学库自动链接），但为了可移植性建议加上
- **Windows MSVC**：不需要

### 输出结果

```
array([1, 2, 3, 4, 5], shape=1, dtype=uint8)
```

---

## 一维数组

### 创建一维数组

```c
// 整数数组 - 自动类型检测
NCArray *a = NC_INT(1, 2, 3);              // 小值 → uint8
NCArray *b = NC_INT(100, 200, 300);        // 中等值 → uint16
NCArray *c = NC_INT(100000, 200000);       // 大值 → int64

// 浮点数组 - 自动类型检测
NCArray *d = NC_FLOAT(1.0, 2.0, 3.0);      // 简单小数 → float32
NCArray *e = NC_FLOAT(1.123456789);        // 高精度 → float64

// 显式指定类型
NCArray *f = NC_INT32(1, 2, 3);            // 强制 int32
NCArray *g = NC_FLOAT64(1.0, 2.0);         // 强制 float64
```

### 类型自动检测规则

**整数：**

| 数值范围 | 选择的类型 |
|---------|-----------|
| 0 到 255 | uint8 |
| -128 到 127 | int8 |
| 256 到 65535 | uint16 |
| -32768 到 32767 | int16 |
| 65536 到 4294967295 | uint32 |
| -2147483648 到 2147483647 | int32 |
| 更大的值 | uint64 或 int64 |

**浮点：**

| 精度 | 选择的类型 |
|-----|-----------|
| 简单小数（1.0, 2.5） | float32 |
| 高精度（1.123456789） | float64 |

### 数组属性

```c
NCArray *arr = NC_INT(10, 20, 30, 40, 50);

// 数据类型名称
printf("dtype: %s\n", nc_dtype_name(arr->dtype));  // "uint8"

// 维度数量
printf("ndim: %d\n", arr->ndim);                    // 1

// 形状（各维度大小的数组）
printf("shape: (%lld,)\n", arr->shape[0]);          // (5,)

// 元素总数
printf("size: %zu\n", nc_size(arr));                // 5

// 每个元素的字节数
printf("itemsize: %zu\n", arr->itemsize);           // 1 (uint8)
```

### 访问元素

```c
NCArray *arr = NC_INT(10, 20, 30, 40, 50);

// 读取元素（返回 double）
double val = nc_get_value_as_double(arr, 0);  // 10.0
double val2 = nc_get_value_as_double(arr, 2); // 30.0

// 遍历所有元素
for (size_t i = 0; i < nc_size(arr); i++) {
    printf("arr[%zu] = %.0f\n", i, nc_get_value_as_double(arr, i));
}
```

### 数组创建函数

```c
int64_t shape[1] = {5};  // 一维数组，5个元素

// 全零
NCArray *zeros = nc_zeros(1, shape, NC_FLOAT64);
// [0, 0, 0, 0, 0]

// 全一
NCArray *ones = nc_ones(1, shape, NC_INT32);
// [1, 1, 1, 1, 1]

// 填充指定值
int64_t val = 42;
NCArray *full = nc_full(1, shape, &val, NC_INT64);
// [42, 42, 42, 42, 42]

// 范围数组（类似 Python 的 range 或 numpy.arange）
NCArray *arange = nc_arange(0.0, 10.0, 2.0, NC_FLOAT64);
// [0, 2, 4, 6, 8]  （起始=0，结束=10，步长=2）

// 等分数组（类似 numpy.linspace）
NCArray *linspace = nc_linspace(0.0, 1.0, 5, true, NC_FLOAT64);
// [0, 0.25, 0.5, 0.75, 1]  （从0到1均匀分布5个数）
```

---

## 二维数组

### 创建二维数组

```c
// NC_INT2D(行数, 列数, 数据...)
// 数据按行主序排列（从左到右，从上到下）

NCArray *mat = NC_INT2D(2, 3, 1, 2, 3, 4, 5, 6);
// 创建：
// [[1, 2, 3],
//  [4, 5, 6]]

NCArray *fmat = NC_FLOAT2D(2, 2, 1.0, 2.0, 3.0, 4.0);
// 创建：
// [[1.0, 2.0],
//  [3.0, 4.0]]
```

### 矩阵属性

```c
NCArray *mat = NC_INT2D(2, 3, 1, 2, 3, 4, 5, 6);

printf("行数: %lld\n", mat->shape[0]);        // 2
printf("列数: %lld\n", mat->shape[1]);       // 3
printf("总元素数: %zu\n", nc_size(mat));     // 6
printf("维度数: %d\n", mat->ndim);           // 2
```

### 访问二维元素

```c
NCArray *mat = NC_INT2D(2, 3, 1, 2, 3, 4, 5, 6);

// 元素按行主序存储
// 索引 0 = 第0行第0列 = 1
// 索引 1 = 第0行第1列 = 2
// 索引 2 = 第0行第2列 = 3
// 索引 3 = 第1行第0列 = 4
// 以此类推

double val = nc_get_value_as_double(mat, 0);  // 1.0
double val = nc_get_value_as_double(mat, 3);  // 4.0

// 计算索引：行号 * 列数 + 列号
int row = 1, col = 2;
double val = nc_get_value_as_double(mat, row * 3 + col);  // 6.0
```

### 数组操作

```c
NCArray *mat = NC_INT2D(2, 3, 1, 2, 3, 4, 5, 6);

// 转置（交换行和列）
NCArray *trans = nctranspose(mat, NULL);
// [[1, 4],
//  [2, 5],
//  [3, 6]]

// 展平（转为一维）
NCArray *flat = ncflatten(mat);
// [1, 2, 3, 4, 5, 6]

// 拼接（连接数组）
NCArray *a = NC_INT(1, 2, 3);
NCArray *b = NC_INT(4, 5, 6);
NCArray *arrays[2] = {a, b};
NCArray *concat = nc_concatenate(arrays, 2, 0);
// [1, 2, 3, 4, 5, 6]
```

---

## 定点数

### 浮点数的问题

**浮点数的精度不均匀：**

接近 0 时精度很高，接近大数时精度很差。

```
接近 0 的精度：    0.0000000000000001（16位小数）
接近 1000 的精度： 0.0001（4位小数）
```

这意味着：
- 小数可以精确表示
- 大数会丢失精度
- 不适合需要均匀精度的应用

### 解决方案：定点数

**定点数的精度在任何地方都均匀：**

```
Q8.8 格式的精度：1/256 ≈ 0.0039（到处都一样！）
在 0：      精度 = 0.0039
在 100：    精度 = 0.0039
在 127.99： 精度 = 0.0039
```

### Q 格式表示法

```
Q<整数位数>.<小数位数>

例如：Q8.8
├── 8 位整数部分（含符号位）
├── 8 位小数部分
└── 总计：16 位

范围：-128.0 到 127.99609375
步长：1/256 = 0.00390625
```

### 常用 Q 格式

| 格式 | 总位数 | 有符号范围 | 步长 | C 类型 |
|-----|-------|----------|------|-------|
| Q4.4 | 8 | -8 到 7.94 | 0.0625 | int8 |
| Q8.8 | 16 | -128 到 128 | 0.0039 | int16 |
| Q16.16 | 32 | -32768 到 32768 | 0.000015 | int32 |
| Q24.8 | 32 | -8M 到 8M | 0.0039 | int32 |

### 创建定点数数组

```c
// 从 double 值创建
double values[] = {0.0, 0.25, 0.5, 0.75, 1.0};
NCArray *fx = nc_fixed_from_values(
    8,      // 整数位数
    8,      // 小数位数
    true,   // 是否有符号
    values, // 输入数组
    5       // 元素个数
);

// 打印（指定小数位数）
nc_fixed_print(fx, 8);
// fixed-point([0, 0.25, 0.5, 0.75, 1], shape=1, frac_bits=8, dtype=int16)
```

### 定点数运算

```c
double a_vals[] = {1.0, 2.0, 3.0};
double b_vals[] = {0.5, 0.5, 0.5};

NCArray *fa = nc_fixed_from_values(8, 8, true, a_vals, 3);
NCArray *fb = nc_fixed_from_values(8, 8, true, b_vals, 3);

// 加法
NCArray *sum = nc_fixed_add(fa, fb, 8);
// [1.5, 2.5, 3.5]

// 乘法
// Q8.8 × Q8.8 → Q8.8
NCArray *prod1 = nc_fixed_multiply(fa, fb, 8, 8, 8, NC_INT16);
// [0.5, 1.0, 1.5]

// Q8.8 × Q8.8 → Q16.16（保留更多精度）
NCArray *prod2 = nc_fixed_multiply(fa, fb, 8, 8, 16, NC_INT32);
// [0.5, 1.0, 1.5] 精度更高
```

### 定点数随机数

```c
nc_random_seed(42);  // 设置种子以保证可复现

int64_t shape[1] = {10};

// 全范围随机
NCArray *rand_full = nc_fixed_random_rand(8, 8, true, 1, shape);
// 数值：-128 到 128（Q8.8 范围）

// 指定范围随机
NCArray *rand_range = nc_fixed_random_uniform(
    8, 8, true,    // 格式
    -1.0, 1.0,     // 最小值, 最大值
    1, shape       // ndim, shape
);
// 数值：-1.0 到 1.0
```

### 何时使用定点数

**使用定点数当：**
- 需要整个范围内精度均匀
- 金融计算（避免浮点舍入误差）
- 嵌入式系统或 DSP
- 信号处理
- 游戏物理

**使用浮点数当：**
- 需要非常大的动态范围
- 科学计算涉及指数尺度
- 内存受限（float32 和 int32 占用相同空间）

---

## 矩阵乘法

### 基本矩阵乘法

```c
// A 是 2×3
NCArray *A = NC_INT2D(2, 3, 1, 2, 3, 4, 5, 6);
// [[1, 2, 3],
//  [4, 5, 6]]

// B 是 3×2
NCArray *B = NC_INT2D(3, 2, 7, 8, 9, 10, 11, 12);
// [[7, 8],
//  [9, 10],
//  [11, 12]]

// 矩阵乘法：(2×3) × (3×2) = (2×2)
NCArray *C = nc_matmul(A, B);

// 结果 C：
// [[1*7+2*9+3*11,  1*8+2*10+3*12],
//  [4*7+5*9+6*11,  4*8+5*10+6*12]]
// = [[58, 64],
//    [139, 154]]
```

### 矩阵乘法规则

```
A @ B 要能运算：
- A 的列数必须等于 B 的行数
- 如果 A 是 (m×n)，B 是 (n×p)，结果是 (m×p)

例如：
(2×3) × (3×4) = (2×4)   ✓ 有效
(3×2) × (2×3) = (3×3)   ✓ 有效
(2×3) × (4×5)            ✗ 无效（3 ≠ 4）
```

### 点积

```c
NCArray *v1 = NC_FLOAT(1.0, 2.0, 3.0);
NCArray *v2 = NC_FLOAT(4.0, 5.0, 6.0);

// 点积：1×4 + 2×5 + 3×6 = 32
NCArray *result = nc_dot(v1, v2);
// 结果：32.0
```

### 特殊矩阵

```c
// 单位矩阵（对角线为1，其余为0）
NCArray *I = nc_identity(4, NC_FLOAT64);
// [[1, 0, 0, 0],
//  [0, 1, 0, 0],
//  [0, 0, 1, 0],
//  [0, 0, 0, 1]]

// 对角矩阵（可指定偏移）
NCArray *eye = nc_eye(3, 4, 0, NC_FLOAT64);  // 3行4列，偏移0
// [[1, 0, 0, 0],
//  [0, 1, 0, 0],
//  [0, 0, 1, 0]]

// 从一维数组创建对角矩阵
NCArray *diag = nc_diag(NC_INT(1, 2, 3, 4), 0);
// [[1, 0, 0, 0],
//  [0, 2, 0, 0],
//  [0, 0, 3, 0],
//  [0, 0, 0, 4]]
```

---

## 常用操作

### 算术运算

```c
NCArray *a = NC_INT(1, 2, 3, 4, 5);
NCArray *b = NC_INT(10, 20, 30, 40, 50);

NCArray *sum = nc_add(a, b);         // [11, 22, 33, 44, 55]
NCArray *diff = nc_subtract(b, a);   // [9, 18, 27, 36, 45]
NCArray *prod = nc_multiply(a, b);   // [10, 40, 90, 160, 250]
NCArray *quot = nc_divide(b, a);     // [10, 10, 10, 10, 10]

NCArray *power = nc_power(a, NC_INT(2, 2, 2, 2, 2));  // [1, 4, 9, 16, 25]
```

### 数学函数

```c
#include <math.h>  // 使用 M_PI

NCArray *x = NC_FLOAT(0.0, M_PI/4, M_PI/2, M_PI);

nc_sin(x);    // 正弦
nc_cos(x);    // 余弦
nc_tan(x);    // 正切
nc_exp(x);    // e^x
nc_log(x);    // 自然对数
nc_sqrt(x);   // 平方根
nc_abs(NC_FLOAT(-1, 2, -3));  // [1, 2, 3]
```

### 归约操作

```c
NCArray *arr = NC_INT(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

NCArray *s = nc_sum(arr, NULL, 0);    // 55
NCArray *m = nc_mean(arr, NULL, 0);   // 5.5
NCArray *mn = nc_min(arr, NULL, 0);   // 1
NCArray *mx = nc_max(arr, NULL, 0);   // 10

double sum_val = nc_get_value_as_double(s, 0);  // 55.0
```

### 内存管理

```c
// 创建数组
NCArray *arr = NC_INT(1, 2, 3);

// 引用计数
NCArray *ref = nc_retain(arr);  // 增加引用计数

// 用完后释放
nc_free(arr);    // 减少引用计数
nc_free(ref);    // 计数到0时，内存被释放

// 另一个名称
nc_release(arr); // 同 nc_free()
```

---

## 总览表

| 功能 | 代码 | 说明 |
|-----|------|------|
| 一维整数 | `NC_INT(1, 2, 3)` | 自动类型检测 |
| 一维浮点 | `NC_FLOAT(1.0, 2.0)` | 自动选择 float32/float64 |
| 二维数组 | `NC_INT2D(2, 3, 1,2,3,4,5,6)` | 行主序 |
| 定点数 | `nc_fixed_from_values(8,8,true,arr,n)` | Q8.8 格式 |
| 矩阵乘法 | `nc_matmul(A, B)` | A @ B |
| 点积 | `nc_dot(v1, v2)` | 向量点积 |
| 转置 | `nctranspose(A, NULL)` | 交换行列 |
| 求和 | `nc_sum(arr, NULL, 0)` | 所有元素求和 |

---

## 许可

GNU General Public License v3 (GPLv3)