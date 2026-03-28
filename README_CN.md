# NumC - NumPy风格的C语言数组计算库

一个纯C语言实现的NumPy风格数组计算库，优雅设计，功能全面。

## 特性

- **N维数组**：支持任意形状的多维数组
- **数据类型**：bool, int8-64, uint8-64, float32/64, complex64/128
- **Python风格数组字面量**：`NC_INT(1, 2, 3)` 语法，自动类型检测
- **广播机制**：自动处理不同形状数组间的运算
- **数学函数**：三角函数、指数、对数、幂函数等
- **线性代数**：矩阵乘法、点积、叉积等
- **归约操作**：sum, mean, variance, std, min, max等
- **随机数生成**：均匀分布、正态分布、整数采样
- **数组操作**：reshape, transpose, concatenate, split等
- **I/O**：二进制文件保存和加载
- **内存管理**：引用计数的retain/release

## 单文件版本（推荐）

只需一个头文件，直接 `#include` 使用：

```bash
gcc -o program program.c -lm
```

### 使用方法

```c
#define NC_IMPLEMENTATION
#include "NumC.h"
#include <stdio.h>

int main() {
    // Python风格的数组创建
    NCArray *a = NC_INT(1, 2, 3, 4, 5);           // [1, 2, 3, 4, 5]
    NCArray *b = NC_FLOAT(1.0, 2.0, 3.0);        // [1.0, 2.0, 3.0]
    NCArray *c = NC_INT2D(2, 3, 1, 2, 3, 4, 5, 6); // [[1,2,3],[4,5,6]]
    
    // 运算
    NCArray *sum = nc_add(a, a);
    NCArray *prod = nc_matmul(c, c);
    NCArray *mean = nc_mean(a, NULL, 0);
    
    // 打印
    nc_print(sum);
    nc_print(prod);
    printf("mean = %.2f\n", ((double*)mean->data)[0]);
    
    // 内存管理
    nc_release(a);
    nc_release(b);
    nc_release(c);
    nc_release(sum);
    nc_release(prod);
    nc_release(mean);
    
    return 0;
}
```

## 数组字面量

```c
// 1D 数组 - 自动选择最小能容纳的类型
NC_INT(1, 2, 3)              // -> int8
NC_INT(100, 200, 300)        // -> int16  
NC_INT(100000, 200000)       // -> int64

// Float数组 - 自动选择精度
NC_FLOAT(1.0, 2.0, 3.0)     // -> float32
NC_FLOAT(1.123456789, ...)    // -> float64 (高精度)

// 2D 数组
NC_INT2D(2, 3, 1, 2, 3, 4, 5, 6)      // [[1,2,3],[4,5,6]]
NC_FLOAT2D(2, 2, 1.0, 2.0, 3.0, 4.0)  // [[1,2],[3,4]]

// 显式类型
NC_INT8(...)    NC_INT16(...)    NC_INT32(...)    NC_INT64(...)
NC_UINT8(...)   NC_UINT16(...)   NC_UINT32(...)   NC_UINT64(...)
NC_FLOAT32(...) NC_FLOAT64(...)
```

## 主要API

| 函数 | 说明 |
|------|------|
| `nc_add(a,b)` | 加法 |
| `nc_subtract(a,b)` | 减法 |
| `nc_multiply(a,b)` | 乘法 |
| `nc_divide(a,b)` | 除法 |
| `nc_matmul(a,b)` | 矩阵乘法 |
| `nc_sum(a,axis,n)` | 求和 |
| `nc_mean(a,axis,n)` | 均值 |
| `nc_min(a,axis,n)` | 最小值 |
| `nc_max(a,axis,n)` | 最大值 |
| `nc_print(a)` | 打印 |
| `nc_release(a)` | 释放内存 |

## 编译选项

```bash
# 单文件编译（推荐）
gcc -o program program.c -lm

# 或定义宏后包含
#define NC_IMPLEMENTATION
#include "NumC.h"
```

## 许可

GNU General Public License v3 (GPLv3)