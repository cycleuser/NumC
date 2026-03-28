# NumC - NumPy-like Library for C

A pure C implementation of a NumPy-style array computation library with elegant design and comprehensive features.

## Features

- **N-dimensional arrays**: Support for multi-dimensional arrays of any shape
- **Data types**: bool, int8-64, uint8-64, float32/64, complex64/128
- **Python-style array literals**: `NC_INT(1, 2, 3)` syntax with automatic type detection
- **Broadcasting**: Automatic handling of operations between arrays of different shapes
- **Mathematical functions**: Trigonometric, exponential, logarithmic, power functions, etc.
- **Linear algebra**: Matrix multiplication, dot product, cross product, etc.
- **Reduction operations**: sum, mean, variance, std, min, max, etc.
- **Random number generation**: Uniform distribution, normal distribution, integer sampling
- **Array manipulation**: reshape, transpose, concatenate, split, etc.
- **I/O**: Binary file save and load
- **Memory management**: Reference counting with retain/release

## Single Header Version (Recommended)

Just one header file, directly `#include` and use:

```bash
gcc -o program program.c -lm
```

### Usage

```c
#define NC_IMPLEMENTATION
#include "NumC.h"
#include <stdio.h>

int main() {
    // Python-style array creation
    NCArray *a = NC_INT(1, 2, 3, 4, 5);             // [1, 2, 3, 4, 5]
    NCArray *b = NC_FLOAT(1.0, 2.0, 3.0);           // [1.0, 2.0, 3.0]
    NCArray *c = NC_INT2D(2, 3, 1, 2, 3, 4, 5, 6); // [[1,2,3],[4,5,6]]
    
    // Operations
    NCArray *sum = nc_add(a, a);
    NCArray *prod = nc_matmul(c, c);
    NCArray *mean = nc_mean(a, NULL, 0);
    
    // Print
    nc_print(sum);
    nc_print(prod);
    printf("mean = %.2f\n", ((double*)mean->data)[0]);
    
    // Memory management
    nc_release(a);
    nc_release(b);
    nc_release(c);
    nc_release(sum);
    nc_release(prod);
    nc_release(mean);
    
    return 0;
}
```

## Array Literals

```c
// 1D arrays - automatically selects smallest fitting type
NC_INT(1, 2, 3)              // -> int8
NC_INT(100, 200, 300)        // -> int16  
NC_INT(100000, 200000)       // -> int64

// Float arrays - automatically selects precision
NC_FLOAT(1.0, 2.0, 3.0)     // -> float32
NC_FLOAT(1.123456789, ...)    // -> float64 (high precision)

// 2D arrays
NC_INT2D(2, 3, 1, 2, 3, 4, 5, 6)      // [[1,2,3],[4,5,6]]
NC_FLOAT2D(2, 2, 1.0, 2.0, 3.0, 4.0)  // [[1,2],[3,4]]

// Explicit types
NC_INT8(...)    NC_INT16(...)    NC_INT32(...)    NC_INT64(...)
NC_UINT8(...)   NC_UINT16(...)   NC_UINT32(...)   NC_UINT64(...)
NC_FLOAT32(...) NC_FLOAT64(...)
```

## Main API

| Function | Description |
|----------|-------------|
| `nc_add(a,b)` | Addition |
| `nc_subtract(a,b)` | Subtraction |
| `nc_multiply(a,b)` | Multiplication |
| `nc_divide(a,b)` | Division |
| `nc_matmul(a,b)` | Matrix multiplication |
| `nc_sum(a,axis,n)` | Sum |
| `nc_mean(a,axis,n)` | Mean |
| `nc_min(a,axis,n)` | Minimum |
| `nc_max(a,axis,n)` | Maximum |
| `nc_print(a)` | Print |
| `nc_release(a)` | Free memory |

## Compilation

```bash
# Single file compilation (recommended)
gcc -o program program.c -lm

# Or define macro before include
#define NC_IMPLEMENTATION
#include "NumC.h"
```

## License

GNU General Public License v3 (GPLv3)