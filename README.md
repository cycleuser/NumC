# NumC - NumPy-like Library for C

A pure C implementation of a NumPy-like array computing library with elegant design and comprehensive functionality.

## Features

- **N-dimensional arrays**: Full support for multi-dimensional arrays with arbitrary shapes
- **Data types**: Support for bool, int8-64, uint8-64, float32/64, complex64/128
- **Broadcasting**: Automatic broadcasting for operations between arrays of different shapes
- **Mathematical functions**: Full set of trigonometric, exponential, logarithmic, and power functions
- **Linear algebra**: Matrix multiplication, dot product, cross product, and more
- **Reduction operations**: sum, mean, variance, std, min, max, etc.
- **Random number generation**: Uniform, normal, integer, and random sampling
- **Array manipulation**: Reshape, transpose, concatenate, split, tile, repeat
- **FFT support**: Basic Fourier transform operations
- **I/O**: Save and load arrays to/from binary files
- **Memory management**: Reference counting with retain/release

## Building

```bash
mkdir build && cd build
cmake ..
make
```

### Build Options

- `NC_BUILD_TESTS`: Build test suite (ON/OFF)
- `NC_BUILD_EXAMPLES`: Build examples (ON/OFF)
- `NC_USE_OPENMP`: Enable OpenMP support (OFF/ON)

## Usage

```c
#include "NumC.h"
#include <stdio.h>

int main() {
    // Create arrays
    NCArray *a = nc_arange(0.0, 10.0, 1.0, NC_FLOAT64);
    NCArray *b = nc_ones(2, (int64_t[]){2, 3}, NC_FLOAT64);
    
    // Operations
    NCArray *sum = nc_add(a, b);
    NCArray *product = nc_matmul(a, b);
    NCArray *mean = nc_mean(a, NULL, 0);
    
    // Print results
    nc_print(sum);
    nc_print(product);
    
    // Memory management
    nc_release(a);
    nc_release(b);
    nc_release(sum);
    nc_release(product);
    nc_release(mean);
    
    return 0;
}
```

## API Reference

### Array Creation

| Function | Description |
|----------|-------------|
| `nc_empty(ndim, shape, dtype)` | Create uninitialized array |
| `nc_zeros(ndim, shape, dtype)` | Create zero-initialized array |
| `nc_ones(ndim, shape, dtype)` | Create array filled with ones |
| `nc_full(ndim, shape, value, dtype)` | Create array filled with value |
| `nc_arange(start, stop, step, dtype)` | Create array with range of values |
| `nc_linspace(start, stop, num, endpoint, dtype)` | Create linearly spaced values |
| `nc_identity(n, dtype)` | Create identity matrix |
| `nc_eye(n, m, k, dtype)` | Create matrix with diagonal |

### Array Operations

| Function | Description |
|----------|-------------|
| `nc_reshape(arr, ndim, shape)` | Change array shape |
| `nc_transpose(arr, axes)` | Transpose array |
| `nc_concatenate(arrays, n, axis)` | Concatenate arrays |
| `nc_split(arr, n, axis)` | Split array into sections |
| `nc_tile(arr, reps, n)` | Tile array |
| `nc_repeat(arr, repeats, axis)` | Repeat array elements |

### Element-wise Operations

| Function | Description |
|----------|-------------|
| `nc_add(a, b)` | Element-wise addition |
| `nc_subtract(a, b)` | Element-wise subtraction |
| `nc_multiply(a, b)` | Element-wise multiplication |
| `nc_divide(a, b)` | Element-wise division |
| `nc_power(a, b)` | Element-wise power |
| `nc_mod(a, b)` | Element-wise modulo |

### Mathematical Functions

| Function | Description |
|----------|-------------|
| `nc_abs(arr)` | Absolute value |
| `nc_sin(arr)` | Sine |
| `nc_cos(arr)` | Cosine |
| `nc_tan(arr)` | Tangent |
| `nc_exp(arr)` | Exponential |
| `nc_log(arr)` | Natural logarithm |
| `nc_sqrt(arr)` | Square root |
| `nc_pow(arr, n)` | Power |

### Reduction Operations

| Function | Description |
|----------|-------------|
| `nc_sum(arr, axis, naxis)` | Sum of elements |
| `nc_mean(arr, axis, naxis)` | Mean of elements |
| `nc_var(arr, axis, naxis)` | Variance |
| `nc_std(arr, axis, naxis)` | Standard deviation |
| `nc_min(arr, axis, naxis)` | Minimum value |
| `nc_max(arr, axis, naxis)` | Maximum value |

### Linear Algebra

| Function | Description |
|----------|-------------|
| `nc_dot(a, b)` | Dot product |
| `nc_matmul(a, b)` | Matrix multiplication |
| `nc_inner(a, b)` | Inner product |
| `nc_outer(a, b)` | Outer product |
| `nc_cross(a, b, axis)` | Cross product |
| `nc_trace(arr, offset, axis1, axis2)` | Trace |

### Memory Management

| Function | Description |
|----------|-------------|
| `nc_retain(arr)` | Increment reference count |
| `nc_release(arr)` | Decrement reference count |
| `nc_copy(arr)` | Create a copy |
| `nc_free(arr)` | Alias for release |

## Testing

```bash
cd build
ctest
./tests/test_numc
```

## License

GNU General Public License v3 (GPLv3)
