# NumC Tutorial

## Table of Contents

1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [One-Dimensional Arrays](#one-dimensional-arrays)
5. [Two-Dimensional Arrays](#two-dimensional-arrays)
6. [Fixed-Point Numbers](#fixed-point-numbers)
7. [Matrix Multiplication](#matrix-multiplication)
8. [Common Operations](#common-operations)

---

## Getting Started

### What is NumC?

NumC is a NumPy-like library for C that provides:

- **Single header file**: Just one `NumC.h` file - no complex build system
- **Python-style syntax**: `NC_INT(1, 2, 3)` creates arrays easily
- **Auto type detection**: Automatically selects the smallest suitable data type
- **Fixed-point support**: Uniform precision across entire range
- **No dependencies**: Pure C, only requires standard library

### Download

**Option 1: Clone the repository**

```bash
git clone https://github.com/cycleuser/NumC.git
cd NumC
```

**Option 2: Download single file**

```bash
# Using curl
curl -O https://raw.githubusercontent.com/cycleuser/NumC/main/NumC.h

# Or using wget
wget https://raw.githubusercontent.com/cycleuser/NumC/main/NumC.h
```

**Option 3: Download from GitHub webpage**

Visit: https://github.com/cycleuser/NumC

Click "NumC.h" → Click "Raw" button → Save page as `NumC.h`

---

## Installation

### Requirements

- C compiler (gcc, clang, or any C99-compatible compiler)
- Standard C library
- Math library (usually pre-installed on all systems)

### Directory Structure

After downloading, you should have:

```
your-project/
├── NumC.h          # The library (only file you need)
└── main.c          # Your code
```

### Include in Your Project

```c
#define NC_IMPLEMENTATION
#include "NumC.h"
```

**Important**: You MUST define `NC_IMPLEMENTATION` in exactly ONE source file before including `NumC.h`. This includes the implementation code. In other source files, just use `#include "NumC.h"` without the define.

---

## Quick Start

### Your First Program

Create `main.c`:

```c
#define NC_IMPLEMENTATION
#include "NumC.h"
#include <stdio.h>

int main() {
    // Create an array
    NCArray *a = NC_INT(1, 2, 3, 4, 5);
    
    // Print it
    nc_print(a);
    
    // Free memory
    nc_free(a);
    
    return 0;
}
```

### Compile and Run

**Using GCC (Linux/macOS/Windows MinGW):**

```bash
gcc -o main main.c -lm
./main
```

**Using Clang (macOS default):**

```bash
clang -o main main.c -lm
./main
```

**Using MSVC (Windows Visual Studio):**

```cmd
cl main.c
main.exe
```

### What is `-lm`?

The `-lm` flag links the math library (`libm`). It's needed because NumC uses math functions like `sin()`, `cos()`, `sqrt()`, etc.

- **Linux**: `-lm` is required
- **macOS**: Not strictly needed (math library is auto-linked), but recommended for portability
- **Windows MSVC**: Not needed

### Output

```
array([1, 2, 3, 4, 5], shape=1, dtype=uint8)
```

---

## One-Dimensional Arrays

### Creating 1D Arrays

```c
// Integer arrays with auto type detection
NCArray *a = NC_INT(1, 2, 3);              // Small values → uint8
NCArray *b = NC_INT(100, 200, 300);        // Medium values → uint16
NCArray *c = NC_INT(100000, 200000);       // Large values → int64

// Float arrays with auto type detection
NCArray *d = NC_FLOAT(1.0, 2.0, 3.0);      // Simple decimals → float32
NCArray *e = NC_FLOAT(1.123456789);        // High precision → float64

// Explicit type specification
NCArray *f = NC_INT32(1, 2, 3);            // Force int32
NCArray *g = NC_FLOAT64(1.0, 2.0);         // Force float64
```

### Type Auto-Detection Rules

**Integers:**

| Value Range | Selected Type |
|-------------|---------------|
| 0 to 255 | uint8 |
| -128 to 127 | int8 |
| 256 to 65535 | uint16 |
| -32768 to 32767 | int16 |
| 65536 to 4294967295 | uint32 |
| -2147483648 to 2147483647 | int32 |
| Larger values | uint64 or int64 |

**Floats:**

| Precision | Selected Type |
|-----------|---------------|
| Simple decimals (1.0, 2.5) | float32 |
| High precision (1.123456789) | float64 |

### Array Properties

```c
NCArray *arr = NC_INT(10, 20, 30, 40, 50);

// Data type name
printf("dtype: %s\n", nc_dtype_name(arr->dtype));  // "uint8"

// Number of dimensions
printf("ndim: %d\n", arr->ndim);                    // 1

// Shape (array of dimension sizes)
printf("shape: (%lld,)\n", arr->shape[0]);          // (5,)

// Total number of elements
printf("size: %zu\n", nc_size(arr));                // 5

// Size of each element in bytes
printf("itemsize: %zu\n", arr->itemsize);           // 1 (for uint8)
```

### Accessing Elements

```c
NCArray *arr = NC_INT(10, 20, 30, 40, 50);

// Read element as double
double val = nc_get_value_as_double(arr, 0);  // 10.0
double val2 = nc_get_value_as_double(arr, 2); // 30.0

// Iterate through all elements
for (size_t i = 0; i < nc_size(arr); i++) {
    printf("arr[%zu] = %.0f\n", i, nc_get_value_as_double(arr, i));
}
```

### Array Creation Functions

```c
int64_t shape[1] = {5};  // Shape for 1D array with 5 elements

// Zeros
NCArray *zeros = nc_zeros(1, shape, NC_FLOAT64);
// [0, 0, 0, 0, 0]

// Ones
NCArray *ones = nc_ones(1, shape, NC_INT32);
// [1, 1, 1, 1, 1]

// Fill with value
int64_t val = 42;
NCArray *full = nc_full(1, shape, &val, NC_INT64);
// [42, 42, 42, 42, 42]

// Range (like Python's range or numpy.arange)
NCArray *arange = nc_arange(0.0, 10.0, 2.0, NC_FLOAT64);
// [0, 2, 4, 6, 8]  (start=0, stop=10, step=2)

// Linearly spaced (like numpy.linspace)
NCArray *linspace = nc_linspace(0.0, 1.0, 5, true, NC_FLOAT64);
// [0, 0.25, 0.5, 0.75, 1]  (5 evenly spaced values from 0 to 1)
```

---

## Two-Dimensional Arrays

### Creating 2D Arrays

```c
// NC_INT2D(rows, cols, values...)
// Values are in row-major order (left to right, top to bottom)

NCArray *mat = NC_INT2D(2, 3, 1, 2, 3, 4, 5, 6);
// Creates:
// [[1, 2, 3],
//  [4, 5, 6]]

NCArray *fmat = NC_FLOAT2D(2, 2, 1.0, 2.0, 3.0, 4.0);
// Creates:
// [[1.0, 2.0],
//  [3.0, 4.0]]
```

### Matrix Properties

```c
NCArray *mat = NC_INT2D(2, 3, 1, 2, 3, 4, 5, 6);

printf("Rows: %lld\n", mat->shape[0]);        // 2
printf("Columns: %lld\n", mat->shape[1]);    // 3
printf("Total elements: %zu\n", nc_size(mat)); // 6
printf("Dimensions: %d\n", mat->ndim);        // 2
```

### Accessing 2D Elements

```c
NCArray *mat = NC_INT2D(2, 3, 1, 2, 3, 4, 5, 6);

// Elements are stored in row-major order
// Index 0 = row 0, col 0 = 1
// Index 1 = row 0, col 1 = 2
// Index 2 = row 0, col 2 = 3
// Index 3 = row 1, col 0 = 4
// etc.

double val = nc_get_value_as_double(mat, 0);  // 1.0
double val = nc_get_value_as_double(mat, 3);  // 4.0

// Calculate index: row * cols + col
int row = 1, col = 2;
double val = nc_get_value_as_double(mat, row * 3 + col);  // 6.0
```

### Array Manipulation

```c
NCArray *mat = NC_INT2D(2, 3, 1, 2, 3, 4, 5, 6);

// Transpose (swap rows and columns)
NCArray *trans = nctranspose(mat, NULL);
// [[1, 4],
//  [2, 5],
//  [3, 6]]

// Flatten (convert to 1D)
NCArray *flat = ncflatten(mat);
// [1, 2, 3, 4, 5, 6]

// Concatenate (join arrays)
NCArray *a = NC_INT(1, 2, 3);
NCArray *b = NC_INT(4, 5, 6);
NCArray *arrays[2] = {a, b};
NCArray *concat = nc_concatenate(arrays, 2, 0);
// [1, 2, 3, 4, 5, 6]
```

---

## Fixed-Point Numbers

### The Problem with Floating-Point

**Floating-point numbers have non-uniform precision:**

Near 0, floats have very high precision. Near large values, precision is poor.

```
Float precision near 0:     0.0000000000000001 (16 decimal places)
Float precision near 1000:  0.0001 (4 decimal places)
```

This means:
- Small numbers can be represented precisely
- Large numbers lose precision
- Not suitable for applications requiring uniform accuracy

### The Solution: Fixed-Point Numbers

**Fixed-point has uniform precision everywhere:**

```
Q8.8 format precision:  1/256 ≈ 0.0039 (same everywhere!)
At 0:       precision = 0.0039
At 100:     precision = 0.0039
At 127.99:  precision = 0.0039
```

### Q Format Notation

```
Q<integer_bits>.<fraction_bits>

Example: Q8.8
├── 8 bits for integer part (including sign)
├── 8 bits for fraction part
└── Total: 16 bits

Range: -128.0 to 127.99609375
Step:  1/256 = 0.00390625
```

### Common Q Formats

| Format | Total Bits | Range (Signed) | Step Size | C Type |
|--------|-----------|----------------|-----------|--------|
| Q4.4 | 8 | -8 to 7.94 | 0.0625 | int8 |
| Q8.8 | 16 | -128 to 128 | 0.0039 | int16 |
| Q16.16 | 32 | -32768 to 32768 | 0.000015 | int32 |
| Q24.8 | 32 | -8M to 8M | 0.0039 | int32 |

### Creating Fixed-Point Arrays

```c
// From double values
double values[] = {0.0, 0.25, 0.5, 0.75, 1.0};
NCArray *fx = nc_fixed_from_values(
    8,      // integer bits
    8,      // fraction bits
    true,   // signed
    values, // input array
    5       // count
);

// Print with fraction bits
nc_fixed_print(fx, 8);
// fixed-point([0, 0.25, 0.5, 0.75, 1], shape=1, frac_bits=8, dtype=int16)
```

### Fixed-Point Arithmetic

```c
double a_vals[] = {1.0, 2.0, 3.0};
double b_vals[] = {0.5, 0.5, 0.5};

NCArray *fa = nc_fixed_from_values(8, 8, true, a_vals, 3);
NCArray *fb = nc_fixed_from_values(8, 8, true, b_vals, 3);

// Addition
NCArray *sum = nc_fixed_add(fa, fb, 8);
// [1.5, 2.5, 3.5]

// Multiplication
// Q8.8 × Q8.8 → Q8.8
NCArray *prod1 = nc_fixed_multiply(fa, fb, 8, 8, 8, NC_INT16);
// [0.5, 1.0, 1.5]

// Q8.8 × Q8.8 → Q16.16 (preserves more precision)
NCArray *prod2 = nc_fixed_multiply(fa, fb, 8, 8, 16, NC_INT32);
// [0.5, 1.0, 1.5] with more precision
```

### Fixed-Point Random Numbers

```c
nc_random_seed(42);  // Set seed for reproducibility

int64_t shape[1] = {10};

// Random in full range
NCArray *rand_full = nc_fixed_random_rand(8, 8, true, 1, shape);
// Values: -128 to 128 (Q8.8 range)

// Random in specific range
NCArray *rand_range = nc_fixed_random_uniform(
    8, 8, true,    // format
    -1.0, 1.0,     // min, max
    1, shape       // ndim, shape
);
// Values: -1.0 to 1.0
```

### When to Use Fixed-Point

**Use fixed-point when:**
- You need uniform precision across the entire range
- Financial calculations (avoid floating-point rounding errors)
- Embedded systems or DSP
- Signal processing
- Game physics

**Use floating-point when:**
- You need a very large dynamic range
- Scientific computing with exponential scales
- Memory is limited (float32 uses same space as int32)

---

## Matrix Multiplication

### Basic Matrix Multiply

```c
// A is 2×3
NCArray *A = NC_INT2D(2, 3, 1, 2, 3, 4, 5, 6);
// [[1, 2, 3],
//  [4, 5, 6]]

// B is 3×2
NCArray *B = NC_INT2D(3, 2, 7, 8, 9, 10, 11, 12);
// [[7, 8],
//  [9, 10],
//  [11, 12]]

// Matrix multiplication: (2×3) × (3×2) = (2×2)
NCArray *C = nc_matmul(A, B);

// Result C:
// [[1*7+2*9+3*11,  1*8+2*10+3*12],
//  [4*7+5*9+6*11,  4*8+5*10+6*12]]
// = [[58, 64],
//    [139, 154]]
```

### Matrix Multiply Rules

```
For A @ B to work:
- A columns must equal B rows
- If A is (m×n) and B is (n×p), result is (m×p)

Example:
(2×3) × (3×4) = (2×4)   ✓ Valid
(3×2) × (2×3) = (3×3)   ✓ Valid
(2×3) × (4×5)            ✗ Invalid (3 ≠ 4)
```

### Dot Product

```c
NCArray *v1 = NC_FLOAT(1.0, 2.0, 3.0);
NCArray *v2 = NC_FLOAT(4.0, 5.0, 6.0);

// Dot product: 1×4 + 2×5 + 3×6 = 32
NCArray *result = nc_dot(v1, v2);
// Result: 32.0
```

### Special Matrices

```c
// Identity matrix (ones on diagonal, zeros elsewhere)
NCArray *I = nc_identity(4, NC_FLOAT64);
// [[1, 0, 0, 0],
//  [0, 1, 0, 0],
//  [0, 0, 1, 0],
//  [0, 0, 0, 1]]

// Eye matrix (diagonal ones, can specify offset)
NCArray *eye = nc_eye(3, 4, 0, NC_FLOAT64);  // 3 rows, 4 cols, offset 0
// [[1, 0, 0, 0],
//  [0, 1, 0, 0],
//  [0, 0, 1, 0]]

// Diagonal matrix from 1D array
NCArray *diag = nc_diag(NC_INT(1, 2, 3, 4), 0);
// [[1, 0, 0, 0],
//  [0, 2, 0, 0],
//  [0, 0, 3, 0],
//  [0, 0, 0, 4]]
```

---

## Common Operations

### Arithmetic

```c
NCArray *a = NC_INT(1, 2, 3, 4, 5);
NCArray *b = NC_INT(10, 20, 30, 40, 50);

NCArray *sum = nc_add(a, b);         // [11, 22, 33, 44, 55]
NCArray *diff = nc_subtract(b, a);   // [9, 18, 27, 36, 45]
NCArray *prod = nc_multiply(a, b);   // [10, 40, 90, 160, 250]
NCArray *quot = nc_divide(b, a);     // [10, 10, 10, 10, 10]

NCArray *power = nc_power(a, NC_INT(2, 2, 2, 2, 2));  // [1, 4, 9, 16, 25]
```

### Math Functions

```c
#include <math.h>  // For M_PI

NCArray *x = NC_FLOAT(0.0, M_PI/4, M_PI/2, M_PI);

nc_sin(x);    // Sine
nc_cos(x);    // Cosine
nc_tan(x);    // Tangent
nc_exp(x);    // e^x
nc_log(x);    // Natural logarithm
nc_sqrt(x);   // Square root
nc_abs(NC_FLOAT(-1, 2, -3));  // [1, 2, 3]
```

### Reductions

```c
NCArray *arr = NC_INT(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

NCArray *s = nc_sum(arr, NULL, 0);    // 55
NCArray *m = nc_mean(arr, NULL, 0);   // 5.5
NCArray *mn = nc_min(arr, NULL, 0);   // 1
NCArray *mx = nc_max(arr, NULL, 0);   // 10

double sum_val = nc_get_value_as_double(s, 0);  // 55.0
```

### Memory Management

```c
// Create array
NCArray *arr = NC_INT(1, 2, 3);

// Reference counting
NCArray *ref = nc_retain(arr);  // Increment reference count

// Free when done
nc_free(arr);    // Decrement reference count
nc_free(ref);    // When count reaches 0, memory is freed

// Alternative name
nc_release(arr); // Same as nc_free()
```

---

## Summary Table

| Feature | Code | Description |
|---------|------|-------------|
| 1D integer | `NC_INT(1, 2, 3)` | Auto type detection |
| 1D float | `NC_FLOAT(1.0, 2.0)` | Auto float32/float64 |
| 2D array | `NC_INT2D(2, 3, 1,2,3,4,5,6)` | Row-major order |
| Fixed-point | `nc_fixed_from_values(8,8,true,arr,n)` | Q8.8 format |
| Matrix multiply | `nc_matmul(A, B)` | A @ B |
| Dot product | `nc_dot(v1, v2)` | Vector dot product |
| Transpose | `nctranspose(A, NULL)` | Swap rows/columns |
| Sum | `nc_sum(arr, NULL, 0)` | Sum all elements |

---

## License

GNU General Public License v3 (GPLv3)