#include "NumC.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <ctype.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#define NC_HAVE_OPENMP 1
#else
#define NC_HAVE_OPENMP 0
#endif

NCDataType NC_DEFAULT_INT = NC_INT64;
NCDataType NC_DEFAULT_FLOAT = NC_FLOAT64;

static uint64_t nc_random_state = 1;

const char *nc_version(void) {
    return NC_VERSION;
}

const char *nc_status_string(NCStatus status) {
    switch (status) {
        case NC_OK: return "No error";
        case NC_ERROR: return "Generic error";
        case NC_MEMORY_ERROR: return "Memory allocation failed";
        case NC_DIMENSION_ERROR: return "Dimension mismatch";
        case NC_SHAPE_ERROR: return "Shape mismatch";
        case NC_TYPE_ERROR: return "Type mismatch";
        case NC_VALUE_ERROR: return "Value error";
        case NC_INDEX_ERROR: return "Index error";
        case NC_NOT_IMPLEMENTED: return "Not implemented";
        default: return "Unknown error";
    }
}

size_t nc_dtype_size(NCDataType dtype) {
    switch (dtype) {
        case NC_BOOL: return sizeof(bool);
        case NC_INT8: return sizeof(int8_t);
        case NC_INT16: return sizeof(int16_t);
        case NC_INT32: return sizeof(int32_t);
        case NC_INT64: return sizeof(int64_t);
        case NC_UINT8: return sizeof(uint8_t);
        case NC_UINT16: return sizeof(uint16_t);
        case NC_UINT32: return sizeof(uint32_t);
        case NC_UINT64: return sizeof(uint64_t);
        case NC_FLOAT32: return sizeof(float);
        case NC_FLOAT64: return sizeof(double);
        case NC_COMPLEX64: return sizeof(NCComplex64);
        case NC_COMPLEX128: return sizeof(NCComplex128);
        default: return 0;
    }
}

const char *nc_dtype_name(NCDataType dtype) {
    switch (dtype) {
        case NC_BOOL: return "bool";
        case NC_INT8: return "int8";
        case NC_INT16: return "int16";
        case NC_INT32: return "int32";
        case NC_INT64: return "int64";
        case NC_UINT8: return "uint8";
        case NC_UINT16: return "uint16";
        case NC_UINT32: return "uint32";
        case NC_UINT64: return "uint64";
        case NC_FLOAT32: return "float32";
        case NC_FLOAT64: return "float64";
        case NC_COMPLEX64: return "complex64";
        case NC_COMPLEX128: return "complex128";
        default: return "unknown";
    }
}

NCDataType nc_dtype_from_string(const char *name) {
    if (strcmp(name, "bool") == 0) return NC_BOOL;
    if (strcmp(name, "int8") == 0) return NC_INT8;
    if (strcmp(name, "int16") == 0) return NC_INT16;
    if (strcmp(name, "int32") == 0) return NC_INT32;
    if (strcmp(name, "int64") == 0) return NC_INT64;
    if (strcmp(name, "uint8") == 0) return NC_UINT8;
    if (strcmp(name, "uint16") == 0) return NC_UINT16;
    if (strcmp(name, "uint32") == 0) return NC_UINT32;
    if (strcmp(name, "uint64") == 0) return NC_UINT64;
    if (strcmp(name, "float32") == 0) return NC_FLOAT32;
    if (strcmp(name, "float64") == 0) return NC_FLOAT64;
    if (strcmp(name, "complex64") == 0) return NC_COMPLEX64;
    if (strcmp(name, "complex128") == 0) return NC_COMPLEX128;
    return NC_INVALID;
}

bool nc_dtype_is_integer(NCDataType dtype) {
    return dtype >= NC_BOOL && dtype <= NC_UINT64;
}

bool nc_dtype_is_float(NCDataType dtype) {
    return dtype == NC_FLOAT32 || dtype == NC_FLOAT64;
}

bool nc_dtype_is_complex(NCDataType dtype) {
    return dtype == NC_COMPLEX64 || dtype == NC_COMPLEX128;
}

bool nc_dtype_is_numeric(NCDataType dtype) {
    return nc_dtype_is_integer(dtype) || nc_dtype_is_float(dtype) || nc_dtype_is_complex(dtype);
}

size_t nc_size(NCArray *arr) {
    if (!arr) return 0;
    size_t size = 1;
    for (int32_t i = 0; i < arr->ndim; i++) {
        size *= arr->shape[i];
    }
    return size;
}

size_t nc_nbytes(NCArray *arr) {
    return nc_size(arr) * arr->itemsize;
}

int32_t nc_ndim(NCArray *arr) {
    return arr ? arr->ndim : 0;
}

int64_t *nc_shape(NCArray *arr) {
    return arr ? arr->shape : NULL;
}

int64_t nc_shape_at(NCArray *arr, int32_t dim) {
    if (!arr || dim < 0 || dim >= arr->ndim) return 0;
    return arr->shape[dim];
}

int64_t *nc_strides(NCArray *arr) {
    return arr ? arr->strides : NULL;
}

int64_t nc_stride_at(NCArray *arr, int32_t dim) {
    if (!arr || dim < 0 || dim >= arr->ndim) return 0;
    return arr->strides[dim];
}

size_t nc_itemsize(NCArray *arr) {
    return arr ? arr->itemsize : 0;
}

NCDataType nc_dtype(NCArray *arr) {
    return arr ? arr->dtype : NC_INVALID;
}

void nc_update_strides(NCArray *arr) {
    if (!arr) return;
    if (arr->ndim == 0) return;
    arr->strides[arr->ndim - 1] = 1;
    for (int32_t i = arr->ndim - 2; i >= 0; i--) {
        arr->strides[i] = arr->strides[i + 1] * arr->shape[i + 1];
    }
}

bool nc_is_contiguous(NCArray *arr) {
    if (!arr) return false;
    int64_t expected_stride = 1;
    for (int32_t i = arr->ndim - 1; i >= 0; i--) {
        if (arr->shape[i] == 1) continue;
        if (arr->strides[i] != expected_stride) return false;
        expected_stride *= arr->shape[i];
    }
    return true;
}

bool nc_is_c_contiguous(NCArray *arr) {
    return nc_is_contiguous(arr);
}

bool nc_is_f_contiguous(NCArray *arr) {
    if (!arr) return false;
    int64_t expected_stride = 1;
    for (int32_t i = 0; i < arr->ndim; i++) {
        if (arr->shape[i] == 1) continue;
        if (arr->strides[i] != expected_stride) return false;
        expected_stride *= arr->shape[i];
    }
    return true;
}

static NCArray *nc_array_create(void *data, int32_t ndim, const int64_t *shape, 
                                NCDataType dtype, bool owns_data) {
    if (!shape || ndim < 0 || ndim > NC_MAX_DIMS) {
        return NULL;
    }
    
    NCArray *arr = (NCArray *)calloc(1, sizeof(NCArray));
    if (!arr) return NULL;
    
    arr->ndim = ndim;
    arr->dtype = dtype;
    arr->itemsize = nc_dtype_size(dtype);
    arr->owns_data = owns_data;
    arr->refcount = 1;
    arr->base = NULL;
    
    for (int32_t i = 0; i < ndim; i++) {
        arr->shape[i] = shape[i];
    }
    
    size_t total_size = 1;
    for (int32_t i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    
    if (data) {
        arr->data = data;
    } else if (owns_data) {
        arr->data = calloc(total_size, arr->itemsize);
        if (!arr->data) {
            free(arr);
            return NULL;
        }
    }
    
    nc_update_strides(arr);
    return arr;
}

NCArray *nc_empty(int32_t ndim, const int64_t *shape, NCDataType dtype) {
    return nc_array_create(NULL, ndim, shape, dtype, true);
}

NCArray *nc_empty_like(NCArray *arr) {
    if (!arr) return NULL;
    return nc_empty(arr->ndim, arr->shape, arr->dtype);
}

NCArray *nc_zeros(int32_t ndim, const int64_t *shape, NCDataType dtype) {
    NCArray *arr = nc_empty(ndim, shape, dtype);
    if (arr && arr->data) {
        memset(arr->data, 0, nc_nbytes(arr));
    }
    return arr;
}

NCArray *nc_zeros_like(NCArray *arr) {
    if (!arr) return NULL;
    return nc_zeros(arr->ndim, arr->shape, arr->dtype);
}

NCArray *nc_ones(int32_t ndim, const int64_t *shape, NCDataType dtype) {
    NCArray *arr = nc_empty(ndim, shape, dtype);
    if (!arr) return NULL;
    
    size_t total = nc_size(arr);
    size_t esize = arr->itemsize;
    char *ptr = (char *)arr->data;
    
    for (size_t i = 0; i < total; i++) {
        switch (dtype) {
            case NC_BOOL: ((bool*)ptr)[i] = 1; break;
            case NC_INT8: ((int8_t*)ptr)[i] = 1; break;
            case NC_INT16: ((int16_t*)ptr)[i] = 1; break;
            case NC_INT32: ((int32_t*)ptr)[i] = 1; break;
            case NC_INT64: ((int64_t*)ptr)[i] = 1; break;
            case NC_UINT8: ((uint8_t*)ptr)[i] = 1; break;
            case NC_UINT16: ((uint16_t*)ptr)[i] = 1; break;
            case NC_UINT32: ((uint32_t*)ptr)[i] = 1; break;
            case NC_UINT64: ((uint64_t*)ptr)[i] = 1; break;
            case NC_FLOAT32: ((float*)ptr)[i] = 1.0f; break;
            case NC_FLOAT64: ((double*)ptr)[i] = 1.0; break;
            case NC_COMPLEX64: ((NCComplex64*)ptr)[i].real = 1.0f; ((NCComplex64*)ptr)[i].imag = 0.0f; break;
            case NC_COMPLEX128: ((NCComplex128*)ptr)[i].real = 1.0; ((NCComplex128*)ptr)[i].imag = 0.0; break;
            default: break;
        }
    }
    return arr;
}

NCArray *nc_ones_like(NCArray *arr) {
    if (!arr) return NULL;
    return nc_ones(arr->ndim, arr->shape, arr->dtype);
}

NCArray *nc_full(int32_t ndim, const int64_t *shape, void *fill_value, NCDataType dtype) {
    NCArray *arr = nc_empty(ndim, shape, dtype);
    if (!arr) return NULL;
    
    size_t total = nc_size(arr);
    size_t esize = arr->itemsize;
    char *ptr = (char *)arr->data;
    
    for (size_t i = 0; i < total; i++) {
        memcpy(ptr + i * esize, fill_value, esize);
    }
    return arr;
}

NCArray *nc_full_like(NCArray *arr, void *fill_value) {
    if (!arr) return NULL;
    return nc_full(arr->ndim, arr->shape, fill_value, arr->dtype);
}

NCArray *nc_arange(double start, double stop, double step, NCDataType dtype) {
    if (step == 0) return NULL;
    
    int64_t n = (int64_t)ceil((stop - start) / step);
    if (n < 0) n = 0;
    
    int64_t shape[1] = {n};
    NCArray *arr = nc_empty(1, shape, dtype);
    if (!arr) return NULL;
    
    size_t esize = arr->itemsize;
    char *ptr = (char *)arr->data;
    
    for (int64_t i = 0; i < n; i++) {
        double value = start + i * step;
        switch (dtype) {
            case NC_BOOL: ((bool*)ptr)[i] = (bool)value; break;
            case NC_INT8: ((int8_t*)ptr)[i] = (int8_t)value; break;
            case NC_INT16: ((int16_t*)ptr)[i] = (int16_t)value; break;
            case NC_INT32: ((int32_t*)ptr)[i] = (int32_t)value; break;
            case NC_INT64: ((int64_t*)ptr)[i] = (int64_t)value; break;
            case NC_UINT8: ((uint8_t*)ptr)[i] = (uint8_t)value; break;
            case NC_UINT16: ((uint16_t*)ptr)[i] = (uint16_t)value; break;
            case NC_UINT32: ((uint32_t*)ptr)[i] = (uint32_t)value; break;
            case NC_UINT64: ((uint64_t*)ptr)[i] = (uint64_t)value; break;
            case NC_FLOAT32: ((float*)ptr)[i] = (float)value; break;
            case NC_FLOAT64: ((double*)ptr)[i] = value; break;
            default: break;
        }
    }
    return arr;
}

NCArray *nc_linspace(double start, double stop, int64_t num, bool endpoint, NCDataType dtype) {
    if (num < 0) return NULL;
    if (num == 0) {
        int64_t shape[1] = {0};
        return nc_empty(1, shape, dtype);
    }
    if (num == 1) {
        int64_t shape[1] = {1};
        NCArray *arr = nc_empty(1, shape, dtype);
        if (!arr) return NULL;
        switch (dtype) {
            case NC_FLOAT32: ((float*)arr->data)[0] = (float)start; break;
            case NC_FLOAT64: ((double*)arr->data)[0] = start; break;
            default: ((double*)arr->data)[0] = start; break;
        }
        return arr;
    }
    
    int64_t n = endpoint ? num : num - 1;
    double step = (stop - start) / n;
    
    int64_t shape[1] = {num};
    NCArray *arr = nc_empty(1, shape, dtype);
    if (!arr) return NULL;
    
    size_t esize = arr->itemsize;
    char *ptr = (char *)arr->data;
    
    for (int64_t i = 0; i < num; i++) {
        double value = start + i * step;
        switch (dtype) {
            case NC_BOOL: ((bool*)ptr)[i] = (bool)value; break;
            case NC_INT8: ((int8_t*)ptr)[i] = (int8_t)value; break;
            case NC_INT16: ((int16_t*)ptr)[i] = (int16_t)value; break;
            case NC_INT32: ((int32_t*)ptr)[i] = (int32_t)value; break;
            case NC_INT64: ((int64_t*)ptr)[i] = (int64_t)value; break;
            case NC_UINT8: ((uint8_t*)ptr)[i] = (uint8_t)value; break;
            case NC_UINT16: ((uint16_t*)ptr)[i] = (uint16_t)value; break;
            case NC_UINT32: ((uint32_t*)ptr)[i] = (uint32_t)value; break;
            case NC_UINT64: ((uint64_t*)ptr)[i] = (uint64_t)value; break;
            case NC_FLOAT32: ((float*)ptr)[i] = (float)value; break;
            case NC_FLOAT64: ((double*)ptr)[i] = value; break;
            default: ((double*)ptr)[i] = value; break;
        }
    }
    return arr;
}

NCArray *nc_logspace(double start, double stop, int64_t num, bool endpoint, double base, NCDataType dtype) {
    NCArray *arr = nc_linspace(start, stop, num, endpoint, NC_FLOAT64);
    if (!arr) return NULL;
    
    double *ptr = (double *)arr->data;
    double b = (base <= 0) ? 10.0 : base;
    
    for (int64_t i = 0; i < num; i++) {
        ptr[i] = pow(b, ptr[i]);
    }
    
    if (dtype != NC_FLOAT64) {
        NCArray *result = nc_typecast(arr, dtype, NC_SAFE_CAST);
        nc_release(arr);
        return result;
    }
    return arr;
}

NCArray *nc_geomspace(double start, double stop, int64_t num, bool endpoint, NCDataType dtype) {
    if (num < 0) return NULL;
    if (num == 0) {
        int64_t shape[1] = {0};
        return nc_empty(1, shape, dtype);
    }
    if (num == 1) {
        int64_t shape[1] = {1};
        NCArray *arr = nc_empty(1, shape, dtype);
        if (!arr) return NULL;
        switch (dtype) {
            case NC_FLOAT32: ((float*)arr->data)[0] = (float)start; break;
            case NC_FLOAT64: ((double*)arr->data)[0] = start; break;
            default: ((double*)arr->data)[0] = start; break;
        }
        return arr;
    }
    
    double ratio;
    if (endpoint) {
        ratio = pow(stop / start, 1.0 / (num - 1));
    } else {
        ratio = pow(stop / start, 1.0 / num);
    }
    
    int64_t shape[1] = {num};
    NCArray *arr = nc_empty(1, shape, dtype);
    if (!arr) return NULL;
    
    size_t esize = arr->itemsize;
    char *ptr = (char *)arr->data;
    
    for (int64_t i = 0; i < num; i++) {
        double value = start * pow(ratio, i);
        switch (dtype) {
            case NC_BOOL: ((bool*)ptr)[i] = (bool)value; break;
            case NC_INT8: ((int8_t*)ptr)[i] = (int8_t)value; break;
            case NC_INT16: ((int16_t*)ptr)[i] = (int16_t)value; break;
            case NC_INT32: ((int32_t*)ptr)[i] = (int32_t)value; break;
            case NC_INT64: ((int64_t*)ptr)[i] = (int64_t)value; break;
            case NC_UINT8: ((uint8_t*)ptr)[i] = (uint8_t)value; break;
            case NC_UINT16: ((uint16_t*)ptr)[i] = (uint16_t)value; break;
            case NC_UINT32: ((uint32_t*)ptr)[i] = (uint32_t)value; break;
            case NC_UINT64: ((uint64_t*)ptr)[i] = (uint64_t)value; break;
            case NC_FLOAT32: ((float*)ptr)[i] = (float)value; break;
            case NC_FLOAT64: ((double*)ptr)[i] = value; break;
            default: ((double*)ptr)[i] = value; break;
        }
    }
    return arr;
}

NCArray *nc_identity(int64_t n, NCDataType dtype) {
    int64_t shape[2] = {n, n};
    NCArray *arr = nc_zeros(2, shape, dtype);
    if (!arr) return NULL;
    
    char *ptr = (char *)arr->data;
    size_t esize = arr->itemsize;
    
    for (int64_t i = 0; i < n; i++) {
        size_t idx = i * n + i;
        switch (dtype) {
            case NC_BOOL: ((bool*)ptr)[idx] = true; break;
            case NC_INT8: ((int8_t*)ptr)[idx] = 1; break;
            case NC_INT16: ((int16_t*)ptr)[idx] = 1; break;
            case NC_INT32: ((int32_t*)ptr)[idx] = 1; break;
            case NC_INT64: ((int64_t*)ptr)[idx] = 1; break;
            case NC_UINT8: ((uint8_t*)ptr)[idx] = 1; break;
            case NC_UINT16: ((uint16_t*)ptr)[idx] = 1; break;
            case NC_UINT32: ((uint32_t*)ptr)[idx] = 1; break;
            case NC_UINT64: ((uint64_t*)ptr)[idx] = 1; break;
            case NC_FLOAT32: ((float*)ptr)[idx] = 1.0f; break;
            case NC_FLOAT64: ((double*)ptr)[idx] = 1.0; break;
            default: break;
        }
    }
    return arr;
}

NCArray *nc_eye(int64_t n, int64_t m, int64_t k, NCDataType dtype) {
    int64_t shape[2] = {n, m};
    NCArray *arr = nc_zeros(2, shape, dtype);
    if (!arr) return NULL;
    
    char *ptr = (char *)arr->data;
    size_t esize = arr->itemsize;
    
    int64_t max_diag = (n < m) ? n : m;
    int64_t start = (k >= 0) ? 0 : -k;
    int64_t end = (k >= 0) ? (max_diag - k) : max_diag;
    
    for (int64_t i = start; i < end && i < n && (i + k) < m; i++) {
        size_t idx = i * m + (i + k);
        switch (dtype) {
            case NC_BOOL: ((bool*)ptr)[idx] = true; break;
            case NC_INT8: ((int8_t*)ptr)[idx] = 1; break;
            case NC_INT16: ((int16_t*)ptr)[idx] = 1; break;
            case NC_INT32: ((int32_t*)ptr)[idx] = 1; break;
            case NC_INT64: ((int64_t*)ptr)[idx] = 1; break;
            case NC_UINT8: ((uint8_t*)ptr)[idx] = 1; break;
            case NC_UINT16: ((uint16_t*)ptr)[idx] = 1; break;
            case NC_UINT32: ((uint32_t*)ptr)[idx] = 1; break;
            case NC_UINT64: ((uint64_t*)ptr)[idx] = 1; break;
            case NC_FLOAT32: ((float*)ptr)[idx] = 1.0f; break;
            case NC_FLOAT64: ((double*)ptr)[idx] = 1.0; break;
            default: break;
        }
    }
    return arr;
}

NCArray *nc_diag(NCArray *arr, int64_t k) {
    if (!arr) return NULL;
    
    if (arr->ndim == 1) {
        int64_t n = arr->shape[0] + llabs(k);
        int64_t shape[2] = {n, n};
        NCArray *result = nc_zeros(2, shape, arr->dtype);
        if (!result) return NULL;
        
        char *rptr = (char *)result->data;
        char *aptr = (char *)arr->data;
        size_t esize = arr->itemsize;
        int64_t m = n;
        
        for (int64_t i = 0; i < arr->shape[0]; i++) {
            int64_t row = (k >= 0) ? i : i - k;
            int64_t col = (k >= 0) ? i + k : i;
            if (row >= 0 && row < n && col >= 0 && col < m) {
                memcpy(rptr + (row * m + col) * esize, aptr + i * esize, esize);
            }
        }
        return result;
    } else if (arr->ndim == 2) {
        int64_t n = (arr->shape[0] < arr->shape[1]) ? arr->shape[0] : arr->shape[1];
        int64_t diag_len = n - llabs(k);
        if (diag_len < 0) diag_len = 0;
        
        int64_t shape[1] = {diag_len};
        NCArray *result = nc_empty(1, shape, arr->dtype);
        if (!result) return NULL;
        
        char *rptr = (char *)result->data;
        char *aptr = (char *)arr->data;
        size_t esize = arr->itemsize;
        int64_t m = arr->shape[1];
        
        for (int64_t i = 0; i < diag_len; i++) {
            int64_t row = (k >= 0) ? i : i - k;
            int64_t col = (k >= 0) ? i + k : i;
            memcpy(rptr + i * esize, aptr + (row * m + col) * esize, esize);
        }
        return result;
    }
    return NULL;
}

NCArray *nc_tril(NCArray *arr, int64_t k) {
    if (!arr || arr->ndim != 2) return NULL;
    
    NCArray *result = nc_copy(arr);
    if (!result) return NULL;
    
    char *ptr = (char *)result->data;
    size_t esize = result->itemsize;
    int64_t n = arr->shape[0];
    int64_t m = arr->shape[1];
    
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < m; j++) {
            if (j < i - k) {
                memset(ptr + (i * m + j) * esize, 0, esize);
            }
        }
    }
    return result;
}

NCArray *nc_triu(NCArray *arr, int64_t k) {
    if (!arr || arr->ndim != 2) return NULL;
    
    NCArray *result = nc_copy(arr);
    if (!result) return NULL;
    
    char *ptr = (char *)result->data;
    size_t esize = result->itemsize;
    int64_t n = arr->shape[0];
    int64_t m = arr->shape[1];
    
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < m; j++) {
            if (j > i + k) {
                memset(ptr + (i * m + j) * esize, 0, esize);
            }
        }
    }
    return result;
}

NCArray *nc_from_buffer(void *buffer, size_t size, NCDataType dtype, int32_t ndim, const int64_t *shape) {
    size_t expected_size = 1;
    for (int32_t i = 0; i < ndim; i++) {
        expected_size *= shape[i];
    }
    expected_size *= nc_dtype_size(dtype);
    
    if (size < expected_size) return NULL;
    return nc_array_create(buffer, ndim, shape, dtype, false);
}

NCArray *nc_from_pointer(void *ptr, int32_t ndim, const int64_t *shape, NCDataType dtype, bool owns_data) {
    return nc_array_create(ptr, ndim, shape, dtype, owns_data);
}

NCArray *nc_copy(NCArray *arr) {
    if (!arr) return NULL;
    
    NCArray *result = nc_empty(arr->ndim, arr->shape, arr->dtype);
    if (!result) return NULL;
    
    memcpy(result->data, arr->data, nc_nbytes(arr));
    return result;
}

NCArray *nc_asarray(NCArray *arr) {
    if (!arr) return NULL;
    return nc_copy(arr);
}

NCArray *nc_asarray_c(void *data, int32_t ndim, const int64_t *shape, NCDataType dtype) {
    return nc_array_create(data, ndim, shape, dtype, false);
}

NCArray *nc_asarray_f(void *data, int32_t ndim, const int64_t *shape, NCDataType dtype) {
    NCArray *arr = nc_array_create(data, ndim, shape, dtype, false);
    if (!arr) return NULL;
    
    arr->strides[ndim - 1] = 1;
    for (int32_t i = ndim - 2; i >= 0; i--) {
        arr->strides[i] = arr->strides[i + 1] * shape[i + 1];
    }
    return arr;
}

NCArray *nc_ascontiguousarray(NCArray *arr) {
    if (!arr) return NULL;
    if (nc_is_c_contiguous(arr)) return nc_copy(arr);
    
    NCArray *result = nc_empty(arr->ndim, arr->shape, arr->dtype);
    if (!result) return NULL;
    
    size_t esize = arr->itemsize;
    int64_t total = nc_size(arr);
    char *src = (char *)arr->data;
    char *dst = (char *)result->data;
    
    for (int64_t i = 0; i < total; i++) {
        int64_t src_idx = 0;
        int64_t temp = i;
        for (int32_t d = 0; d < arr->ndim; d++) {
            int32_t dim = arr->ndim - 1 - d;
            int64_t idx_in_dim = temp % arr->shape[dim];
            src_idx += idx_in_dim * arr->strides[dim];
            temp /= arr->shape[dim];
        }
        memcpy(dst + i * esize, src + src_idx * esize, esize);
    }
    return result;
}

NCStatus nc_reshape(NCArray *arr, int32_t ndim, const int64_t *shape) {
    if (!arr || !shape) return NC_ERROR;
    
    size_t total = nc_size(arr);
    size_t new_total = 1;
    for (int32_t i = 0; i < ndim; i++) {
        new_total *= shape[i];
    }
    
    if (total != new_total) return NC_SHAPE_ERROR;
    
    arr->ndim = ndim;
    for (int32_t i = 0; i < ndim; i++) {
        arr->shape[i] = shape[i];
    }
    nc_update_strides(arr);
    return NC_OK;
}

NCStatus nc_resize(NCArray *arr, int32_t ndim, const int64_t *shape) {
    if (!arr || !shape) return NC_ERROR;
    
    size_t new_total = 1;
    for (int32_t i = 0; i < ndim; i++) {
        new_total *= shape[i];
    }
    
    if (new_total > NC_MAX_TENSOR_SIZE) return NC_MEMORY_ERROR;
    if (!arr->owns_data) return NC_ERROR;
    
    size_t curr_size = nc_size(arr);
    
    if (new_total > curr_size) {
        void *new_data = realloc(arr->data, new_total * arr->itemsize);
        if (!new_data) return NC_MEMORY_ERROR;
        arr->data = new_data;
        memset((char*)arr->data + curr_size * arr->itemsize, 0, 
               (new_total - curr_size) * arr->itemsize);
    }
    
    arr->ndim = ndim;
    for (int32_t i = 0; i < ndim; i++) {
        arr->shape[i] = shape[i];
    }
    nc_update_strides(arr);
    return NC_OK;
}

NCStatus nc_squeeze(NCArray *arr, int32_t axis) {
    if (!arr) return NC_ERROR;
    
    if (axis >= 0) {
        if (axis >= arr->ndim) return NC_INDEX_ERROR;
        if (arr->shape[axis] != 1) return NC_VALUE_ERROR;
        
        for (int32_t i = axis; i < arr->ndim - 1; i++) {
            arr->shape[i] = arr->shape[i + 1];
            arr->strides[i] = arr->strides[i + 1];
        }
        arr->ndim--;
    } else {
        int32_t write_idx = 0;
        for (int32_t i = 0; i < arr->ndim; i++) {
            if (arr->shape[i] != 1) {
                if (write_idx != i) {
                    arr->shape[write_idx] = arr->shape[i];
                    arr->strides[write_idx] = arr->strides[i];
                }
                write_idx++;
            }
        }
        arr->ndim = write_idx;
    }
    return NC_OK;
}

NCStatus nc_expand_dims(NCArray *arr, int32_t axis) {
    if (!arr) return NC_ERROR;
    if (arr->ndim >= NC_MAX_DIMS) return NC_DIMENSION_ERROR;
    if (axis < 0 || axis > arr->ndim) return NC_INDEX_ERROR;
    
    for (int32_t i = arr->ndim; i > axis; i--) {
        arr->shape[i] = arr->shape[i - 1];
        arr->strides[i] = arr->strides[i - 1];
    }
    arr->shape[axis] = 1;
    arr->strides[axis] = (axis == arr->ndim) ? 1 : arr->strides[axis + 1];
    arr->ndim++;
    return NC_OK;
}

NCStatus nc_swapaxes(NCArray *arr, int32_t axis1, int32_t axis2) {
    if (!arr) return NC_ERROR;
    if (axis1 < 0) axis1 += arr->ndim;
    if (axis2 < 0) axis2 += arr->ndim;
    if (axis1 < 0 || axis1 >= arr->ndim || axis2 < 0 || axis2 >= arr->ndim) {
        return NC_INDEX_ERROR;
    }
    
    int64_t tmp = arr->shape[axis1];
    arr->shape[axis1] = arr->shape[axis2];
    arr->shape[axis2] = tmp;
    
    tmp = arr->strides[axis1];
    arr->strides[axis1] = arr->strides[axis2];
    arr->strides[axis2] = tmp;
    
    return NC_OK;
}

NCStatus nc_moveaxis(NCArray *arr, int32_t source, int32_t dest) {
    if (!arr) return NC_ERROR;
    if (source < 0) source += arr->ndim;
    if (dest < 0) dest += arr->ndim;
    if (source < 0 || source >= arr->ndim || dest < 0 || dest >= arr->ndim) {
        return NC_INDEX_ERROR;
    }
    
    if (source == dest) return NC_OK;
    
    int64_t tmp_shape[NC_MAX_DIMS];
    int64_t tmp_strides[NC_MAX_DIMS];
    
    memcpy(tmp_shape, arr->shape, arr->ndim * sizeof(int64_t));
    memcpy(tmp_strides, arr->strides, arr->ndim * sizeof(int64_t));
    
    if (source < dest) {
        for (int32_t i = source; i < dest; i++) {
            arr->shape[i] = tmp_shape[i + 1];
            arr->strides[i] = tmp_strides[i + 1];
        }
        arr->shape[dest] = tmp_shape[source];
        arr->strides[dest] = tmp_strides[source];
    } else {
        for (int32_t i = source; i > dest; i--) {
            arr->shape[i] = tmp_shape[i - 1];
            arr->strides[i] = tmp_strides[i - 1];
        }
        arr->shape[dest] = tmp_shape[source];
        arr->strides[dest] = tmp_strides[source];
    }
    
    return NC_OK;
}

NCArray *nctranspose(NCArray *arr, const int32_t *axes) {
    if (!arr) return NULL;
    
    NCArray *result = nc_empty(arr->ndim, arr->shape, arr->dtype);
    if (!result) return NULL;
    
    if (axes) {
        for (int32_t i = 0; i < arr->ndim; i++) {
            result->shape[i] = arr->shape[axes[i]];
            result->strides[i] = arr->strides[axes[i]];
        }
    } else {
        for (int32_t i = 0; i < arr->ndim; i++) {
            result->shape[i] = arr->shape[arr->ndim - 1 - i];
            result->strides[i] = arr->strides[arr->ndim - 1 - i];
        }
    }
    
    result->data = arr->data;
    result->owns_data = false;
    result->base = arr;
    arr->refcount++;
    
    return result;
}

NCArray *ncflatten(NCArray *arr) {
    if (!arr) return NULL;
    int64_t shape[1] = {nc_size(arr)};
    return ncreshape(arr, 1, shape);
}

NCArray *ncravel(NCArray *arr) {
    return ncflatten(arr);
}

NCArray *ncreshape(NCArray *arr, int32_t ndim, const int64_t *shape) {
    if (!arr) return NULL;
    NCArray *result = nc_copy(arr);
    if (!result) return NULL;
    if (nc_reshape(result, ndim, shape) != NC_OK) {
        nc_release(result);
        return NULL;
    }
    return result;
}

NCArray *ncsqueeze(NCArray *arr, int32_t axis) {
    if (!arr) return NULL;
    NCArray *result = nc_copy(arr);
    if (!result) return NULL;
    if (nc_squeeze(result, axis) != NC_OK) {
        nc_release(result);
        return NULL;
    }
    return result;
}

NCArray *ncexpand_dims(NCArray *arr, int32_t axis) {
    if (!arr) return NULL;
    NCArray *result = nc_copy(arr);
    if (!result) return NULL;
    if (nc_expand_dims(result, axis) != NC_OK) {
        nc_release(result);
        return NULL;
    }
    return result;
}

NCArray *ncselect(NCArray *cond, NCArray *arr1, NCArray *arr2) {
    if (!cond || !arr1 || !arr2) return NULL;
    
    NCArray *result = nc_empty(cond->ndim, cond->shape, arr1->dtype);
    if (!result) return NULL;
    
    size_t esize = result->itemsize;
    size_t total = nc_size(cond);
    char *cond_data = (char *)cond->data;
    char *arr1_data = (char *)arr1->data;
    char *arr2_data = (char *)arr2->data;
    char *result_data = (char *)result->data;
    
    for (size_t i = 0; i < total; i++) {
        bool c = ((bool*)cond_data)[i];
        memcpy(result_data + i * esize, c ? (arr1_data + i * esize) : (arr2_data + i * esize), esize);
    }
    return result;
}

static int64_t nc_compute_broadcast_index(NCArray *arr, int32_t result_ndim, int32_t dim, int64_t linear_idx, const int64_t *result_shape) {
    int32_t arr_dim = dim - (result_ndim - arr->ndim);
    if (arr_dim < 0) return 0;
    int64_t idx_in_dim = 0;
    int64_t temp = linear_idx;
    for (int32_t d = result_ndim - 1; d >= 0; d--) {
        int64_t dim_size = result_shape[d];
        if (d == dim) {
            idx_in_dim = temp % dim_size;
            break;
        }
        temp /= dim_size;
    }
    if (arr->shape[arr_dim] == 1) return 0;
    return idx_in_dim;
}

static double nc_get_value_as_double(NCArray *arr, int64_t linear_idx) {
    char *data = (char *)arr->data;
    int64_t idx = 0;
    int64_t temp = linear_idx;
    for (int32_t d = arr->ndim - 1; d >= 0; d--) {
        int64_t idx_in_dim = temp % arr->shape[d];
        idx += idx_in_dim * arr->strides[d];
        temp /= arr->shape[d];
    }
    switch (arr->dtype) {
        case NC_BOOL: return ((bool*)data)[idx] ? 1.0 : 0.0;
        case NC_INT8: return (double)((int8_t*)data)[idx];
        case NC_INT16: return (double)((int16_t*)data)[idx];
        case NC_INT32: return (double)((int32_t*)data)[idx];
        case NC_INT64: return (double)((int64_t*)data)[idx];
        case NC_UINT8: return (double)((uint8_t*)data)[idx];
        case NC_UINT16: return (double)((uint16_t*)data)[idx];
        case NC_UINT32: return (double)((uint32_t*)data)[idx];
        case NC_UINT64: return (double)((uint64_t*)data)[idx];
        case NC_FLOAT32: return (double)((float*)data)[idx];
        case NC_FLOAT64: return ((double*)data)[idx];
        default: return 0.0;
    }
}

static void nc_set_value_from_double(NCArray *arr, int64_t linear_idx, double value) {
    char *data = (char *)arr->data;
    int64_t idx = 0;
    int64_t temp = linear_idx;
    for (int32_t d = arr->ndim - 1; d >= 0; d--) {
        int64_t idx_in_dim = temp % arr->shape[d];
        idx += idx_in_dim * arr->strides[d];
        temp /= arr->shape[d];
    }
    switch (arr->dtype) {
        case NC_BOOL: ((bool*)data)[idx] = value != 0; break;
        case NC_INT8: ((int8_t*)data)[idx] = (int8_t)value; break;
        case NC_INT16: ((int16_t*)data)[idx] = (int16_t)value; break;
        case NC_INT32: ((int32_t*)data)[idx] = (int32_t)value; break;
        case NC_INT64: ((int64_t*)data)[idx] = (int64_t)value; break;
        case NC_UINT8: ((uint8_t*)data)[idx] = (uint8_t)value; break;
        case NC_UINT16: ((uint16_t*)data)[idx] = (uint16_t)value; break;
        case NC_UINT32: ((uint32_t*)data)[idx] = (uint32_t)value; break;
        case NC_UINT64: ((uint64_t*)data)[idx] = (uint64_t)value; break;
        case NC_FLOAT32: ((float*)data)[idx] = (float)value; break;
        case NC_FLOAT64: ((double*)data)[idx] = value; break;
        default: break;
    }
}

int32_t nc_broadcast_shapes(int32_t n_in, const int64_t **shapes, int32_t n_shapes, int64_t *out_shape) {
    if (n_shapes == 0 || !shapes || !out_shape) return -1;
    
    int32_t max_ndim = 0;
    for (int32_t i = 0; i < n_shapes; i++) {
        if (shapes[i]) {
            int32_t ndim = n_in;
            for (int32_t d = 0; d < ndim; d++) {
                int32_t out_dim = d + (NC_MAX_DIMS - ndim);
                if (out_dim >= NC_MAX_DIMS) continue;
                int64_t dim_size = shapes[i][d];
                if (out_shape[out_dim] == 0) {
                    out_shape[out_dim] = dim_size;
                } else if (dim_size != out_shape[out_dim] && dim_size != 1) {
                    return -1;
                }
            }
            max_ndim = (ndim > max_ndim) ? ndim : max_ndim;
        }
    }
    
    for (int32_t d = 0; d < max_ndim; d++) {
        int32_t out_dim = d + (NC_MAX_DIMS - max_ndim);
        if (out_dim >= NC_MAX_DIMS) continue;
        if (out_shape[out_dim] == 0) out_shape[out_dim] = 1;
    }
    
    return max_ndim;
}

bool nc_can_broadcast(NCArray *a, NCArray *b) {
    if (!a || !b) return false;
    if (a->ndim > NC_MAX_DIMS || b->ndim > NC_MAX_DIMS) return false;
    
    int32_t max_ndim = (a->ndim > b->ndim) ? a->ndim : b->ndim;
    int64_t a_shape[NC_MAX_DIMS] = {0};
    int64_t b_shape[NC_MAX_DIMS] = {0};
    
    int32_t a_offset = max_ndim - a->ndim;
    int32_t b_offset = max_ndim - b->ndim;
    
    for (int32_t i = 0; i < a->ndim; i++) {
        a_shape[i + a_offset] = a->shape[i];
    }
    for (int32_t i = 0; i < b->ndim; i++) {
        b_shape[i + b_offset] = b->shape[i];
    }
    
    for (int32_t i = 0; i < max_ndim; i++) {
        int64_t a_dim = a_shape[i] ? a_shape[i] : 1;
        int64_t b_dim = b_shape[i] ? b_shape[i] : 1;
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            return false;
        }
    }
    return true;
}

NCArray *nc_broadcast_to(NCArray *arr, int32_t ndim, const int64_t *shape) {
    if (!arr || !shape) return NULL;
    
    if (ndim < arr->ndim) return NULL;
    
    int64_t result_shape[NC_MAX_DIMS] = {0};
    int32_t offset = ndim - arr->ndim;
    for (int32_t i = 0; i < arr->ndim; i++) {
        result_shape[i + offset] = arr->shape[i];
    }
    
    for (int32_t i = 0; i < ndim; i++) {
        if (result_shape[i] == 0) result_shape[i] = shape[i];
        else if (shape[i] != result_shape[i]) {
            if (result_shape[i] == 1) result_shape[i] = shape[i];
            else return NULL;
        }
    }
    
    NCArray *result = nc_empty(ndim, result_shape, arr->dtype);
    if (!result) return NULL;
    
    size_t total = nc_size(result);
    for (size_t i = 0; i < total; i++) {
        double val = nc_get_value_as_double(arr, i);
        nc_set_value_from_double(result, i, val);
    }
    
    return result;
}

#define NC_BINARY_OP(name, op) \
NCArray *name(NCArray *a, NCArray *b) { \
    if (!a || !b) return NULL; \
    \
    int32_t max_ndim = (a->ndim > b->ndim) ? a->ndim : b->ndim; \
    int64_t result_shape[NC_MAX_DIMS] = {1}; \
    int32_t result_ndim = nc_broadcast_shapes(max_ndim, (const int64_t**)&(int64_t[]){1,1}, 2, result_shape); \
    \
    NCArray *result = nc_empty(result_ndim, result_shape, NC_FLOAT64); \
    if (!result) return NULL; \
    \
    size_t total = nc_size(result); \
    for (size_t i = 0; i < total; i++) { \
        double av = nc_get_value_as_double(a, i); \
        double bv = nc_get_value_as_double(b, i); \
        ((double*)result->data)[i] = av op bv; \
    } \
    return result; \
}

NC_BINARY_OP(nc_add, +)
NC_BINARY_OP(nc_subtract, -)
NC_BINARY_OP(nc_multiply, *)
NC_BINARY_OP(nc_divide, /)

NCArray *nc_power(NCArray *a, NCArray *b) {
    if (!a || !b) return NULL;
    
    int32_t max_ndim = (a->ndim > b->ndim) ? a->ndim : b->ndim;
    int64_t result_shape[NC_MAX_DIMS] = {1};
    nc_broadcast_shapes(max_ndim, (const int64_t**)&(int64_t[]){1,1}, 2, result_shape);
    
    NCArray *result = nc_empty(max_ndim, result_shape, NC_FLOAT64);
    if (!result) return NULL;
    
    size_t total = nc_size(result);
    for (size_t i = 0; i < total; i++) {
        double av = nc_get_value_as_double(a, i);
        double bv = nc_get_value_as_double(b, i);
        ((double*)result->data)[i] = pow(av, bv);
    }
    return result;
}

NCArray *nc_mod(NCArray *a, NCArray *b) {
    if (!a || !b) return NULL;
    
    int32_t max_ndim = (a->ndim > b->ndim) ? a->ndim : b->ndim;
    int64_t result_shape[NC_MAX_DIMS] = {1};
    nc_broadcast_shapes(max_ndim, (const int64_t**)&(int64_t[]){1,1}, 2, result_shape);
    
    NCArray *result = nc_empty(max_ndim, result_shape, NC_FLOAT64);
    if (!result) return NULL;
    
    size_t total = nc_size(result);
    for (size_t i = 0; i < total; i++) {
        double av = nc_get_value_as_double(a, i);
        double bv = nc_get_value_as_double(b, i);
        ((double*)result->data)[i] = fmod(av, bv);
    }
    return result;
}
NC_BINARY_OP(nc_less, <)
NC_BINARY_OP(nc_greater, >)
NC_BINARY_OP(nc_less_equal, <=)
NC_BINARY_OP(nc_greater_equal, >=)

NCArray *nc_equal(NCArray *a, NCArray *b) {
    if (!a || !b) return NULL;
    
    int32_t max_ndim = (a->ndim > b->ndim) ? a->ndim : b->ndim;
    int64_t result_shape[NC_MAX_DIMS] = {1};
    nc_broadcast_shapes(max_ndim, (const int64_t**)&(int64_t[]){1,1}, 2, result_shape);
    
    NCArray *result = nc_empty(max_ndim, result_shape, NC_BOOL);
    if (!result) return NULL;
    
    size_t total = nc_size(result);
    bool *r_data = (bool *)result->data;
    for (size_t i = 0; i < total; i++) {
        double av = nc_get_value_as_double(a, i);
        double bv = nc_get_value_as_double(b, i);
        r_data[i] = (av == bv);
    }
    return result;
}

NCArray *nc_not_equal(NCArray *a, NCArray *b) {
    if (!a || !b) return NULL;
    NCArray *result = nc_equal(a, b);
    if (!result) return NULL;
    bool *data = (bool*)result->data;
    size_t total = nc_size(result);
    for (size_t i = 0; i < total; i++) {
        data[i] = !data[i];
    }
    return result;
}

NCArray *nc_floor_divide(NCArray *a, NCArray *b) {
    if (!a || !b) return NULL;
    
    int32_t max_ndim = (a->ndim > b->ndim) ? a->ndim : b->ndim;
    int64_t result_shape[NC_MAX_DIMS] = {1};
    nc_broadcast_shapes(max_ndim, (const int64_t**)&(int64_t[]){1,1}, 2, result_shape);
    
    NCArray *result = nc_empty(max_ndim, result_shape, NC_FLOAT64);
    if (!result) return NULL;
    
    size_t total = nc_size(result);
    for (size_t i = 0; i < total; i++) {
        double av = nc_get_value_as_double(a, i);
        double bv = nc_get_value_as_double(b, i);
        ((double*)result->data)[i] = floor(av / bv);
    }
    return result;
}

NCArray *nc_true_divide(NCArray *a, NCArray *b) {
    if (!a || !b) return NULL;
    
    int32_t max_ndim = (a->ndim > b->ndim) ? a->ndim : b->ndim;
    int64_t result_shape[NC_MAX_DIMS] = {1};
    nc_broadcast_shapes(max_ndim, (const int64_t**)&(int64_t[]){1,1}, 2, result_shape);
    
    NCArray *result = nc_empty(max_ndim, result_shape, NC_FLOAT64);
    if (!result) return NULL;
    
    size_t total = nc_size(result);
    for (size_t i = 0; i < total; i++) {
        double av = nc_get_value_as_double(a, i);
        double bv = nc_get_value_as_double(b, i);
        ((double*)result->data)[i] = bv != 0 ? av / bv : NAN;
    }
    return result;
}

#define NC_LOGICAL_OP(name, op) \
NCArray *name(NCArray *a, NCArray *b) { \
    if (!a || !b) return NULL; \
    \
    int32_t max_ndim = (a->ndim > b->ndim) ? a->ndim : b->ndim; \
    int64_t result_shape[NC_MAX_DIMS] = {1}; \
    nc_broadcast_shapes(max_ndim, (const int64_t**)&(int64_t[]){1,1}, 2, result_shape); \
    \
    NCArray *result = nc_empty(max_ndim, result_shape, NC_BOOL); \
    if (!result) return NULL; \
    \
    size_t total = nc_size(result); \
    bool *r_data = (bool *)result->data; \
    for (size_t i = 0; i < total; i++) { \
        bool av = nc_get_value_as_double(a, i) != 0; \
        bool bv = nc_get_value_as_double(b, i) != 0; \
        r_data[i] = av op bv; \
    } \
    return result; \
}

NC_LOGICAL_OP(nc_logical_and, &&)
NC_LOGICAL_OP(nc_logical_or, ||)
NC_LOGICAL_OP(nc_logical_xor, !=)

NCArray *nc_logical_not(NCArray *a) {
    if (!a) return NULL;
    NCArray *result = nc_empty(a->ndim, a->shape, NC_BOOL);
    if (!result) return NULL;
    
    size_t total = nc_size(a);
    bool *r_data = (bool *)result->data;
    for (size_t i = 0; i < total; i++) {
        r_data[i] = !nc_get_value_as_double(a, i);
    }
    return result;
}

#define NC_BITWISE_OP(name, op) \
NCArray *name(NCArray *a, NCArray *b) { \
    if (!a || !b) return NULL; \
    \
    int32_t max_ndim = (a->ndim > b->ndim) ? a->ndim : b->ndim; \
    int64_t result_shape[NC_MAX_DIMS] = {1}; \
    nc_broadcast_shapes(max_ndim, (const int64_t**)&(int64_t[]){1,1}, 2, result_shape); \
    \
    NCArray *result = nc_empty(max_ndim, result_shape, a->dtype); \
    if (!result) return NULL; \
    \
    size_t total = nc_size(result); \
    for (size_t i = 0; i < total; i++) { \
        int64_t av = (int64_t)nc_get_value_as_double(a, i); \
        int64_t bv = (int64_t)nc_get_value_as_double(b, i); \
        nc_set_value_from_double(result, i, (double)(av op bv)); \
    } \
    return result; \
}

NC_BITWISE_OP(nc_bitwise_and, &)
NC_BITWISE_OP(nc_bitwise_or, |)
NC_BITWISE_OP(nc_bitwise_xor, ^)

NCArray *nc_bitwise_not(NCArray *a) {
    if (!a) return NULL;
    NCArray *result = nc_empty(a->ndim, a->shape, a->dtype);
    if (!result) return NULL;
    
    size_t total = nc_size(a);
    for (size_t i = 0; i < total; i++) {
        int64_t av = (int64_t)nc_get_value_as_double(a, i);
        nc_set_value_from_double(result, i, (double)(~av));
    }
    return result;
}

NC_BITWISE_OP(nc_left_shift, <<)
NC_BITWISE_OP(nc_right_shift, >>)

#define NC_UNARY_MATH_OP(name, func) \
NCArray *name(NCArray *arr) { \
    if (!arr) return NULL; \
    NCArray *result = nc_empty(arr->ndim, arr->shape, NC_FLOAT64); \
    if (!result) return NULL; \
    \
    size_t total = nc_size(arr); \
    double *r_data = (double *)result->data; \
    for (size_t i = 0; i < total; i++) { \
        double v = nc_get_value_as_double(arr, i); \
        r_data[i] = func(v); \
    } \
    return result; \
}

NC_UNARY_MATH_OP(nc_abs, fabs)
NC_UNARY_MATH_OP(nc_fabs, fabs)

NCArray *nc_sign(NCArray *arr) {
    if (!arr) return NULL;
    NCArray *result = nc_empty(arr->ndim, arr->shape, arr->dtype);
    if (!result) return NULL;
    
    size_t total = nc_size(arr);
    for (size_t i = 0; i < total; i++) {
        double v = nc_get_value_as_double(arr, i);
        nc_set_value_from_double(result, i, (v > 0) - (v < 0));
    }
    return result;
}

NC_UNARY_MATH_OP(nc_floor, floor)
NC_UNARY_MATH_OP(nc_ceil, ceil)
NC_UNARY_MATH_OP(nc_round, round)
NC_UNARY_MATH_OP(nc_trunc, trunc)

NC_UNARY_MATH_OP(nc_exp, exp)
NC_UNARY_MATH_OP(nc_expm1, expm1)
NC_UNARY_MATH_OP(nc_exp2, exp2)
NC_UNARY_MATH_OP(nc_log, log)
NC_UNARY_MATH_OP(nc_log1p, log1p)
NC_UNARY_MATH_OP(nc_log2, log2)
NC_UNARY_MATH_OP(nc_log10, log10)

NC_UNARY_MATH_OP(nc_sqrt, sqrt)
NC_UNARY_MATH_OP(nc_cbrt, cbrt)

NCArray *nc_square(NCArray *arr) {
    if (!arr) return NULL;
    NCArray *result = nc_empty(arr->ndim, arr->shape, NC_FLOAT64);
    if (!result) return NULL;
    
    size_t total = nc_size(arr);
    double *r_data = (double *)result->data;
    for (size_t i = 0; i < total; i++) {
        double v = nc_get_value_as_double(arr, i);
        r_data[i] = v * v;
    }
    return result;
}

NC_UNARY_MATH_OP(nc_sin, sin)
NC_UNARY_MATH_OP(nc_cos, cos)
NC_UNARY_MATH_OP(nc_tan, tan)
NC_UNARY_MATH_OP(nc_arcsin, asin)
NC_UNARY_MATH_OP(nc_arccos, acos)
NC_UNARY_MATH_OP(nc_arctan, atan)
NC_UNARY_MATH_OP(nc_sinh, sinh)
NC_UNARY_MATH_OP(nc_cosh, cosh)
NC_UNARY_MATH_OP(nc_tanh, tanh)
NC_UNARY_MATH_OP(nc_arcsinh, asinh)
NC_UNARY_MATH_OP(nc_arccosh, acosh)
NC_UNARY_MATH_OP(nc_arctanh, atanh)

NCArray *nc_deg2rad(NCArray *arr) {
    if (!arr) return NULL;
    NCArray *result = nc_empty(arr->ndim, arr->shape, NC_FLOAT64);
    if (!result) return NULL;
    
    size_t total = nc_size(arr);
    double *r_data = (double *)result->data;
    for (size_t i = 0; i < total; i++) {
        double v = nc_get_value_as_double(arr, i);
        r_data[i] = v * M_PI / 180.0;
    }
    return result;
}

NCArray *nc_rad2deg(NCArray *arr) {
    if (!arr) return NULL;
    NCArray *result = nc_empty(arr->ndim, arr->shape, NC_FLOAT64);
    if (!result) return NULL;
    
    size_t total = nc_size(arr);
    double *r_data = (double *)result->data;
    for (size_t i = 0; i < total; i++) {
        double v = nc_get_value_as_double(arr, i);
        r_data[i] = v * 180.0 / M_PI;
    }
    return result;
}

NCArray *nc_negate(NCArray *arr) {
    if (!arr) return NULL;
    NCArray *result = nc_empty(arr->ndim, arr->shape, arr->dtype);
    if (!result) return NULL;
    
    size_t total = nc_size(arr);
    for (size_t i = 0; i < total; i++) {
        double v = nc_get_value_as_double(arr, i);
        nc_set_value_from_double(result, i, -v);
    }
    return result;
}

NCArray *nc_invert(NCArray *arr) {
    return nc_bitwise_not(arr);
}

NCArray *nc_dot(NCArray *a, NCArray *b) {
    if (!a || !b) return NULL;
    
    if (a->ndim == 1 && b->ndim == 1) {
        if (a->shape[0] != b->shape[0]) return NULL;
        NCArray *result = nc_zeros(0, NULL, NC_FLOAT64);
        if (!result) return NULL;
        
        double sum = 0;
        for (int64_t i = 0; i < a->shape[0]; i++) {
            sum += nc_get_value_as_double(a, i) * nc_get_value_as_double(b, i);
        }
        ((double*)result->data)[0] = sum;
        return result;
    }
    
    return nc_matmul(a, b);
}

NCArray *nc_matmul(NCArray *a, NCArray *b) {
    if (!a || !b) return NULL;
    
    if (a->ndim < 1 || b->ndim < 1) return NULL;
    
    int64_t a_rows = a->shape[a->ndim - 2];
    int64_t a_cols = a->shape[a->ndim - 1];
    int64_t b_rows = b->shape[b->ndim - 2];
    int64_t b_cols = b->shape[b->ndim - 1];
    
    if (a_cols != b_rows) return NULL;
    
    int32_t batch_a = 1, batch_b = 1;
    for (int32_t i = 0; i < a->ndim - 2; i++) batch_a *= a->shape[i];
    for (int32_t i = 0; i < b->ndim - 2; i++) batch_b *= b->shape[i];
    
    int32_t result_ndim = (a->ndim > b->ndim) ? a->ndim : b->ndim;
    int64_t result_shape[NC_MAX_DIMS];
    for (int32_t i = 0; i < result_ndim - 2; i++) {
        int64_t a_dim = (i < a->ndim - 2) ? a->shape[i] : 1;
        int64_t b_dim = (i < b->ndim - 2) ? b->shape[i] : 1;
        result_shape[i] = (a_dim > b_dim) ? a_dim : b_dim;
    }
    result_shape[result_ndim - 2] = a_rows;
    result_shape[result_ndim - 1] = b_cols;
    
    NCArray *result = nc_zeros(result_ndim, result_shape, NC_FLOAT64);
    if (!result) return NULL;
    
    int64_t total_batches = batch_a * batch_b;
    for (int64_t batch = 0; batch < total_batches; batch++) {
        int64_t a_batch = batch / batch_b;
        int64_t b_batch = batch % batch_b;
        
        for (int64_t i = 0; i < a_rows; i++) {
            for (int64_t j = 0; j < b_cols; j++) {
                double sum = 0;
                for (int64_t k = 0; k < a_cols; k++) {
                    int64_t a_idx = a_batch * (a_rows * a_cols) + i * a_cols + k;
                    int64_t b_idx = b_batch * (b_rows * b_cols) + k * b_cols + j;
                    sum += nc_get_value_as_double(a, a_idx) * nc_get_value_as_double(b, b_idx);
                }
                int64_t r_idx = batch * (a_rows * b_cols) + i * b_cols + j;
                ((double*)result->data)[r_idx] = sum;
            }
        }
    }
    
    return result;
}

NCArray *nc_inner(NCArray *a, NCArray *b) {
    if (!a || !b) return NULL;
    
    if (a->ndim != b->ndim) return NULL;
    
    NCArray *result = nc_zeros(0, NULL, NC_FLOAT64);
    if (!result) return NULL;
    
    double sum = 0;
    size_t total = nc_size(a);
    for (size_t i = 0; i < total; i++) {
        sum += nc_get_value_as_double(a, i) * nc_get_value_as_double(b, i);
    }
    ((double*)result->data)[0] = sum;
    return result;
}

NCArray *nc_outer(NCArray *a, NCArray *b) {
    if (!a || !b) return NULL;
    if (a->ndim != 1 || b->ndim != 1) return NULL;
    
    int64_t shape[2] = {a->shape[0], b->shape[0]};
    NCArray *result = nc_empty(2, shape, NC_FLOAT64);
    if (!result) return NULL;
    
    double *r_data = (double *)result->data;
    for (int64_t i = 0; i < a->shape[0]; i++) {
        double av = nc_get_value_as_double(a, i);
        for (int64_t j = 0; j < b->shape[0]; j++) {
            r_data[i * b->shape[0] + j] = av * nc_get_value_as_double(b, j);
        }
    }
    return result;
}

NCArray *nc_cross(NCArray *a, NCArray *b, int32_t axis) {
    if (!a || !b) return NULL;
    if (a->ndim != 1 || b->ndim != 1) return NULL;
    if (a->shape[0] != 3 || b->shape[0] != 3) return NULL;
    
    NCArray *result = nc_empty(1, (int64_t[]){3}, NC_FLOAT64);
    if (!result) return NULL;
    
    double *r_data = (double *)result->data;
    double a0 = nc_get_value_as_double(a, 0);
    double a1 = nc_get_value_as_double(a, 1);
    double a2 = nc_get_value_as_double(a, 2);
    double b0 = nc_get_value_as_double(b, 0);
    double b1 = nc_get_value_as_double(b, 1);
    double b2 = nc_get_value_as_double(b, 2);
    
    r_data[0] = a1 * b2 - a2 * b1;
    r_data[1] = a2 * b0 - a0 * b2;
    r_data[2] = a0 * b1 - a1 * b0;
    
    return result;
}

NCArray *nc_trace(NCArray *arr, int32_t offset, int32_t axis1, int32_t axis2) {
    if (!arr || arr->ndim < 2) return NULL;
    
    int64_t n = arr->shape[axis1];
    int64_t m = arr->shape[axis2];
    int64_t k = (offset >= 0) ? offset : -offset;
    
    int64_t diag_len = (n < m - k) ? n : m - k;
    if (diag_len < 0) diag_len = 0;
    
    NCArray *result = nc_zeros(1, (int64_t[]){diag_len}, arr->dtype);
    if (!result) return NULL;
    
    char *src = (char *)arr->data;
    char *dst = (char *)result->data;
    size_t esize = arr->itemsize;
    
    int64_t arr_stride_1 = arr->strides[axis1];
    int64_t arr_stride_2 = arr->strides[axis2];
    
    for (int64_t i = 0; i < diag_len; i++) {
        int64_t row = (offset >= 0) ? i : i - offset;
        int64_t col = (offset >= 0) ? i + offset : i;
        int64_t idx = row * arr_stride_1 + col * arr_stride_2;
        memcpy(dst + i * esize, src + idx * esize, esize);
    }
    
    return result;
}

NCArray *nc_sum(NCArray *arr, const int32_t *axis, int32_t naxis) {
    if (!arr) return NULL;
    
    if (naxis == 0 || axis == NULL) {
        NCArray *result = nc_zeros(0, NULL, NC_FLOAT64);
        if (!result) return NULL;
        
        double sum = 0;
        size_t total = nc_size(arr);
        for (size_t i = 0; i < total; i++) {
            sum += nc_get_value_as_double(arr, i);
        }
        ((double*)result->data)[0] = sum;
        return result;
    }
    
    return NULL;
}

NCArray *nc_prod(NCArray *arr, const int32_t *axis, int32_t naxis) {
    if (!arr) return NULL;
    
    NCArray *result = nc_zeros(0, NULL, NC_FLOAT64);
    if (!result) return NULL;
    
    double prod = 1;
    size_t total = nc_size(arr);
    for (size_t i = 0; i < total; i++) {
        prod *= nc_get_value_as_double(arr, i);
    }
    ((double*)result->data)[0] = prod;
    return result;
}

NCArray *nc_mean(NCArray *arr, const int32_t *axis, int32_t naxis) {
    if (!arr) return NULL;
    NCArray *sum = nc_sum(arr, axis, naxis);
    if (!sum) return NULL;
    
    size_t total = nc_size(arr);
    double *data = (double*)sum->data;
    data[0] /= total;
    
    return sum;
}

NCArray *nc_var(NCArray *arr, const int32_t *axis, int32_t naxis) {
    if (!arr) return NULL;
    
    NCArray *mean = nc_mean(arr, axis, naxis);
    if (!mean) return NULL;
    
    NCArray *result = nc_zeros(0, NULL, NC_FLOAT64);
    if (!result) return NULL;
    
    double mu = ((double*)mean->data)[0];
    double var = 0;
    size_t total = nc_size(arr);
    
    for (size_t i = 0; i < total; i++) {
        double diff = nc_get_value_as_double(arr, i) - mu;
        var += diff * diff;
    }
    
    ((double*)result->data)[0] = var / total;
    nc_release(mean);
    return result;
}

NCArray *nc_std(NCArray *arr, const int32_t *axis, int32_t naxis) {
    if (!arr) return NULL;
    NCArray *v = nc_var(arr, axis, naxis);
    if (!v) return NULL;
    
    double *data = (double*)v->data;
    data[0] = sqrt(data[0]);
    return v;
}

NCArray *nc_min(NCArray *arr, const int32_t *axis, int32_t naxis) {
    if (!arr) return NULL;
    
    NCArray *result = nc_zeros(0, NULL, NC_FLOAT64);
    if (!result) return NULL;
    
    double min_val = INFINITY;
    size_t total = nc_size(arr);
    
    for (size_t i = 0; i < total; i++) {
        double v = nc_get_value_as_double(arr, i);
        if (v < min_val) min_val = v;
    }
    ((double*)result->data)[0] = min_val;
    return result;
}

NCArray *nc_max(NCArray *arr, const int32_t *axis, int32_t naxis) {
    if (!arr) return NULL;
    
    NCArray *result = nc_zeros(0, NULL, NC_FLOAT64);
    if (!result) return NULL;
    
    double max_val = -INFINITY;
    size_t total = nc_size(arr);
    
    for (size_t i = 0; i < total; i++) {
        double v = nc_get_value_as_double(arr, i);
        if (v > max_val) max_val = v;
    }
    ((double*)result->data)[0] = max_val;
    return result;
}

NCArray *nc_argmin(NCArray *arr, int32_t axis) {
    if (!arr) return NULL;
    
    NCArray *result = nc_zeros(1, (int64_t[]){nc_size(arr)}, NC_INT64);
    if (!result) return NULL;
    
    int64_t *r_data = (int64_t*)result->data;
    size_t total = nc_size(arr);
    
    int64_t min_idx = 0;
    double min_val = INFINITY;
    
    for (size_t i = 0; i < total; i++) {
        double v = nc_get_value_as_double(arr, i);
        if (v < min_val) {
            min_val = v;
            min_idx = i;
        }
    }
    r_data[0] = min_idx;
    return result;
}

NCArray *nc_argmax(NCArray *arr, int32_t axis) {
    if (!arr) return NULL;
    
    NCArray *result = nc_zeros(1, (int64_t[]){nc_size(arr)}, NC_INT64);
    if (!result) return NULL;
    
    int64_t *r_data = (int64_t*)result->data;
    size_t total = nc_size(arr);
    
    int64_t max_idx = 0;
    double max_val = -INFINITY;
    
    for (size_t i = 0; i < total; i++) {
        double v = nc_get_value_as_double(arr, i);
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }
    r_data[0] = max_idx;
    return result;
}

NCArray *nc_all(NCArray *arr, const int32_t *axis, int32_t naxis) {
    if (!arr) return NULL;
    
    NCArray *result = nc_zeros(0, NULL, NC_BOOL);
    if (!result) return NULL;
    
    size_t total = nc_size(arr);
    bool all_true = true;
    
    for (size_t i = 0; i < total; i++) {
        if (!nc_get_value_as_double(arr, i)) {
            all_true = false;
            break;
        }
    }
    ((bool*)result->data)[0] = all_true;
    return result;
}

NCArray *nc_any(NCArray *arr, const int32_t *axis, int32_t naxis) {
    if (!arr) return NULL;
    
    NCArray *result = nc_zeros(0, NULL, NC_BOOL);
    if (!result) return NULL;
    
    size_t total = nc_size(arr);
    bool any_true = false;
    
    for (size_t i = 0; i < total; i++) {
        if (nc_get_value_as_double(arr, i)) {
            any_true = true;
            break;
        }
    }
    ((bool*)result->data)[0] = any_true;
    return result;
}

NCArray *nc_cumsum(NCArray *arr, int32_t axis) {
    if (!arr) return NULL;
    return nc_copy(arr);
}

NCArray *nc_cumprod(NCArray *arr, int32_t axis) {
    if (!arr) return NULL;
    return nc_copy(arr);
}

NCArray *nc_nancumsum(NCArray *arr, int32_t axis) {
    return nc_cumsum(arr, axis);
}

NCArray *nc_nancumprod(NCArray *arr, int32_t axis) {
    return nc_cumprod(arr, axis);
}

int64_t nc_count_nonzero(NCArray *arr) {
    if (!arr) return 0;
    
    size_t total = nc_size(arr);
    int64_t count = 0;
    
    for (size_t i = 0; i < total; i++) {
        if (nc_get_value_as_double(arr, i) != 0) count++;
    }
    return count;
}

bool nc_isnan(NCArray *arr) {
    if (!arr) return false;
    size_t total = nc_size(arr);
    
    for (size_t i = 0; i < total; i++) {
        double v = nc_get_value_as_double(arr, i);
        if (isnan(v)) return true;
    }
    return false;
}

bool nc_isinf(NCArray *arr) {
    if (!arr) return false;
    size_t total = nc_size(arr);
    
    for (size_t i = 0; i < total; i++) {
        double v = nc_get_value_as_double(arr, i);
        if (isinf(v)) return true;
    }
    return false;
}

bool nc_isfinite(NCArray *arr) {
    if (!arr) return false;
    size_t total = nc_size(arr);
    
    for (size_t i = 0; i < total; i++) {
        double v = nc_get_value_as_double(arr, i);
        if (!isfinite(v)) return false;
    }
    return true;
}

bool nc_isneginf(NCArray *arr) {
    if (!arr) return false;
    size_t total = nc_size(arr);
    
    for (size_t i = 0; i < total; i++) {
        double v = nc_get_value_as_double(arr, i);
        if (v == -INFINITY) return true;
    }
    return false;
}

bool nc_isposinf(NCArray *arr) {
    if (!arr) return false;
    size_t total = nc_size(arr);
    
    for (size_t i = 0; i < total; i++) {
        double v = nc_get_value_as_double(arr, i);
        if (v == INFINITY) return true;
    }
    return false;
}

double nc_nanmin(NCArray *arr) { return NAN; }
double nc_nanmax(NCArray *arr) { return NAN; }
double nc_nanmean(NCArray *arr) { return NAN; }
double nc_nanstd(NCArray *arr) { return NAN; }
double nc_nanvar(NCArray *arr) { return NAN; }
double nc_nansum(NCArray *arr) { return NAN; }
double nc_nanprod(NCArray *arr) { return NAN; }

NCArray *nc_where(NCArray *condition, NCArray *x, NCArray *y) {
    return ncselect(condition, x, y);
}

NCArray *nc_nan_to_num(NCArray *arr) {
    if (!arr) return NULL;
    return nc_copy(arr);
}

NCArray *nc_interp(NCArray *x, NCArray *xp, NCArray *fp, double left, double right) {
    if (!x || !xp || !fp) return NULL;
    return nc_copy(x);
}

NCArray *nc_concatenate(NCArray **arrays, int32_t n, int32_t axis) {
    if (!arrays || n <= 0) return NULL;
    
    if (axis < 0) axis = arrays[0]->ndim - 1;
    if (axis >= arrays[0]->ndim) return NULL;
    
    for (int32_t i = 1; i < n; i++) {
        if (arrays[i]->ndim != arrays[0]->ndim) return NULL;
        for (int32_t j = 0; j < arrays[0]->ndim; j++) {
            if (j != axis && arrays[i]->shape[j] != arrays[0]->shape[j]) return NULL;
        }
    }
    
    int64_t total_size = 0;
    for (int32_t i = 0; i < n; i++) {
        total_size += arrays[i]->shape[axis];
    }
    
    int32_t result_ndim = arrays[0]->ndim;
    int64_t result_shape[NC_MAX_DIMS];
    for (int32_t i = 0; i < result_ndim; i++) {
        result_shape[i] = (i == axis) ? total_size : arrays[0]->shape[i];
    }
    
    NCArray *result = nc_empty(result_ndim, result_shape, arrays[0]->dtype);
    if (!result) return NULL;
    
    char *dst = (char *)result->data;
    size_t esize = result->itemsize;
    int64_t offset = 0;
    
    for (int32_t i = 0; i < n; i++) {
        size_t arr_size = nc_size(arrays[i]);
        memcpy(dst + offset * esize, arrays[i]->data, arr_size * esize);
        offset += arrays[i]->shape[axis];
    }
    
    return result;
}

NCArray *nc_stack(NCArray **arrays, int32_t n, int32_t axis) {
    if (!arrays || n <= 0) return NULL;
    
    if (axis < 0) axis = arrays[0]->ndim;
    
    int32_t result_ndim = arrays[0]->ndim + 1;
    int64_t result_shape[NC_MAX_DIMS];
    
    for (int32_t i = 0; i < axis; i++) {
        result_shape[i] = arrays[0]->shape[i];
    }
    result_shape[axis] = n;
    for (int32_t i = axis + 1; i < result_ndim; i++) {
        result_shape[i] = arrays[0]->shape[i - 1];
    }
    
    NCArray *result = nc_empty(result_ndim, result_shape, arrays[0]->dtype);
    if (!result) return NULL;
    
    size_t esize = result->itemsize;
    size_t single_size = nc_size(arrays[0]);
    
    for (int32_t i = 0; i < n; i++) {
        int64_t offset = 0;
        int64_t temp = i;
        for (int32_t d = result_ndim - 1; d >= 0; d--) {
            if (d == axis) {
                offset += temp % result_shape[d];
                temp /= result_shape[d];
            } else {
                int32_t arr_dim = (d > axis) ? d - 1 : d;
                int64_t idx_in_dim = temp % arrays[0]->shape[arr_dim];
                offset += idx_in_dim * arrays[0]->strides[arr_dim];
                temp /= arrays[0]->shape[arr_dim];
            }
        }
        memcpy((char*)result->data + offset * esize, arrays[i]->data, single_size * esize);
    }
    
    return result;
}

NCArray *nc_vstack(NCArray **arrays, int32_t n) {
    if (!arrays || n <= 0) return NULL;
    
    if (arrays[0]->ndim == 1) {
        NCArray *arr2d[100];
        for (int32_t i = 0; i < n; i++) {
            arr2d[i] = ncexpand_dims(arrays[i], 0);
        }
        NCArray *result = nc_stack(arr2d, n, 0);
        for (int32_t i = 0; i < n; i++) {
            nc_release(arr2d[i]);
        }
        return result;
    }
    return nc_stack(arrays, n, 0);
}

NCArray *nc_hstack(NCArray **arrays, int32_t n) {
    if (!arrays || n <= 0) return NULL;
    
    if (arrays[0]->ndim == 1) {
        return nc_concatenate(arrays, n, 0);
    }
    return nc_concatenate(arrays, n, 1);
}

NCArray *nc_dstack(NCArray **arrays, int32_t n) {
    if (!arrays || n <= 0) return NULL;
    
    if (arrays[0]->ndim == 1) {
        NCArray *arr2d[100];
        for (int32_t i = 0; i < n; i++) {
            arr2d[i] = ncexpand_dims(arrays[i], 0);
            arr2d[i] = ncexpand_dims(arr2d[i], 0);
        }
        NCArray *result = nc_stack(arr2d, n, 0);
        for (int32_t i = 0; i < n; i++) {
            nc_release(arr2d[i]);
        }
        return result;
    } else if (arrays[0]->ndim == 2) {
        return nc_stack(arrays, n, 2);
    }
    return NULL;
}

NCArray *nc_split(NCArray *arr, int32_t n_sections, int32_t axis) {
    if (!arr || n_sections <= 0) return NULL;
    
    if (arr->shape[axis] % n_sections != 0) return NULL;
    
    int64_t section_size = arr->shape[axis] / n_sections;
    NCArray *result = nc_empty(1, (int64_t[]){n_sections}, NC_INT64);
    return result;
}

NCArray **nc_array_split(NCArray *arr, int32_t n_sections, int32_t axis) {
    if (!arr || n_sections <= 0) return NULL;
    
    NCArray **results = (NCArray**)malloc(n_sections * sizeof(NCArray*));
    if (!results) return NULL;
    
    int64_t section_size = arr->shape[axis] / n_sections;
    
    for (int32_t i = 0; i < n_sections; i++) {
        int64_t starts[NC_MAX_DIMS] = {0};
        int64_t stops[NC_MAX_DIMS];
        int64_t steps[NC_MAX_DIMS];
        for (int32_t d = 0; d < arr->ndim; d++) {
            stops[d] = arr->shape[d];
            steps[d] = 1;
        }
        starts[axis] = i * section_size;
        stops[axis] = (i + 1) * section_size;
        results[i] = nc_slice(arr, starts, stops, steps, arr->ndim);
    }
    
    return results;
}

NCArray *nc_tile(NCArray *arr, const int32_t *reps, int32_t n_reps) {
    if (!arr || !reps || n_reps <= 0) return NULL;
    
    int32_t result_ndim = arr->ndim + n_reps;
    if (result_ndim > NC_MAX_DIMS) return NULL;
    
    int64_t result_shape[NC_MAX_DIMS];
    for (int32_t i = 0; i < n_reps; i++) {
        result_shape[i] = reps[i];
    }
    for (int32_t i = 0; i < arr->ndim; i++) {
        result_shape[n_reps + i] = arr->shape[i];
    }
    
    NCArray *result = nc_empty(result_ndim, result_shape, arr->dtype);
    if (!result) return NULL;
    
    size_t total = nc_size(result);
    for (size_t i = 0; i < total; i++) {
        double v = nc_get_value_as_double(arr, i);
        nc_set_value_from_double(result, i, v);
    }
    
    return result;
}

NCArray *nc_repeat(NCArray *arr, int64_t repeats, int32_t axis) {
    if (!arr || repeats < 0) return NULL;
    
    size_t arr_size = nc_size(arr);
    int64_t result_shape[1] = {arr_size * repeats};
    
    NCArray *result = nc_empty(1, result_shape, arr->dtype);
    if (!result) return NULL;
    
    char *src = (char *)arr->data;
    char *dst = (char *)result->data;
    size_t esize = arr->itemsize;
    
    for (size_t i = 0; i < arr_size; i++) {
        for (int64_t j = 0; j < repeats; j++) {
            memcpy(dst + (i * repeats + j) * esize, src + i * esize, esize);
        }
    }
    
    return result;
}

NCArray *nc_pad(NCArray *arr, const int64_t *pad_width, NCArray *constant_values) {
    if (!arr || !pad_width) return NULL;
    return nc_copy(arr);
}

NCArray *nc_extract(NCArray *condition, NCArray *arr) {
    if (!condition || !arr) return NULL;
    return nc_copy(arr);
}

void *nc_getitem(NCArray *arr, const int64_t *indices, int32_t n_indices) {
    if (!arr || !indices) return NULL;
    return NULL;
}

NCStatus nc_setitem(NCArray *arr, const int64_t *indices, int32_t n_indices, void *value) {
    if (!arr || !indices || !value) return NC_ERROR;
    return NC_OK;
}

NCArray *nc_slice(NCArray *arr, const int64_t *starts, const int64_t *stops, const int64_t *steps, int32_t n_slices) {
    if (!arr || !starts || !stops || !steps) return NULL;
    return nc_copy(arr);
}

NCView *nc_view(NCArray *arr, const int64_t *starts, const int64_t *stops, const int64_t *steps, int32_t n_slices) {
    if (!arr) return NULL;
    NCView *view = (NCView*)calloc(1, sizeof(NCView));
    if (!view) return NULL;
    view->array = arr;
    arr->refcount++;
    return view;
}

NCStatus nc_set_slice(NCArray *arr, NCView *view, NCArray *value) {
    if (!arr || !view || !value) return NC_ERROR;
    return NC_OK;
}

NCArray *nc_fft_fft(NCArray *arr, int32_t n, int32_t axis) { return nc_copy(arr); }
NCArray *nc_fft_ifft(NCArray *arr, int32_t n, int32_t axis) { return nc_copy(arr); }
NCArray *nc_fft_fft2(NCArray *arr, const int32_t *s, const int32_t *axes) { return nc_copy(arr); }
NCArray *nc_fft_ifft2(NCArray *arr, const int32_t *s, const int32_t *axes) { return nc_copy(arr); }
NCArray *nc_fft_fftn(NCArray *arr, const int32_t *s, const int32_t *axes) { return nc_copy(arr); }
NCArray *nc_fft_ifftn(NCArray *arr, const int32_t *s, const int32_t *axes) { return nc_copy(arr); }
NCArray *nc_fft_rfft(NCArray *arr, int32_t n, int32_t axis) { return nc_copy(arr); }
NCArray *nc_fft_irfft(NCArray *arr, int32_t n, int32_t axis) { return nc_copy(arr); }
NCArray *nc_fft_hfft(NCArray *arr, int32_t n, int32_t axis) { return nc_copy(arr); }
NCArray *nc_fft_ihfft(NCArray *arr, int32_t n, int32_t axis) { return nc_copy(arr); }

NCArray *nc_fft_fftfreq(int64_t n, double d, NCDataType dtype) {
    return nc_linspace(0, 1.0/n, n, false, dtype);
}

NCArray *nc_fft_rfftfreq(int64_t n, double d, NCDataType dtype) {
    return nc_linspace(0, 1.0/n, n/2 + 1, false, dtype);
}

NCArray *nc_sort(NCArray *arr, int32_t axis) { return nc_copy(arr); }
NCArray *nc_argsort(NCArray *arr, int32_t axis) { return nc_copy(arr); }
NCArray *nc_searchsorted(NCArray *arr, NCArray *v, const char *side) { return nc_copy(arr); }
NCArray *nc_partition(NCArray *arr, int32_t kth, int32_t axis) { return nc_copy(arr); }

NCArray *nc_random_rand(int32_t ndim, const int64_t *shape) {
    NCArray *arr = nc_empty(ndim, shape, NC_FLOAT64);
    if (!arr) return NULL;
    
    double *data = (double*)arr->data;
    size_t total = nc_size(arr);
    
    for (size_t i = 0; i < total; i++) {
        data[i] = (double)rand() / RAND_MAX;
    }
    return arr;
}

NCArray *nc_random_randn(int32_t ndim, const int64_t *shape, NCDataType dtype) {
    NCArray *arr = nc_empty(ndim, shape, dtype);
    if (!arr) return NULL;
    
    size_t total = nc_size(arr);
    
    for (size_t i = 0; i < total; i++) {
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        double g = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        
        switch (dtype) {
            case NC_FLOAT32: ((float*)arr->data)[i] = (float)g; break;
            case NC_FLOAT64: ((double*)arr->data)[i] = g; break;
            default: ((double*)arr->data)[i] = g; break;
        }
    }
    return arr;
}

NCArray *nc_random_randint(int64_t low, int64_t high, int32_t ndim, const int64_t *shape) {
    NCArray *arr = nc_empty(ndim, shape, NC_INT64);
    if (!arr) return NULL;
    
    int64_t *data = (int64_t*)arr->data;
    size_t total = nc_size(arr);
    
    for (size_t i = 0; i < total; i++) {
        data[i] = low + (int64_t)((high - low) * (double)rand() / RAND_MAX);
    }
    return arr;
}

NCArray *nc_random_random(int32_t ndim, const int64_t *shape, NCDataType dtype) {
    return nc_random_rand(ndim, shape);
}

NCArray *nc_random_choice(int64_t n, int64_t size, bool replace) {
    if (n <= 0 || size <= 0) return NULL;
    
    int64_t shape[1] = {size};
    NCArray *arr = nc_empty(1, shape, NC_INT64);
    if (!arr) return NULL;
    
    int64_t *data = (int64_t*)arr->data;
    for (int64_t i = 0; i < size; i++) {
        data[i] = (int64_t)(n * (double)rand() / RAND_MAX);
    }
    return arr;
}

void nc_random_seed(uint64_t seed) {
    nc_random_state = seed;
    srand((unsigned int)seed);
}

void nc_random_shuffle(NCArray *arr) {
    if (!arr) return;
    size_t total = nc_size(arr);
    for (size_t i = total - 1; i > 0; i--) {
        size_t j = (size_t)((i + 1) * (double)rand() / RAND_MAX);
        double temp = nc_get_value_as_double(arr, i);
        nc_set_value_from_double(arr, i, nc_get_value_as_double(arr, j));
        nc_set_value_from_double(arr, j, temp);
    }
}

void nc_random_permutation(NCArray *arr) {
    nc_random_shuffle(arr);
}

NCArray *nc_linalg_norm(NCArray *arr, const char *ord) {
    if (!arr) return NULL;
    
    NCArray *result = nc_zeros(0, NULL, NC_FLOAT64);
    if (!result) return NULL;
    
    double sum = 0;
    size_t total = nc_size(arr);
    
    for (size_t i = 0; i < total; i++) {
        double v = nc_get_value_as_double(arr, i);
        sum += v * v;
    }
    ((double*)result->data)[0] = sqrt(sum);
    return result;
}

NCArray *nc_linalg_dot(NCArray *a, NCArray *b) { return nc_dot(a, b); }
NCArray *nc_linalg_matmul(NCArray *a, NCArray *b) { return nc_matmul(a, b); }
NCArray *nc_linalg_svd(NCArray *a, bool full_matrices) { return nc_copy(a); }
NCArray *nc_linalg_eig(NCArray *a) { return nc_copy(a); }
NCArray *nc_linalg_eigh(NCArray *a) { return nc_copy(a); }
NCArray *nc_linalg_eigvals(NCArray *a) { return nc_copy(a); }
NCArray *nc_linalg_eigvalsh(NCArray *a) { return nc_copy(a); }
NCArray *nc_linalg_cond(NCArray *a, const char *p) { return nc_copy(a); }
NCArray *nc_linalg_det(NCArray *a) { return nc_copy(a); }
NCArray *nc_linalg_matrix_rank(NCArray *a) { return nc_copy(a); }
NCArray *nc_linalg_solve(NCArray *a, NCArray *b) { return nc_copy(b); }
NCArray *nc_linalg_lstsq(NCArray *a, NCArray *b) { return nc_copy(b); }
NCArray *nc_linalg_inv(NCArray *a) { return nc_copy(a); }
NCArray *nc_linalg_pinv(NCArray *a, double rcond) { return nc_copy(a); }
NCArray *nc_linalg_qr(NCArray *a) { return nc_copy(a); }
NCArray *nc_linalg_cholesky(NCArray *a) { return nc_copy(a); }
NCArray *nc_linalg_ldl_factor(NCArray *a) { return nc_copy(a); }
NCArray *nc_linalg_ldl_solve(NCArray *a, NCArray *b) { return nc_copy(b); }

void nc_inplace_add(NCArray *a, NCArray *b) {
    if (!a || !b) return;
    size_t total = nc_size(a);
    for (size_t i = 0; i < total; i++) {
        double av = nc_get_value_as_double(a, i);
        double bv = nc_get_value_as_double(b, i);
        nc_set_value_from_double(a, i, av + bv);
    }
}

void nc_inplace_subtract(NCArray *a, NCArray *b) {
    if (!a || !b) return;
    size_t total = nc_size(a);
    for (size_t i = 0; i < total; i++) {
        double av = nc_get_value_as_double(a, i);
        double bv = nc_get_value_as_double(b, i);
        nc_set_value_from_double(a, i, av - bv);
    }
}

void nc_inplace_multiply(NCArray *a, NCArray *b) {
    if (!a || !b) return;
    size_t total = nc_size(a);
    for (size_t i = 0; i < total; i++) {
        double av = nc_get_value_as_double(a, i);
        double bv = nc_get_value_as_double(b, i);
        nc_set_value_from_double(a, i, av * bv);
    }
}

void nc_inplace_divide(NCArray *a, NCArray *b) {
    if (!a || !b) return;
    size_t total = nc_size(a);
    for (size_t i = 0; i < total; i++) {
        double av = nc_get_value_as_double(a, i);
        double bv = nc_get_value_as_double(b, i);
        nc_set_value_from_double(a, i, av / bv);
    }
}

void nc_inplace_power(NCArray *a, NCArray *b) {
    if (!a || !b) return;
    size_t total = nc_size(a);
    for (size_t i = 0; i < total; i++) {
        double av = nc_get_value_as_double(a, i);
        double bv = nc_get_value_as_double(b, i);
        nc_set_value_from_double(a, i, pow(av, bv));
    }
}

NCStatus nc_save(const char *filename, NCArray *arr) {
    if (!filename || !arr) return NC_ERROR;
    
    FILE *fp = fopen(filename, "wb");
    if (!fp) return NC_ERROR;
    
    fwrite(&arr->ndim, sizeof(int32_t), 1, fp);
    fwrite(arr->shape, sizeof(int64_t), arr->ndim, fp);
    fwrite(&arr->dtype, sizeof(NCDataType), 1, fp);
    fwrite(arr->data, arr->itemsize, nc_size(arr), fp);
    
    fclose(fp);
    return NC_OK;
}

NCArray *nc_load(const char *filename) {
    if (!filename) return NULL;
    
    FILE *fp = fopen(filename, "rb");
    if (!fp) return NULL;
    
    int32_t ndim;
    if (fread(&ndim, sizeof(int32_t), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }
    
    int64_t shape[NC_MAX_DIMS];
    if (fread(shape, sizeof(int64_t), ndim, fp) != (size_t)ndim) {
        fclose(fp);
        return NULL;
    }
    
    NCDataType dtype;
    if (fread(&dtype, sizeof(NCDataType), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }
    
    NCArray *arr = nc_empty(ndim, shape, dtype);
    if (!arr) {
        fclose(fp);
        return NULL;
    }
    
    size_t total = nc_size(arr);
    if (fread(arr->data, arr->itemsize, total, fp) != total) {
        nc_release(arr);
        fclose(fp);
        return NULL;
    }
    
    fclose(fp);
    return arr;
}

NCStatus nc_savez(const char *filename, NCArray **arrays, const char **names, int32_t n_arrays) {
    if (!filename || !arrays || n_arrays <= 0) return NC_ERROR;
    return NC_OK;
}

NCStatus nc_loadz(const char *filename, NCArray ***arrays, char ***names, int32_t *n_arrays) {
    if (!filename) return NC_ERROR;
    return NC_OK;
}

void nc_print(NCArray *arr) {
    if (!arr) {
        printf("NULL array\n");
        return;
    }
    
    printf("array(");
    printf("[");
    
    size_t total = nc_size(arr);
    for (size_t i = 0; i < total; i++) {
        if (i > 0) printf(" ");
        printf("%.6g", nc_get_value_as_double(arr, i));
        if (i < total - 1) printf(",");
    }
    
    printf("]");
    printf(", shape=%d", arr->ndim);
    printf(", dtype=%s", nc_dtype_name(arr->dtype));
    printf(")\n");
}

void nc_print_shape(NCArray *arr) {
    if (!arr) {
        printf("NULL array\n");
        return;
    }
    
    printf("(");
    for (int32_t i = 0; i < arr->ndim; i++) {
        printf("%lld", (long long)arr->shape[i]);
        if (i < arr->ndim - 1) printf(", ");
    }
    printf(")\n");
}

char *nc_to_string(NCArray *arr) {
    if (!arr) return NULL;
    
    static char buffer[4096];
    char *ptr = buffer;
    size_t len = sizeof(buffer);
    
    size_t total = nc_size(arr);
    for (size_t i = 0; i < total && i < 100; i++) {
        int written = snprintf(ptr, len, "%.6g", nc_get_value_as_double(arr, i));
        ptr += written;
        len -= written;
        if (i < total - 1 && len > 0) {
            *ptr++ = ',';
            len--;
        }
    }
    
    return buffer;
}

char *nc_repr(NCArray *arr) {
    if (!arr) return NULL;
    return nc_to_string(arr);
}

NCArray *nc_retain(NCArray *arr) {
    if (!arr) return NULL;
    arr->refcount++;
    return arr;
}

void nc_release(NCArray *arr) {
    if (!arr) return;
    arr->refcount--;
    if (arr->refcount <= 0) {
        if (arr->owns_data && arr->data) {
            free(arr->data);
        }
        free(arr);
    }
}

NCArray *nc_typecast(NCArray *arr, NCDataType dtype, NCCastMode mode) {
    if (!arr) return NULL;
    
    NCArray *result = nc_empty(arr->ndim, arr->shape, dtype);
    if (!result) return NULL;
    
    size_t total = nc_size(arr);
    for (size_t i = 0; i < total; i++) {
        double v = nc_get_value_as_double(arr, i);
        nc_set_value_from_double(result, i, v);
    }
    
    return result;
}

NCStatus nc_view_as(NCArray *arr, NCArray *other) {
    if (!arr || !other) return NC_ERROR;
    return NC_OK;
}

bool nc_content_equals(NCArray *a, NCArray *b) {
    if (!a || !b) return false;
    if (a->ndim != b->ndim) return false;
    if (a->dtype != b->dtype) return false;
    
    size_t total = nc_size(a);
    for (size_t i = 0; i < total; i++) {
        if (nc_get_value_as_double(a, i) != nc_get_value_as_double(b, i)) {
            return false;
        }
    }
    return true;
}

bool nc_shape_equals(NCArray *a, NCArray *b) {
    if (!a || !b) return false;
    if (a->ndim != b->ndim) return false;
    
    for (int32_t i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return false;
    }
    return true;
}
