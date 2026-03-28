#ifndef NUMC_H
#define NUMC_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NC_VERSION "0.1.0"
#define NC_MAX_DIMS 16
#define NC_MAX_TENSOR_SIZE 2147483647

typedef enum {
    NC_OK = 0,
    NC_ERROR = -1,
    NC_MEMORY_ERROR = -2,
    NC_DIMENSION_ERROR = -3,
    NC_SHAPE_ERROR = -4,
    NC_TYPE_ERROR = -5,
    NC_VALUE_ERROR = -6,
    NC_INDEX_ERROR = -7,
    NC_NOT_IMPLEMENTED = -8
} NCStatus;

typedef enum {
    NC_INVALID = 0,
    NC_BOOL,
    NC_INT8,
    NC_INT16,
    NC_INT32,
    NC_INT64,
    NC_UINT8,
    NC_UINT16,
    NC_UINT32,
    NC_UINT64,
    NC_FLOAT32,
    NC_FLOAT64,
    NC_COMPLEX64,
    NC_COMPLEX128,
    NC_STRING
} NCDataType;

typedef enum {
    NC_ROW_MAJOR = 0,
    NC_COL_MAJOR = 1
} NCOrder;

typedef enum {
    NC_NO_CAST = 0,
    NC_SAFE_CAST = 1,
    NC_FORCE_CAST = 2
} NCCastMode;

typedef struct NCArray {
    void *data;
    int32_t ndim;
    int64_t shape[NC_MAX_DIMS];
    int64_t strides[NC_MAX_DIMS];
    size_t itemsize;
    NCDataType dtype;
    bool owns_data;
    int refcount;
    struct NCArray *base;
} NCArray;

typedef struct NCTensor {
    NCArray base;
    int64_t offset;
    NCOrder order;
} NCTensor;

typedef struct NCComplex64 {
    float real;
    float imag;
} NCComplex64;

typedef struct NCComplex128 {
    double real;
    double imag;
} NCComplex128;

typedef struct NCView {
    NCArray *array;
    int64_t start[NC_MAX_DIMS];
    int64_t step[NC_MAX_DIMS];
    int64_t count[NC_MAX_DIMS];
} NCView;

typedef struct NCIterator {
    NCArray *array;
    int64_t index;
    int64_t current[NC_MAX_DIMS];
    void *current_ptr;
} NCIterator;

typedef NCArray* (*NCArrayFunc0)(void);
typedef NCArray* (*NCArrayFunc1)(NCArray*);
typedef NCArray* (*NCArrayFunc2)(NCArray*, NCArray*);
typedef void (*NCUnaryOp)(void *, void *, size_t);
typedef void (*NCBinaryOp)(void *, void *, void *, size_t);

extern NCDataType NC_DEFAULT_INT;
extern NCDataType NC_DEFAULT_FLOAT;

const char *nc_version(void);
const char *nc_status_string(NCStatus status);

size_t nc_dtype_size(NCDataType dtype);
const char *nc_dtype_name(NCDataType dtype);
NCDataType nc_dtype_from_string(const char *name);
bool nc_dtype_is_integer(NCDataType dtype);
bool nc_dtype_is_float(NCDataType dtype);
bool nc_dtype_is_complex(NCDataType dtype);
bool nc_dtype_is_numeric(NCDataType dtype);

size_t nc_size(NCArray *arr);
size_t nc_nbytes(NCArray *arr);
int32_t nc_ndim(NCArray *arr);
int64_t *nc_shape(NCArray *arr);
int64_t nc_shape_at(NCArray *arr, int32_t dim);
int64_t *nc_strides(NCArray *arr);
int64_t nc_stride_at(NCArray *arr, int32_t dim);
size_t nc_itemsize(NCArray *arr);
NCDataType nc_dtype(NCArray *arr);
bool nc_is_contiguous(NCArray *arr);
bool nc_is_c_contiguous(NCArray *arr);
bool nc_is_f_contiguous(NCArray *arr);
void nc_update_strides(NCArray *arr);

NCArray *nc_empty(int32_t ndim, const int64_t *shape, NCDataType dtype);
NCArray *nc_empty_like(NCArray *arr);
NCArray *nc_zeros(int32_t ndim, const int64_t *shape, NCDataType dtype);
NCArray *nc_zeros_like(NCArray *arr);
NCArray *nc_ones(int32_t ndim, const int64_t *shape, NCDataType dtype);
NCArray *nc_ones_like(NCArray *arr);
NCArray *nc_full(int32_t ndim, const int64_t *shape, void *fill_value, NCDataType dtype);
NCArray *nc_full_like(NCArray *arr, void *fill_value);
NCArray *nc_arange(double start, double stop, double step, NCDataType dtype);
NCArray *nc_linspace(double start, double stop, int64_t num, bool endpoint, NCDataType dtype);
NCArray *nc_logspace(double start, double stop, int64_t num, bool endpoint, double base, NCDataType dtype);
NCArray *nc_geomspace(double start, double stop, int64_t num, bool endpoint, NCDataType dtype);
NCArray *nc_identity(int64_t n, NCDataType dtype);
NCArray *nc_eye(int64_t n, int64_t m, int64_t k, NCDataType dtype);
NCArray *nc_diag(NCArray *arr, int64_t k);
NCArray *nc_tril(NCArray *arr, int64_t k);
NCArray *nc_triu(NCArray *arr, int64_t k);
NCArray *nc_from_buffer(void *buffer, size_t size, NCDataType dtype, int32_t ndim, const int64_t *shape);
NCArray *nc_from_pointer(void *ptr, int32_t ndim, const int64_t *shape, NCDataType dtype, bool owns_data);
NCArray *nc_copy(NCArray *arr);
NCArray *nc_asarray(NCArray *arr);
NCArray *nc_asarray_c(void *data, int32_t ndim, const int64_t *shape, NCDataType dtype);
NCArray *nc_asarray_f(void *data, int32_t ndim, const int64_t *shape, NCDataType dtype);
NCArray *nc_ascontiguousarray(NCArray *arr);

NCStatus nc_reshape(NCArray *arr, int32_t ndim, const int64_t *shape);
NCStatus nc_resize(NCArray *arr, int32_t ndim, const int64_t *shape);
NCStatus nc_squeeze(NCArray *arr, int32_t axis);
NCStatus nc_expand_dims(NCArray *arr, int32_t axis);
NCStatus nc_swapaxes(NCArray *arr, int32_t axis1, int32_t axis2);
NCStatus nc_moveaxis(NCArray *arr, int32_t source, int32_t dest);
NCArray *nctranspose(NCArray *arr, const int32_t *axes);
NCArray *ncflatten(NCArray *arr);
NCArray *ncravel(NCArray *arr);
NCArray *ncreshape(NCArray *arr, int32_t ndim, const int64_t *shape);
NCArray *ncsqueeze(NCArray *arr, int32_t axis);
NCArray *ncexpand_dims(NCArray *arr, int32_t axis);
NCArray *ncselect(NCArray *cond, NCArray *arr1, NCArray *arr2);

NCArray *nc_add(NCArray *a, NCArray *b);
NCArray *nc_subtract(NCArray *a, NCArray *b);
NCArray *nc_multiply(NCArray *a, NCArray *b);
NCArray *nc_divide(NCArray *a, NCArray *b);
NCArray *nc_power(NCArray *a, NCArray *b);
NCArray *nc_mod(NCArray *a, NCArray *b);
NCArray *nc_floor_divide(NCArray *a, NCArray *b);
NCArray *nc_true_divide(NCArray *a, NCArray *b);

NCArray *nc_equal(NCArray *a, NCArray *b);
NCArray *nc_not_equal(NCArray *a, NCArray *b);
NCArray *nc_less(NCArray *a, NCArray *b);
NCArray *nc_less_equal(NCArray *a, NCArray *b);
NCArray *nc_greater(NCArray *a, NCArray *b);
NCArray *nc_greater_equal(NCArray *a, NCArray *b);

NCArray *nc_logical_and(NCArray *a, NCArray *b);
NCArray *nc_logical_or(NCArray *a, NCArray *b);
NCArray *nc_logical_xor(NCArray *a, NCArray *b);
NCArray *nc_logical_not(NCArray *a);

NCArray *nc_bitwise_and(NCArray *a, NCArray *b);
NCArray *nc_bitwise_or(NCArray *a, NCArray *b);
NCArray *nc_bitwise_xor(NCArray *a, NCArray *b);
NCArray *nc_bitwise_not(NCArray *a);
NCArray *nc_left_shift(NCArray *a, NCArray *b);
NCArray *nc_right_shift(NCArray *a, NCArray *b);

NCArray *nc_abs(NCArray *arr);
NCArray *nc_fabs(NCArray *arr);
NCArray *nc_sign(NCArray *arr);
NCArray *nc_floor(NCArray *arr);
NCArray *nc_ceil(NCArray *arr);
NCArray *nc_round(NCArray *arr);
NCArray *nc_trunc(NCArray *arr);

NCArray *nc_exp(NCArray *arr);
NCArray *nc_expm1(NCArray *arr);
NCArray *nc_exp2(NCArray *arr);
NCArray *nc_log(NCArray *arr);
NCArray *nc_log1p(NCArray *arr);
NCArray *nc_log2(NCArray *arr);
NCArray *nc_log10(NCArray *arr);

NCArray *nc_sqrt(NCArray *arr);
NCArray *nc_cbrt(NCArray *arr);
NCArray *nc_square(NCArray *arr);

NCArray *nc_sin(NCArray *arr);
NCArray *nc_cos(NCArray *arr);
NCArray *nc_tan(NCArray *arr);
NCArray *nc_arcsin(NCArray *arr);
NCArray *nc_arccos(NCArray *arr);
NCArray *nc_arctan(NCArray *arr);
NCArray *nc_sinh(NCArray *arr);
NCArray *nc_cosh(NCArray *arr);
NCArray *nc_tanh(NCArray *arr);
NCArray *nc_arcsinh(NCArray *arr);
NCArray *nc_arccosh(NCArray *arr);
NCArray *nc_arctanh(NCArray *arr);

NCArray *nc_deg2rad(NCArray *arr);
NCArray *nc_rad2deg(NCArray *arr);

NCArray *nc_negate(NCArray *arr);
NCArray *nc_invert(NCArray *arr);

NCArray *nc_dot(NCArray *a, NCArray *b);
NCArray *nc_matmul(NCArray *a, NCArray *b);
NCArray *nc_inner(NCArray *a, NCArray *b);
NCArray *nc_outer(NCArray *a, NCArray *b);
NCArray *nc_cross(NCArray *a, NCArray *b, int32_t axis);
NCArray *nc_trace(NCArray *arr, int32_t offset, int32_t axis1, int32_t axis2);
NCArray *nc_einsum(const char *subscripts, NCArray *arrays, ...);
NCArray *nc_tensordot(NCArray *a, NCArray *b, const int32_t *axes_a, int32_t n_axes_a, const int32_t *axes_b, int32_t n_axes_b);

NCArray *nc_sum(NCArray *arr, const int32_t *axis, int32_t naxis);
NCArray *nc_prod(NCArray *arr, const int32_t *axis, int32_t naxis);
NCArray *nc_mean(NCArray *arr, const int32_t *axis, int32_t naxis);
NCArray *nc_var(NCArray *arr, const int32_t *axis, int32_t naxis);
NCArray *nc_std(NCArray *arr, const int32_t *axis, int32_t naxis);
NCArray *nc_min(NCArray *arr, const int32_t *axis, int32_t naxis);
NCArray *nc_max(NCArray *arr, const int32_t *axis, int32_t naxis);
NCArray *nc_argmin(NCArray *arr, int32_t axis);
NCArray *nc_argmax(NCArray *arr, int32_t axis);
NCArray *nc_all(NCArray *arr, const int32_t *axis, int32_t naxis);
NCArray *nc_any(NCArray *arr, const int32_t *axis, int32_t naxis);
NCArray *nc_cumsum(NCArray *arr, int32_t axis);
NCArray *nc_cumprod(NCArray *arr, int32_t axis);
NCArray *nc_nancumsum(NCArray *arr, int32_t axis);
NCArray *nc_nancumprod(NCArray *arr, int32_t axis);

int64_t nc_count_nonzero(NCArray *arr);
bool nc_isnan(NCArray *arr);
bool nc_isinf(NCArray *arr);
bool nc_isfinite(NCArray *arr);
bool nc_isneginf(NCArray *arr);
bool nc_isposinf(NCArray *arr);

double nc_nanmin(NCArray *arr);
double nc_nanmax(NCArray *arr);
double nc_nanmean(NCArray *arr);
double nc_nanstd(NCArray *arr);
double nc_nanvar(NCArray *arr);
double nc_nansum(NCArray *arr);
double nc_nanprod(NCArray *arr);

NCArray *nc_where(NCArray *condition, NCArray *x, NCArray *y);
NCArray *nc_nan_to_num(NCArray *arr);
NCArray *nc_interp(NCArray *x, NCArray *xp, NCArray *fp, double left, double right);

NCArray *nc_concatenate(NCArray **arrays, int32_t n, int32_t axis);
NCArray *nc_stack(NCArray **arrays, int32_t n, int32_t axis);
NCArray *nc_vstack(NCArray **arrays, int32_t n);
NCArray *nc_hstack(NCArray **arrays, int32_t n);
NCArray *nc_dstack(NCArray **arrays, int32_t n);
NCArray *nc_split(NCArray *arr, int32_t n_sections, int32_t axis);
NCArray **nc_array_split(NCArray *arr, int32_t n_sections, int32_t axis);
NCArray *nc_tile(NCArray *arr, const int32_t *reps, int32_t n_reps);
NCArray *nc_repeat(NCArray *arr, int64_t repeats, int32_t axis);
NCArray *nc_pad(NCArray *arr, const int64_t *pad_width, NCArray *constant_values);
NCArray *nc_extract(NCArray *condition, NCArray *arr);

void *nc_getitem(NCArray *arr, const int64_t *indices, int32_t n_indices);
NCStatus nc_setitem(NCArray *arr, const int64_t *indices, int32_t n_indices, void *value);
NCArray *nc_slice(NCArray *arr, const int64_t *starts, const int64_t *stops, const int64_t *steps, int32_t n_slices);
NCView *nc_view(NCArray *arr, const int64_t *starts, const int64_t *stops, const int64_t *steps, int32_t n_slices);
NCStatus nc_set_slice(NCArray *arr, NCView *view, NCArray *value);

NCArray *nc_fft_fft(NCArray *arr, int32_t n, int32_t axis);
NCArray *nc_fft_ifft(NCArray *arr, int32_t n, int32_t axis);
NCArray *nc_fft_fft2(NCArray *arr, const int32_t *s, const int32_t *axes);
NCArray *nc_fft_ifft2(NCArray *arr, const int32_t *s, const int32_t *axes);
NCArray *nc_fft_fftn(NCArray *arr, const int32_t *s, const int32_t *axes);
NCArray *nc_fft_ifftn(NCArray *arr, const int32_t *s, const int32_t *axes);
NCArray *nc_fft_rfft(NCArray *arr, int32_t n, int32_t axis);
NCArray *nc_fft_irfft(NCArray *arr, int32_t n, int32_t axis);
NCArray *nc_fft_hfft(NCArray *arr, int32_t n, int32_t axis);
NCArray *nc_fft_ihfft(NCArray *arr, int32_t n, int32_t axis);
NCArray *nc_fft_fftfreq(int64_t n, double d, NCDataType dtype);
NCArray *nc_fft_rfftfreq(int64_t n, double d, NCDataType dtype);

NCArray *nc_sort(NCArray *arr, int32_t axis);
NCArray *nc_argsort(NCArray *arr, int32_t axis);
NCArray *nc_searchsorted(NCArray *arr, NCArray *v, const char *side);
NCArray *nc_partition(NCArray *arr, int32_t kth, int32_t axis);

NCArray *nc_random_rand(int32_t ndim, const int64_t *shape);
NCArray *nc_random_randn(int32_t ndim, const int64_t *shape, NCDataType dtype);
NCArray *nc_random_randint(int64_t low, int64_t high, int32_t ndim, const int64_t *shape);
NCArray *nc_random_random(int32_t ndim, const int64_t *shape, NCDataType dtype);
NCArray *nc_random_choice(int64_t n, int64_t size, bool replace);
void nc_random_seed(uint64_t seed);
void nc_random_shuffle(NCArray *arr);
void nc_random_permutation(NCArray *arr);

NCArray *nc_linalg_norm(NCArray *arr, const char *ord);
NCArray *nc_linalg_dot(NCArray *a, NCArray *b);
NCArray *nc_linalg_matmul(NCArray *a, NCArray *b);
NCArray *nc_linalg_svd(NCArray *a, bool full_matrices);
NCArray *nc_linalg_eig(NCArray *a);
NCArray *nc_linalg_eigh(NCArray *a);
NCArray *nc_linalg_eigvals(NCArray *a);
NCArray *nc_linalg_eigvalsh(NCArray *a);
NCArray *nc_linalg_cond(NCArray *a, const char *p);
NCArray *nc_linalg_det(NCArray *a);
NCArray *nc_linalg_matrix_rank(NCArray *a);
NCArray *nc_linalg_solve(NCArray *a, NCArray *b);
NCArray *nc_linalg_lstsq(NCArray *a, NCArray *b);
NCArray *nc_linalg_inv(NCArray *a);
NCArray *nc_linalg_pinv(NCArray *a, double rcond);
NCArray *nc_linalg_qr(NCArray *a);
NCArray *nc_linalg_cholesky(NCArray *a);
NCArray *nc_linalg_ldl_factor(NCArray *a);
NCArray *nc_linalg_ldl_solve(NCArray *a, NCArray *b);

void nc_inplace_add(NCArray *a, NCArray *b);
void nc_inplace_subtract(NCArray *a, NCArray *b);
void nc_inplace_multiply(NCArray *a, NCArray *b);
void nc_inplace_divide(NCArray *a, NCArray *b);
void nc_inplace_power(NCArray *a, NCArray *b);

NCArray *nc_broadcast_to(NCArray *arr, int32_t ndim, const int64_t *shape);
bool nc_can_broadcast(NCArray *a, NCArray *b);
int32_t nc_broadcast_shapes(int32_t n_in, const int64_t **shapes, int32_t n_shapes, int64_t *out_shape);

NCStatus nc_save(const char *filename, NCArray *arr);
NCArray *nc_load(const char *filename);
NCStatus nc_savez(const char *filename, NCArray **arrays, const char **names, int32_t n_arrays);
NCStatus nc_loadz(const char *filename, NCArray ***arrays, char ***names, int32_t *n_arrays);

void nc_print(NCArray *arr);
void nc_print_shape(NCArray *arr);
char *nc_to_string(NCArray *arr);
char *nc_repr(NCArray *arr);

NCArray *nc_retain(NCArray *arr);
void nc_release(NCArray *arr);

NCArray *nc_typecast(NCArray *arr, NCDataType dtype, NCCastMode mode);
NCStatus nc_view_as(NCArray *arr, NCArray *other);
bool nc_content_equals(NCArray *a, NCArray *b);
bool nc_shape_equals(NCArray *a, NCArray *b);

#define nc_free(arr) nc_release(arr)

NCArray *nc_make_1d(NCDataType dtype, int64_t n, ...);
NCArray *nc_make_2d(NCDataType dtype, int64_t rows, int64_t cols, ...);
NCArray *nc_make_1d_auto(int n, ...);
NCArray *nc_make_2d_auto(int64_t rows, int64_t cols, int n, ...);
NCArray *nc_make_1d_float_auto(int n, ...);
NCArray *nc_make_2d_float_auto(int64_t rows, int64_t cols, int n, ...);

#define _NC_COUNT_ARGS(...) _NC_COUNT_ARGS_IMPL(__VA_ARGS__, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
#define _NC_COUNT_ARGS_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _N, ...) _N

#define NC_INT(...) \
    nc_make_1d_auto(_NC_COUNT_ARGS(__VA_ARGS__), (int64_t)__VA_ARGS__)

#define NC_FLOAT(...) \
    nc_make_1d_float_auto(_NC_COUNT_ARGS(__VA_ARGS__), (double)__VA_ARGS__)

#define NC_INT2D(_rows, _cols, ...) \
    nc_make_2d_auto(_rows, _cols, _NC_COUNT_ARGS(__VA_ARGS__), (int64_t)__VA_ARGS__)

#define NC_FLOAT2D(_rows, _cols, ...) \
    nc_make_2d_float_auto(_rows, _cols, _NC_COUNT_ARGS(__VA_ARGS__), (double)__VA_ARGS__)

#define NC_INT8(...) nc_make_1d(NC_INT8, _NC_COUNT_ARGS(__VA_ARGS__), (int64_t)__VA_ARGS__)
#define NC_INT16(...) nc_make_1d(NC_INT16, _NC_COUNT_ARGS(__VA_ARGS__), (int64_t)__VA_ARGS__)
#define NC_INT32(...) nc_make_1d(NC_INT32, _NC_COUNT_ARGS(__VA_ARGS__), (int64_t)__VA_ARGS__)
#define NC_INT64(...) nc_make_1d(NC_INT64, _NC_COUNT_ARGS(__VA_ARGS__), (int64_t)__VA_ARGS__)
#define NC_UINT8(...) nc_make_1d(NC_UINT8, _NC_COUNT_ARGS(__VA_ARGS__), (int64_t)__VA_ARGS__)
#define NC_UINT16(...) nc_make_1d(NC_UINT16, _NC_COUNT_ARGS(__VA_ARGS__), (int64_t)__VA_ARGS__)
#define NC_UINT32(...) nc_make_1d(NC_UINT32, _NC_COUNT_ARGS(__VA_ARGS__), (int64_t)__VA_ARGS__)
#define NC_UINT64(...) nc_make_1d(NC_UINT64, _NC_COUNT_ARGS(__VA_ARGS__), (int64_t)__VA_ARGS__)
#define NC_FLOAT32(...) nc_make_1d(NC_FLOAT32, _NC_COUNT_ARGS(__VA_ARGS__), (double)__VA_ARGS__)
#define NC_FLOAT64(...) nc_make_1d(NC_FLOAT64, _NC_COUNT_ARGS(__VA_ARGS__), (double)__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif
