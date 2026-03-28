#include "NumC.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define EPSILON 1e-10

void test_creation() {
    printf("Testing array creation...\n");
    
    int64_t shape1d[1] = {5};
    NCArray *arr1d = nc_zeros(1, shape1d, NC_FLOAT64);
    assert(arr1d != NULL);
    assert(nc_ndim(arr1d) == 1);
    assert(nc_shape_at(arr1d, 0) == 5);
    assert(nc_size(arr1d) == 5);
    nc_release(arr1d);
    
    int64_t shape2d[2] = {3, 4};
    NCArray *arr2d = nc_ones(2, shape2d, NC_FLOAT64);
    assert(arr2d != NULL);
    assert(nc_ndim(arr2d) == 2);
    assert(nc_shape_at(arr2d, 0) == 3);
    assert(nc_shape_at(arr2d, 1) == 4);
    assert(nc_size(arr2d) == 12);
    nc_release(arr2d);
    
    NCArray *identity = nc_identity(4, NC_FLOAT64);
    assert(identity != NULL);
    assert(nc_ndim(identity) == 2);
    assert(nc_shape_at(identity, 0) == 4);
    assert(nc_shape_at(identity, 1) == 4);
    nc_release(identity);
    
    NCArray *eye = nc_eye(3, 5, 1, NC_FLOAT64);
    assert(eye != NULL);
    nc_release(eye);
    
    NCArray *range = nc_arange(0.0, 10.0, 1.0, NC_FLOAT64);
    assert(range != NULL);
    assert(nc_size(range) == 10);
    nc_release(range);
    
    NCArray *linspace_arr = nc_linspace(0.0, 1.0, 5, true, NC_FLOAT64);
    assert(linspace_arr != NULL);
    assert(nc_size(linspace_arr) == 5);
    nc_release(linspace_arr);
    
    printf("  Array creation tests passed!\n");
}

void test_arithmetic() {
    printf("Testing arithmetic operations...\n");
    
    NCArray *a = nc_arange(1.0, 5.0, 1.0, NC_FLOAT64);
    NCArray *b = nc_arange(5.0, 9.0, 1.0, NC_FLOAT64);
    
    NCArray *sum = nc_add(a, b);
    assert(sum != NULL);
    for (int64_t i = 0; i < nc_size(sum); i++) {
        double expected = (i + 1) + (i + 5);
        assert(fabs(((double*)sum->data)[i] - expected) < EPSILON);
    }
    nc_release(sum);
    
    NCArray *diff = nc_subtract(b, a);
    assert(diff != NULL);
    for (int64_t i = 0; i < nc_size(diff); i++) {
        assert(fabs(((double*)diff->data)[i] - 4.0) < EPSILON);
    }
    nc_release(diff);
    
    NCArray *prod = nc_multiply(a, b);
    assert(prod != NULL);
    nc_release(prod);
    
    NCArray *quot = nc_divide(b, a);
    assert(quot != NULL);
    nc_release(quot);
    
    NCArray *pow_arr = nc_power(a, nc_full(1, (int64_t[]){4}, (double[]){2.0}, NC_FLOAT64));
    assert(pow_arr != NULL);
    nc_release(pow_arr);
    
    nc_release(a);
    nc_release(b);
    
    printf("  Arithmetic tests passed!\n");
}

void test_comparison() {
    printf("Testing comparison operations...\n");
    
    NCArray *a = nc_full(1, (int64_t[]){5}, (double[]){5.0}, NC_FLOAT64);
    NCArray *b = nc_full(1, (int64_t[]){5}, (double[]){5.0}, NC_FLOAT64);
    
    NCArray *eq = nc_equal(a, b);
    assert(eq != NULL);
    assert(((bool*)eq->data)[0] == true);
    nc_release(eq);
    
    NCArray *neq = nc_not_equal(a, b);
    assert(neq != NULL);
    assert(((bool*)neq->data)[0] == false);
    nc_release(neq);
    
    NCArray *greater = nc_greater(b, a);
    assert(greater != NULL);
    assert(((bool*)greater->data)[0] == false);
    nc_release(greater);
    
    nc_release(a);
    nc_release(b);
    
    printf("  Comparison tests passed!\n");
}

void test_math_functions() {
    printf("Testing mathematical functions...\n");
    
    NCArray *arr = nc_arange(0.0, 2 * M_PI, 0.1, NC_FLOAT64);
    
    NCArray *sin_arr = nc_sin(arr);
    assert(sin_arr != NULL);
    nc_release(sin_arr);
    
    NCArray *cos_arr = nc_cos(arr);
    assert(cos_arr != NULL);
    nc_release(cos_arr);
    
    NCArray *exp_arr = nc_exp(nc_full(1, (int64_t[]){1}, (double[]){0.0}, NC_FLOAT64));
    assert(exp_arr != NULL);
    nc_release(exp_arr);
    
    NCArray *log_arr = nc_log(nc_full(1, (int64_t[]){1}, (double[]){1.0}, NC_FLOAT64));
    assert(log_arr != NULL);
    nc_release(log_arr);
    
    NCArray *sqrt_arr = nc_sqrt(nc_full(1, (int64_t[]){1}, (double[]){4.0}, NC_FLOAT64));
    assert(sqrt_arr != NULL);
    nc_release(sqrt_arr);
    
    nc_release(arr);
    
    printf("  Mathematical function tests passed!\n");
}

void test_reductions() {
    printf("Testing reduction operations...\n");
    
    NCArray *arr = nc_arange(1.0, 6.0, 1.0, NC_FLOAT64);
    
    NCArray *sum = nc_sum(arr, NULL, 0);
    assert(sum != NULL);
    assert(fabs(((double*)sum->data)[0] - 15.0) < EPSILON);
    nc_release(sum);
    
    NCArray *mean = nc_mean(arr, NULL, 0);
    assert(mean != NULL);
    assert(fabs(((double*)mean->data)[0] - 3.5) < EPSILON);
    nc_release(mean);
    
    NCArray *min_arr = nc_min(arr, NULL, 0);
    assert(min_arr != NULL);
    assert(fabs(((double*)min_arr->data)[0] - 1.0) < EPSILON);
    nc_release(min_arr);
    
    NCArray *max_arr = nc_max(arr, NULL, 0);
    assert(max_arr != NULL);
    assert(fabs(((double*)max_arr->data)[0] - 5.0) < EPSILON);
    nc_release(max_arr);
    
    NCArray *var = nc_var(arr, NULL, 0);
    assert(var != NULL);
    nc_release(var);
    
    NCArray *std = nc_std(arr, NULL, 0);
    assert(std != NULL);
    nc_release(std);
    
    nc_release(arr);
    
    printf("  Reduction tests passed!\n");
}

void test_linear_algebra() {
    printf("Testing linear algebra operations...\n");
    
    int64_t shape2d[2] = {2, 2};
    NCArray *a = nc_identity(2, NC_FLOAT64);
    NCArray *b = nc_identity(2, NC_FLOAT64);
    
    NCArray *dot_prod = nc_dot(a, b);
    assert(dot_prod != NULL);
    nc_release(dot_prod);
    
    NCArray *mm = nc_matmul(a, b);
    assert(mm != NULL);
    nc_release(mm);
    
    NCArray *inner_prod = nc_inner(a, b);
    assert(inner_prod != NULL);
    nc_release(inner_prod);
    
    nc_release(a);
    nc_release(b);
    
    printf("  Linear algebra tests passed!\n");
}

void test_array_operations() {
    printf("Testing array manipulation...\n");
    
    NCArray *arr = nc_arange(1.0, 13.0, 1.0, NC_FLOAT64);
    assert(nc_size(arr) == 12);
    
    int64_t new_shape[2] = {3, 4};
    NCStatus status = nc_reshape(arr, 2, new_shape);
    assert(status == NC_OK);
    assert(nc_ndim(arr) == 2);
    nc_release(arr);
    
    NCArray *arr2 = nc_zeros(2, (int64_t[]){3, 3}, NC_FLOAT64);
    NCArray *transposed = nctranspose(arr2, NULL);
    assert(transposed != NULL);
    nc_release(transposed);
    nc_release(arr2);
    
    NCArray *concat_arrs[2];
    concat_arrs[0] = nc_arange(1.0, 3.0, 1.0, NC_FLOAT64);
    concat_arrs[1] = nc_arange(3.0, 5.0, 1.0, NC_FLOAT64);
    NCArray *concat = nc_concatenate(concat_arrs, 2, 0);
    assert(concat != NULL);
    assert(nc_size(concat) == 4);
    nc_release(concat);
    nc_release(concat_arrs[0]);
    nc_release(concat_arrs[1]);
    
    printf("  Array manipulation tests passed!\n");
}

void test_broadcasting() {
    printf("Testing broadcasting...\n");
    
    NCArray *a = nc_zeros(2, (int64_t[]){3, 1}, NC_FLOAT64);
    NCArray *b = nc_zeros(2, (int64_t[]){1, 4}, NC_FLOAT64);
    
    bool can_broadcast = nc_can_broadcast(a, b);
    assert(can_broadcast == true);
    
    NCArray *result = nc_add(a, b);
    assert(result != NULL);
    assert(nc_shape_at(result, 0) == 3);
    assert(nc_shape_at(result, 1) == 4);
    nc_release(result);
    
    nc_release(a);
    nc_release(b);
    
    printf("  Broadcasting tests passed!\n");
}

void test_logical_operations() {
    printf("Testing logical operations...\n");
    
    NCArray *a = nc_full(1, (int64_t[]){3}, (bool[]){true}, NC_BOOL);
    NCArray *b = nc_full(1, (int64_t[]){3}, (bool[]){false}, NC_BOOL);
    
    NCArray *and_result = nc_logical_and(a, b);
    assert(and_result != NULL);
    assert(((bool*)and_result->data)[0] == false);
    nc_release(and_result);
    
    NCArray *or_result = nc_logical_or(a, b);
    assert(or_result != NULL);
    assert(((bool*)or_result->data)[0] == true);
    nc_release(or_result);
    
    NCArray *not_result = nc_logical_not(b);
    assert(not_result != NULL);
    assert(((bool*)not_result->data)[0] == true);
    nc_release(not_result);
    
    nc_release(a);
    nc_release(b);
    
    printf("  Logical operation tests passed!\n");
}

void test_statistics() {
    printf("Testing statistical functions...\n");
    
    NCArray *arr = nc_arange(1.0, 11.0, 1.0, NC_FLOAT64);
    
    NCArray *all_true = nc_all(nc_full(1, (int64_t[]){3}, (bool[]){true}, NC_BOOL), NULL, 0);
    assert(((bool*)all_true->data)[0] == true);
    nc_release(all_true);
    
    NCArray *any_true = nc_any(nc_full(1, (int64_t[]){3}, (bool[]){false, false, true}, NC_BOOL), NULL, 0);
    assert(((bool*)any_true->data)[0] == true);
    nc_release(any_true);
    
    int64_t nonzero_count = nc_count_nonzero(arr);
    assert(nonzero_count == 10);
    
    bool is_fin = nc_isfinite(arr);
    assert(is_fin == true);
    
    nc_release(arr);
    
    printf("  Statistical function tests passed!\n");
}

void test_random() {
    printf("Testing random number generation...\n");
    
    nc_random_seed(42);
    
    NCArray *rand_arr = nc_random_rand(1, (int64_t[]){10});
    assert(rand_arr != NULL);
    assert(nc_size(rand_arr) == 10);
    nc_release(rand_arr);
    
    NCArray *randn_arr = nc_random_randn(1, (int64_t[]){5}, NC_FLOAT64);
    assert(randn_arr != NULL);
    nc_release(randn_arr);
    
    NCArray *randint_arr = nc_random_randint(0, 10, 1, (int64_t[]){5});
    assert(randint_arr != NULL);
    nc_release(randint_arr);
    
    printf("  Random number generation tests passed!\n");
}

void test_memory_management() {
    printf("Testing memory management...\n");
    
    NCArray *arr = nc_zeros(1, (int64_t[]){100}, NC_FLOAT64);
    assert(arr != NULL);
    assert(arr->refcount == 1);
    
    nc_retain(arr);
    assert(arr->refcount == 2);
    
    nc_release(arr);
    assert(arr->refcount == 1);
    
    nc_release(arr);
    
    NCArray *copy = nc_copy(arr);
    assert(copy != NULL);
    assert(copy->refcount == 1);
    assert(nc_size(copy) == nc_size(arr));
    nc_release(copy);
    
    printf("  Memory management tests passed!\n");
}

void test_save_load() {
    printf("Testing save/load...\n");
    
    NCArray *arr = nc_arange(1.0, 11.0, 1.0, NC_FLOAT64);
    
    NCStatus status = nc_save("/tmp/test_numc.bin", arr);
    assert(status == NC_OK);
    
    NCArray *loaded = nc_load("/tmp/test_numc.bin");
    assert(loaded != NULL);
    assert(nc_size(loaded) == nc_size(arr));
    
    bool equals = nc_content_equals(arr, loaded);
    assert(equals == true);
    
    nc_release(arr);
    nc_release(loaded);
    
    printf("  Save/load tests passed!\n");
}

int main() {
    printf("=== NumC Test Suite ===\n\n");
    
    test_creation();
    test_arithmetic();
    test_comparison();
    test_math_functions();
    test_reductions();
    test_linear_algebra();
    test_array_operations();
    test_broadcasting();
    test_logical_operations();
    test_statistics();
    test_random();
    test_memory_management();
    test_save_load();
    
    printf("\n=== All tests passed! ===\n");
    return 0;
}
