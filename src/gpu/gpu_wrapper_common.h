#ifndef ZIG_NN_GPU_WRAPPER_COMMON_H
#define ZIG_NN_GPU_WRAPPER_COMMON_H

#include <stddef.h>
#include <stdint.h>

/* Computes a size product without allowing integer overflow. */
static inline int zig_nn_size_product(size_t left, size_t right, size_t* product) {
    if (product == NULL || (left != 0 && right > SIZE_MAX / left)) {
        return 0;
    }

    *product = left * right;
    return 1;
}

/*
 * Converts an f32 element count to bytes without allowing integer overflow to
 * produce an undersized device allocation or transfer. Zero is a valid byte
 * count here; buffer creation APIs apply their own non-empty policy.
 */
static inline int zig_nn_f32_byte_count(size_t count, size_t* byte_count) {
    return zig_nn_size_product(count, sizeof(float), byte_count);
}

#endif /* ZIG_NN_GPU_WRAPPER_COMMON_H */
