#ifndef ZIG_NN_CUDA_WRAPPER_H
#define ZIG_NN_CUDA_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void* CUDABackendRef;
typedef void* CUDABufferRef;

enum ZigNNCudaKernelId {
    ZIG_NN_CUDA_KERNEL_MATRIX_ADD = 0,
    ZIG_NN_CUDA_KERNEL_MATRIX_SUBTRACT = 1,
    ZIG_NN_CUDA_KERNEL_MATRIX_ELEMENT_WISE_MULTIPLY = 2,
    ZIG_NN_CUDA_KERNEL_APPLY_SIGMOID = 3,
    ZIG_NN_CUDA_KERNEL_APPLY_RELU = 4,
    ZIG_NN_CUDA_KERNEL_APPLY_TANH = 5,
    ZIG_NN_CUDA_KERNEL_APPLY_SWISH = 6,
    ZIG_NN_CUDA_KERNEL_APPLY_GLU = 7,
    ZIG_NN_CUDA_KERNEL_APPLY_SWIGLU = 8,
    ZIG_NN_CUDA_KERNEL_APPLY_LINEAR = 9,
    ZIG_NN_CUDA_KERNEL_APPLY_LINEAR_DERIVATIVE = 10,
    ZIG_NN_CUDA_KERNEL_APPLY_SIGMOID_DERIVATIVE = 11,
    ZIG_NN_CUDA_KERNEL_APPLY_RELU_DERIVATIVE = 12,
    ZIG_NN_CUDA_KERNEL_APPLY_TANH_DERIVATIVE = 13,
    ZIG_NN_CUDA_KERNEL_APPLY_SWISH_DERIVATIVE = 14,
};

CUDABackendRef cuda_backend_create(char* error_buffer, unsigned long error_buffer_len);
void cuda_backend_destroy(CUDABackendRef backend_ref);
int cuda_backend_device_name(CUDABackendRef backend_ref, char* output, unsigned long output_len);

CUDABufferRef cuda_backend_create_buffer(CUDABackendRef backend_ref, unsigned long count);
void cuda_backend_destroy_buffer(CUDABackendRef backend_ref, CUDABufferRef buffer_ref);
int cuda_buffer_upload_f64(CUDABackendRef backend_ref, CUDABufferRef buffer_ref, const double* source, unsigned long count);
int cuda_buffer_download_f64(CUDABackendRef backend_ref, CUDABufferRef buffer_ref, double* destination, unsigned long count);

int cuda_launch_matrix_multiply(CUDABackendRef backend_ref, CUDABufferRef a_ref, CUDABufferRef b_ref, CUDABufferRef result_ref, unsigned int a_rows, unsigned int a_cols, unsigned int b_cols);
int cuda_launch_binary_kernel(CUDABackendRef backend_ref, int kernel_id, CUDABufferRef a_ref, CUDABufferRef b_ref, CUDABufferRef result_ref, unsigned int rows, unsigned int cols);
int cuda_launch_scale(CUDABackendRef backend_ref, CUDABufferRef input_ref, CUDABufferRef result_ref, float scalar, unsigned int rows, unsigned int cols);
int cuda_launch_transpose(CUDABackendRef backend_ref, CUDABufferRef input_ref, CUDABufferRef result_ref, unsigned int rows, unsigned int cols);
int cuda_launch_sum_rows(CUDABackendRef backend_ref, CUDABufferRef input_ref, CUDABufferRef result_ref, unsigned int rows, unsigned int cols);
int cuda_launch_extract_batch(CUDABackendRef backend_ref, CUDABufferRef input_ref, CUDABufferRef result_ref, unsigned int start, unsigned int batch_rows, unsigned int cols);
int cuda_launch_unary_kernel(CUDABackendRef backend_ref, int kernel_id, CUDABufferRef input_ref, CUDABufferRef result_ref, unsigned int size);
int cuda_launch_softmax_rows(CUDABackendRef backend_ref, CUDABufferRef input_ref, CUDABufferRef result_ref, unsigned int rows, unsigned int cols);
int cuda_launch_gated_kernel(CUDABackendRef backend_ref, int kernel_id, CUDABufferRef linear_ref, CUDABufferRef gating_ref, CUDABufferRef result_ref, unsigned int size);

#ifdef __cplusplus
}
#endif

#endif
