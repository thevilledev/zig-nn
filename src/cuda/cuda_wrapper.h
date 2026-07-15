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
    ZIG_NN_CUDA_KERNEL_MATRIX_ADD_ROW_BIAS = 15,
    ZIG_NN_CUDA_KERNEL_APPLY_GELU = 16,
    ZIG_NN_CUDA_KERNEL_APPLY_GELU_DERIVATIVE = 17,
};

CUDABackendRef cuda_backend_create(const char* kernel_source, char* error_buffer, unsigned long error_buffer_len);
void cuda_backend_destroy(CUDABackendRef backend_ref);
int cuda_backend_device_name(CUDABackendRef backend_ref, char* output, unsigned long output_len);
int cuda_backend_synchronize(CUDABackendRef backend_ref);

CUDABufferRef cuda_backend_create_buffer(CUDABackendRef backend_ref, unsigned long count);
void cuda_backend_destroy_buffer(CUDABackendRef backend_ref, CUDABufferRef buffer_ref);
int cuda_buffer_upload_f32(CUDABackendRef backend_ref, CUDABufferRef buffer_ref, const float* source, unsigned long count);
int cuda_buffer_download_f32(CUDABackendRef backend_ref, CUDABufferRef buffer_ref, float* destination, unsigned long count);

int cuda_launch_matrix_multiply(CUDABackendRef backend_ref, CUDABufferRef a_ref, CUDABufferRef b_ref, CUDABufferRef result_ref, unsigned int a_rows, unsigned int a_cols, unsigned int b_cols);
int cuda_launch_batched_matrix_multiply(CUDABackendRef backend_ref, CUDABufferRef a_ref, CUDABufferRef b_ref, CUDABufferRef result_ref, unsigned int batch, unsigned int a_rows, unsigned int a_cols, unsigned int b_rows, unsigned int b_cols, unsigned int transpose_a, unsigned int transpose_b);
int cuda_launch_permute_batch_heads(CUDABackendRef backend_ref, CUDABufferRef input_ref, CUDABufferRef result_ref, unsigned int batch, unsigned int tokens, unsigned int heads, unsigned int width, unsigned int split);
int cuda_launch_binary_kernel(CUDABackendRef backend_ref, int kernel_id, CUDABufferRef a_ref, CUDABufferRef b_ref, CUDABufferRef result_ref, unsigned int rows, unsigned int cols);
int cuda_launch_scale(CUDABackendRef backend_ref, CUDABufferRef input_ref, CUDABufferRef result_ref, float scalar, unsigned int rows, unsigned int cols);
int cuda_launch_optimizer_update(CUDABackendRef backend_ref, CUDABufferRef parameter_ref, CUDABufferRef gradient_ref, CUDABufferRef first_moment_ref, CUDABufferRef second_moment_ref, CUDABufferRef total_squares_ref, unsigned int kind, unsigned int size, float learning_rate, float beta1, float beta2, float epsilon, float weight_decay, float bias_correction1, float bias_correction2, float max_gradient_norm);
int cuda_launch_transpose(CUDABackendRef backend_ref, CUDABufferRef input_ref, CUDABufferRef result_ref, unsigned int rows, unsigned int cols);
int cuda_launch_sum_rows(CUDABackendRef backend_ref, CUDABufferRef input_ref, CUDABufferRef result_ref, unsigned int rows, unsigned int cols);
int cuda_launch_extract_batch(CUDABackendRef backend_ref, CUDABufferRef input_ref, CUDABufferRef result_ref, unsigned int start, unsigned int batch_rows, unsigned int cols);
int cuda_launch_unary_kernel(CUDABackendRef backend_ref, int kernel_id, CUDABufferRef input_ref, CUDABufferRef result_ref, unsigned int size);
int cuda_launch_softmax_rows(CUDABackendRef backend_ref, CUDABufferRef input_ref, CUDABufferRef result_ref, unsigned int rows, unsigned int cols);
int cuda_launch_gated_kernel(CUDABackendRef backend_ref, int kernel_id, CUDABufferRef linear_ref, CUDABufferRef gating_ref, CUDABufferRef result_ref, unsigned int size);
int cuda_launch_layer_norm(CUDABackendRef backend_ref, CUDABufferRef input_ref, CUDABufferRef gamma_ref, CUDABufferRef beta_ref, CUDABufferRef result_ref, unsigned int rows, unsigned int cols, float epsilon);
int cuda_launch_layer_norm_backward(CUDABackendRef backend_ref, CUDABufferRef input_ref, CUDABufferRef gamma_ref, CUDABufferRef output_gradient_ref, CUDABufferRef input_gradient_ref, CUDABufferRef gamma_gradient_ref, CUDABufferRef beta_gradient_ref, unsigned int rows, unsigned int cols, float epsilon);
int cuda_launch_causal_self_attention(CUDABackendRef backend_ref, CUDABufferRef query_ref, CUDABufferRef key_ref, CUDABufferRef value_ref, CUDABufferRef result_ref, unsigned int tokens, unsigned int channels, unsigned int heads);
int cuda_launch_causal_attention_probabilities_backward(CUDABackendRef backend_ref, CUDABufferRef query_ref, CUDABufferRef key_ref, CUDABufferRef value_ref, CUDABufferRef output_gradient_ref, CUDABufferRef probabilities_ref, CUDABufferRef score_gradients_ref, unsigned int tokens, unsigned int channels, unsigned int heads);
int cuda_launch_causal_attention_input_gradients(CUDABackendRef backend_ref, CUDABufferRef query_ref, CUDABufferRef key_ref, CUDABufferRef output_gradient_ref, CUDABufferRef probabilities_ref, CUDABufferRef score_gradients_ref, CUDABufferRef query_gradient_ref, CUDABufferRef key_gradient_ref, CUDABufferRef value_gradient_ref, unsigned int tokens, unsigned int channels, unsigned int heads);
int cuda_launch_embedding_lookup(CUDABackendRef backend_ref, CUDABufferRef table_ref, CUDABufferRef indices_ref, CUDABufferRef result_ref, unsigned int vocabulary_size, unsigned int tokens, unsigned int channels);
int cuda_launch_embedding_gradient(CUDABackendRef backend_ref, CUDABufferRef indices_ref, CUDABufferRef output_gradient_ref, CUDABufferRef result_ref, unsigned int vocabulary_size, unsigned int tokens, unsigned int channels);
int cuda_launch_cached_self_attention(CUDABackendRef backend_ref, CUDABufferRef query_ref, CUDABufferRef key_ref, CUDABufferRef value_ref, CUDABufferRef key_cache_ref, CUDABufferRef value_cache_ref, CUDABufferRef result_ref, unsigned int position, unsigned int channels, unsigned int heads);

#ifdef __cplusplus
}
#endif

#endif
