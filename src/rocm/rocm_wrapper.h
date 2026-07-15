#ifndef ZIG_NN_ROCM_WRAPPER_H
#define ZIG_NN_ROCM_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void* ROCmBackendRef;
typedef void* ROCmBufferRef;

enum ZigNNROCmKernelId {
    ZIG_NN_ROCM_KERNEL_MATRIX_ADD = 0,
    ZIG_NN_ROCM_KERNEL_MATRIX_SUBTRACT = 1,
    ZIG_NN_ROCM_KERNEL_MATRIX_ELEMENT_WISE_MULTIPLY = 2,
    ZIG_NN_ROCM_KERNEL_APPLY_SIGMOID = 3,
    ZIG_NN_ROCM_KERNEL_APPLY_RELU = 4,
    ZIG_NN_ROCM_KERNEL_APPLY_TANH = 5,
    ZIG_NN_ROCM_KERNEL_APPLY_SWISH = 6,
    ZIG_NN_ROCM_KERNEL_APPLY_GLU = 7,
    ZIG_NN_ROCM_KERNEL_APPLY_SWIGLU = 8,
    ZIG_NN_ROCM_KERNEL_APPLY_LINEAR = 9,
    ZIG_NN_ROCM_KERNEL_APPLY_LINEAR_DERIVATIVE = 10,
    ZIG_NN_ROCM_KERNEL_APPLY_SIGMOID_DERIVATIVE = 11,
    ZIG_NN_ROCM_KERNEL_APPLY_RELU_DERIVATIVE = 12,
    ZIG_NN_ROCM_KERNEL_APPLY_TANH_DERIVATIVE = 13,
    ZIG_NN_ROCM_KERNEL_APPLY_SWISH_DERIVATIVE = 14,
    ZIG_NN_ROCM_KERNEL_MATRIX_ADD_ROW_BIAS = 15,
    ZIG_NN_ROCM_KERNEL_APPLY_GELU = 16,
    ZIG_NN_ROCM_KERNEL_APPLY_GELU_DERIVATIVE = 17,
};

ROCmBackendRef rocm_backend_create(const char* kernel_source, char* error_buffer, unsigned long error_buffer_len);
void rocm_backend_destroy(ROCmBackendRef backend_ref);
int rocm_backend_device_name(ROCmBackendRef backend_ref, char* output, unsigned long output_len);
int rocm_backend_synchronize(ROCmBackendRef backend_ref);

ROCmBufferRef rocm_backend_create_buffer(ROCmBackendRef backend_ref, unsigned long count);
void rocm_backend_destroy_buffer(ROCmBackendRef backend_ref, ROCmBufferRef buffer_ref);
int rocm_buffer_upload_f32(ROCmBackendRef backend_ref, ROCmBufferRef buffer_ref, const float* source, unsigned long count);
int rocm_buffer_download_f32(ROCmBackendRef backend_ref, ROCmBufferRef buffer_ref, float* destination, unsigned long count);

int rocm_launch_matrix_multiply(ROCmBackendRef backend_ref, ROCmBufferRef a_ref, ROCmBufferRef b_ref, ROCmBufferRef result_ref, unsigned int a_rows, unsigned int a_cols, unsigned int b_cols);
int rocm_launch_batched_matrix_multiply(ROCmBackendRef backend_ref, ROCmBufferRef a_ref, ROCmBufferRef b_ref, ROCmBufferRef result_ref, unsigned int batch, unsigned int a_rows, unsigned int a_cols, unsigned int b_rows, unsigned int b_cols, unsigned int transpose_a, unsigned int transpose_b);
int rocm_launch_permute_batch_heads(ROCmBackendRef backend_ref, ROCmBufferRef input_ref, ROCmBufferRef result_ref, unsigned int batch, unsigned int tokens, unsigned int heads, unsigned int width, unsigned int split);
int rocm_launch_binary_kernel(ROCmBackendRef backend_ref, int kernel_id, ROCmBufferRef a_ref, ROCmBufferRef b_ref, ROCmBufferRef result_ref, unsigned int rows, unsigned int cols);
int rocm_launch_scale(ROCmBackendRef backend_ref, ROCmBufferRef input_ref, ROCmBufferRef result_ref, float scalar, unsigned int rows, unsigned int cols);
int rocm_launch_optimizer_update(ROCmBackendRef backend_ref, ROCmBufferRef parameter_ref, ROCmBufferRef gradient_ref, ROCmBufferRef first_moment_ref, ROCmBufferRef second_moment_ref, ROCmBufferRef total_squares_ref, unsigned int kind, unsigned int size, float learning_rate, float beta1, float beta2, float epsilon, float weight_decay, float bias_correction1, float bias_correction2, float max_gradient_norm);
int rocm_launch_transpose(ROCmBackendRef backend_ref, ROCmBufferRef input_ref, ROCmBufferRef result_ref, unsigned int rows, unsigned int cols);
int rocm_launch_sum_rows(ROCmBackendRef backend_ref, ROCmBufferRef input_ref, ROCmBufferRef result_ref, unsigned int rows, unsigned int cols);
int rocm_launch_extract_batch(ROCmBackendRef backend_ref, ROCmBufferRef input_ref, ROCmBufferRef result_ref, unsigned int start, unsigned int batch_rows, unsigned int cols);
int rocm_launch_unary_kernel(ROCmBackendRef backend_ref, int kernel_id, ROCmBufferRef input_ref, ROCmBufferRef result_ref, unsigned int size);
int rocm_launch_softmax_rows(ROCmBackendRef backend_ref, ROCmBufferRef input_ref, ROCmBufferRef result_ref, unsigned int rows, unsigned int cols);
int rocm_launch_gated_kernel(ROCmBackendRef backend_ref, int kernel_id, ROCmBufferRef linear_ref, ROCmBufferRef gating_ref, ROCmBufferRef result_ref, unsigned int size);
int rocm_launch_layer_norm(ROCmBackendRef backend_ref, ROCmBufferRef input_ref, ROCmBufferRef gamma_ref, ROCmBufferRef beta_ref, ROCmBufferRef result_ref, unsigned int rows, unsigned int cols, float epsilon);
int rocm_launch_layer_norm_backward(ROCmBackendRef backend_ref, ROCmBufferRef input_ref, ROCmBufferRef gamma_ref, ROCmBufferRef output_gradient_ref, ROCmBufferRef input_gradient_ref, ROCmBufferRef gamma_gradient_ref, ROCmBufferRef beta_gradient_ref, unsigned int rows, unsigned int cols, float epsilon);
int rocm_launch_causal_self_attention(ROCmBackendRef backend_ref, ROCmBufferRef query_ref, ROCmBufferRef key_ref, ROCmBufferRef value_ref, ROCmBufferRef result_ref, unsigned int tokens, unsigned int channels, unsigned int heads);
int rocm_launch_causal_attention_probabilities_backward(ROCmBackendRef backend_ref, ROCmBufferRef query_ref, ROCmBufferRef key_ref, ROCmBufferRef value_ref, ROCmBufferRef output_gradient_ref, ROCmBufferRef probabilities_ref, ROCmBufferRef score_gradients_ref, unsigned int tokens, unsigned int channels, unsigned int heads);
int rocm_launch_causal_attention_input_gradients(ROCmBackendRef backend_ref, ROCmBufferRef query_ref, ROCmBufferRef key_ref, ROCmBufferRef output_gradient_ref, ROCmBufferRef probabilities_ref, ROCmBufferRef score_gradients_ref, ROCmBufferRef query_gradient_ref, ROCmBufferRef key_gradient_ref, ROCmBufferRef value_gradient_ref, unsigned int tokens, unsigned int channels, unsigned int heads);
int rocm_launch_embedding_lookup(ROCmBackendRef backend_ref, ROCmBufferRef table_ref, ROCmBufferRef indices_ref, ROCmBufferRef result_ref, unsigned int vocabulary_size, unsigned int tokens, unsigned int channels);
int rocm_launch_embedding_gradient(ROCmBackendRef backend_ref, ROCmBufferRef indices_ref, ROCmBufferRef output_gradient_ref, ROCmBufferRef result_ref, unsigned int vocabulary_size, unsigned int tokens, unsigned int channels);
int rocm_launch_cached_self_attention(ROCmBackendRef backend_ref, ROCmBufferRef query_ref, ROCmBufferRef key_ref, ROCmBufferRef value_ref, ROCmBufferRef key_cache_ref, ROCmBufferRef value_cache_ref, ROCmBufferRef result_ref, unsigned int position, unsigned int channels, unsigned int heads);

#ifdef __cplusplus
}
#endif

#endif
