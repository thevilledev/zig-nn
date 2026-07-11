#include "rocm_wrapper.h"

#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <hip/hiprtc.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct ZigNNROCmBuffer {
    void* device_ptr;
    unsigned long count;
} ZigNNROCmBuffer;

typedef struct ZigNNROCmBackend {
    int device;
    hipModule_t module;
    rocblas_handle rocblas;
    hipFunction_t matrix_multiply;
    hipFunction_t matrix_add;
    hipFunction_t matrix_add_row_bias;
    hipFunction_t matrix_subtract;
    hipFunction_t matrix_element_wise_multiply;
    hipFunction_t matrix_scale;
    hipFunction_t matrix_transpose;
    hipFunction_t matrix_sum_rows;
    hipFunction_t matrix_extract_batch;
    hipFunction_t apply_linear;
    hipFunction_t apply_linear_derivative;
    hipFunction_t apply_sigmoid;
    hipFunction_t apply_sigmoid_derivative;
    hipFunction_t apply_relu;
    hipFunction_t apply_relu_derivative;
    hipFunction_t apply_tanh;
    hipFunction_t apply_tanh_derivative;
    hipFunction_t apply_swish;
    hipFunction_t apply_swish_derivative;
    hipFunction_t softmax_rows;
    hipFunction_t apply_glu;
    hipFunction_t apply_swiglu;
    hipFunction_t apply_gelu;
    hipFunction_t apply_gelu_derivative;
    hipFunction_t matrix_layer_norm;
    hipFunction_t matrix_layer_norm_backward;
    hipFunction_t causal_self_attention;
    hipFunction_t causal_attention_probabilities_backward;
    hipFunction_t causal_attention_input_gradients;
    hipFunction_t embedding_lookup;
    hipFunction_t embedding_gradient;
    hipFunction_t cached_self_attention;
    char device_name[256];
    char gcn_arch_name[128];
} ZigNNROCmBackend;

static void rocm_copy_message(char* buffer, unsigned long buffer_len, const char* message) {
    if (buffer == NULL || buffer_len == 0) {
        return;
    }

    if (message == NULL) {
        buffer[0] = '\0';
        return;
    }

    size_t copy_len = strlen(message);
    if (copy_len >= buffer_len) {
        copy_len = buffer_len - 1;
    }
    memcpy(buffer, message, copy_len);
    buffer[copy_len] = '\0';
}

static void rocm_copy_driver_error(char* buffer, unsigned long buffer_len, const char* prefix, hipError_t result) {
    char message[1024];
    snprintf(message, sizeof(message), "%s: %s", prefix, hipGetErrorString(result));
    rocm_copy_message(buffer, buffer_len, message);
}

static int rocm_check(hipError_t result, char* buffer, unsigned long buffer_len, const char* prefix) {
    if (result == hipSuccess) {
        return 1;
    }
    rocm_copy_driver_error(buffer, buffer_len, prefix, result);
    return 0;
}

static int rocm_check_hiprtc(hiprtcResult result, char* buffer, unsigned long buffer_len, const char* prefix) {
    if (result == HIPRTC_SUCCESS) {
        return 1;
    }

    char message[1024];
    snprintf(message, sizeof(message), "%s: %s", prefix, hiprtcGetErrorString(result));
    rocm_copy_message(buffer, buffer_len, message);
    return 0;
}

static int rocm_compile_module(ZigNNROCmBackend* backend, const char* kernel_source, char* error_buffer, unsigned long error_buffer_len) {
    if (kernel_source == NULL || kernel_source[0] == '\0') {
        rocm_copy_message(error_buffer, error_buffer_len, "missing ROCm kernel source");
        return 0;
    }
    if (backend == NULL || backend->gcn_arch_name[0] == '\0') {
        rocm_copy_message(error_buffer, error_buffer_len, "missing ROCm GPU architecture name");
        return 0;
    }

    hiprtcProgram program = NULL;
    hiprtcResult hiprtc_status = hiprtcCreateProgram(
        &program,
        kernel_source,
        "kernels.hip",
        0,
        NULL,
        NULL
    );
    if (!rocm_check_hiprtc(hiprtc_status, error_buffer, error_buffer_len, "HIPRTC program creation failed")) {
        return 0;
    }

    char arch_option[192];
    snprintf(arch_option, sizeof(arch_option), "--gpu-architecture=%s", backend->gcn_arch_name);
    const char* options[] = {
        arch_option,
        "--std=c++11",
    };

    hiprtc_status = hiprtcCompileProgram(program, 2, options);
    if (hiprtc_status != HIPRTC_SUCCESS) {
        size_t log_size = 0;
        hiprtcGetProgramLogSize(program, &log_size);
        if (log_size > 1) {
            char* log = (char*)malloc(log_size);
            if (log != NULL) {
                hiprtcGetProgramLog(program, log);
                rocm_copy_message(error_buffer, error_buffer_len, log);
                free(log);
            } else {
                rocm_check_hiprtc(hiprtc_status, error_buffer, error_buffer_len, "HIPRTC compile failed");
            }
        } else {
            rocm_check_hiprtc(hiprtc_status, error_buffer, error_buffer_len, "HIPRTC compile failed");
        }
        hiprtcDestroyProgram(&program);
        return 0;
    }

    size_t code_size = 0;
    if (!rocm_check_hiprtc(hiprtcGetCodeSize(program, &code_size), error_buffer, error_buffer_len, "HIPRTC code size query failed")) {
        hiprtcDestroyProgram(&program);
        return 0;
    }

    char* code = (char*)malloc(code_size);
    if (code == NULL) {
        rocm_copy_message(error_buffer, error_buffer_len, "failed to allocate ROCm code object buffer");
        hiprtcDestroyProgram(&program);
        return 0;
    }

    if (!rocm_check_hiprtc(hiprtcGetCode(program, code), error_buffer, error_buffer_len, "HIPRTC code object extraction failed")) {
        free(code);
        hiprtcDestroyProgram(&program);
        return 0;
    }

    if (!rocm_check(hipModuleLoadData(&backend->module, code), error_buffer, error_buffer_len, "ROCm module load failed")) {
        free(code);
        hiprtcDestroyProgram(&program);
        return 0;
    }

    free(code);
    hiprtcDestroyProgram(&program);
    return 1;
}

static int rocm_get_function(hipModule_t module, hipFunction_t* function, const char* name, char* error_buffer, unsigned long error_buffer_len) {
    hipError_t result = hipModuleGetFunction(function, module, name);
    if (result == hipSuccess) {
        return 1;
    }

    char message[512];
    snprintf(message, sizeof(message), "ROCm kernel function not found: %s", name);
    rocm_copy_message(error_buffer, error_buffer_len, message);
    return 0;
}

static int rocm_load_functions(ZigNNROCmBackend* backend, char* error_buffer, unsigned long error_buffer_len) {
    return rocm_get_function(backend->module, &backend->matrix_multiply, "matrix_multiply", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->matrix_add, "matrix_add", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->matrix_add_row_bias, "matrix_add_row_bias", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->matrix_subtract, "matrix_subtract", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->matrix_element_wise_multiply, "matrix_element_wise_multiply", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->matrix_scale, "matrix_scale", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->matrix_transpose, "matrix_transpose", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->matrix_sum_rows, "matrix_sum_rows", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->matrix_extract_batch, "matrix_extract_batch", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->apply_linear, "apply_linear", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->apply_linear_derivative, "apply_linear_derivative", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->apply_sigmoid, "apply_sigmoid", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->apply_sigmoid_derivative, "apply_sigmoid_derivative", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->apply_relu, "apply_relu", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->apply_relu_derivative, "apply_relu_derivative", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->apply_tanh, "apply_tanh", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->apply_tanh_derivative, "apply_tanh_derivative", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->apply_swish, "apply_swish", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->apply_swish_derivative, "apply_swish_derivative", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->softmax_rows, "softmax_rows", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->apply_glu, "apply_glu", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->apply_swiglu, "apply_swiglu", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->apply_gelu, "apply_gelu", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->apply_gelu_derivative, "apply_gelu_derivative", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->matrix_layer_norm, "matrix_layer_norm", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->matrix_layer_norm_backward, "matrix_layer_norm_backward", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->causal_self_attention, "causal_self_attention", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->causal_attention_probabilities_backward, "causal_attention_probabilities_backward", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->causal_attention_input_gradients, "causal_attention_input_gradients", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->embedding_lookup, "embedding_lookup", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->embedding_gradient, "embedding_gradient", error_buffer, error_buffer_len) &&
        rocm_get_function(backend->module, &backend->cached_self_attention, "cached_self_attention", error_buffer, error_buffer_len);
}

static hipFunction_t rocm_function_for_kernel_id(ZigNNROCmBackend* backend, int kernel_id) {
    if (backend == NULL) {
        return NULL;
    }

    switch (kernel_id) {
        case ZIG_NN_ROCM_KERNEL_MATRIX_ADD:
            return backend->matrix_add;
        case ZIG_NN_ROCM_KERNEL_MATRIX_SUBTRACT:
            return backend->matrix_subtract;
        case ZIG_NN_ROCM_KERNEL_MATRIX_ELEMENT_WISE_MULTIPLY:
            return backend->matrix_element_wise_multiply;
        case ZIG_NN_ROCM_KERNEL_APPLY_SIGMOID:
            return backend->apply_sigmoid;
        case ZIG_NN_ROCM_KERNEL_APPLY_RELU:
            return backend->apply_relu;
        case ZIG_NN_ROCM_KERNEL_APPLY_TANH:
            return backend->apply_tanh;
        case ZIG_NN_ROCM_KERNEL_APPLY_SWISH:
            return backend->apply_swish;
        case ZIG_NN_ROCM_KERNEL_APPLY_GLU:
            return backend->apply_glu;
        case ZIG_NN_ROCM_KERNEL_APPLY_SWIGLU:
            return backend->apply_swiglu;
        case ZIG_NN_ROCM_KERNEL_APPLY_LINEAR:
            return backend->apply_linear;
        case ZIG_NN_ROCM_KERNEL_APPLY_LINEAR_DERIVATIVE:
            return backend->apply_linear_derivative;
        case ZIG_NN_ROCM_KERNEL_APPLY_SIGMOID_DERIVATIVE:
            return backend->apply_sigmoid_derivative;
        case ZIG_NN_ROCM_KERNEL_APPLY_RELU_DERIVATIVE:
            return backend->apply_relu_derivative;
        case ZIG_NN_ROCM_KERNEL_APPLY_TANH_DERIVATIVE:
            return backend->apply_tanh_derivative;
        case ZIG_NN_ROCM_KERNEL_APPLY_SWISH_DERIVATIVE:
            return backend->apply_swish_derivative;
        case ZIG_NN_ROCM_KERNEL_MATRIX_ADD_ROW_BIAS:
            return backend->matrix_add_row_bias;
        case ZIG_NN_ROCM_KERNEL_APPLY_GELU:
            return backend->apply_gelu;
        case ZIG_NN_ROCM_KERNEL_APPLY_GELU_DERIVATIVE:
            return backend->apply_gelu_derivative;
        default:
            return NULL;
    }
}

static int rocm_set_context(ZigNNROCmBackend* backend) {
    if (backend == NULL) {
        return 0;
    }
    return hipSetDevice(backend->device) == hipSuccess;
}

ROCmBackendRef rocm_backend_create(const char* kernel_source, char* error_buffer, unsigned long error_buffer_len) {
    rocm_copy_message(error_buffer, error_buffer_len, NULL);

    if (!rocm_check(hipInit(0), error_buffer, error_buffer_len, "ROCm initialization failed")) {
        return NULL;
    }

    int device_count = 0;
    if (!rocm_check(hipGetDeviceCount(&device_count), error_buffer, error_buffer_len, "ROCm device count query failed")) {
        return NULL;
    }
    if (device_count <= 0) {
        rocm_copy_message(error_buffer, error_buffer_len, "no ROCm devices found");
        return NULL;
    }

    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)calloc(1, sizeof(ZigNNROCmBackend));
    if (backend == NULL) {
        rocm_copy_message(error_buffer, error_buffer_len, "failed to allocate ROCm backend");
        return NULL;
    }

    backend->device = 0;
    if (!rocm_check(hipSetDevice(backend->device), error_buffer, error_buffer_len, "ROCm device selection failed")) {
        free(backend);
        return NULL;
    }

    hipDeviceProp_t properties;
    memset(&properties, 0, sizeof(properties));
    if (!rocm_check(hipGetDeviceProperties(&properties, backend->device), error_buffer, error_buffer_len, "ROCm device properties query failed")) {
        free(backend);
        return NULL;
    }
    snprintf(backend->device_name, sizeof(backend->device_name), "%s", properties.name);
    snprintf(backend->gcn_arch_name, sizeof(backend->gcn_arch_name), "%s", properties.gcnArchName);

    if (!rocm_compile_module(backend, kernel_source, error_buffer, error_buffer_len)) {
        free(backend);
        return NULL;
    }

    if (!rocm_load_functions(backend, error_buffer, error_buffer_len)) {
        hipModuleUnload(backend->module);
        free(backend);
        return NULL;
    }

    if (rocblas_create_handle(&backend->rocblas) != rocblas_status_success) {
        rocm_copy_message(error_buffer, error_buffer_len, "rocBLAS handle creation failed");
        hipModuleUnload(backend->module);
        free(backend);
        return NULL;
    }
    rocblas_set_pointer_mode(backend->rocblas, rocblas_pointer_mode_host);

    return (ROCmBackendRef)backend;
}

void rocm_backend_destroy(ROCmBackendRef backend_ref) {
    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)backend_ref;
    if (backend == NULL) {
        return;
    }

    rocm_set_context(backend);
    if (backend->rocblas != NULL) {
        rocblas_destroy_handle(backend->rocblas);
    }
    if (backend->module != NULL) {
        hipModuleUnload(backend->module);
    }
    free(backend);
}

int rocm_backend_device_name(ROCmBackendRef backend_ref, char* output, unsigned long output_len) {
    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)backend_ref;
    if (backend == NULL || output == NULL || output_len == 0) {
        return 0;
    }

    rocm_copy_message(output, output_len, backend->device_name);
    return 1;
}

int rocm_backend_synchronize(ROCmBackendRef backend_ref) {
    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)backend_ref;
    if (!rocm_set_context(backend)) {
        return 0;
    }
    return hipDeviceSynchronize() == hipSuccess;
}

ROCmBufferRef rocm_backend_create_buffer(ROCmBackendRef backend_ref, unsigned long count) {
    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)backend_ref;
    if (!rocm_set_context(backend)) {
        return NULL;
    }

    ZigNNROCmBuffer* buffer = (ZigNNROCmBuffer*)calloc(1, sizeof(ZigNNROCmBuffer));
    if (buffer == NULL) {
        return NULL;
    }

    hipError_t result = hipMalloc(&buffer->device_ptr, count * sizeof(float));
    if (result != hipSuccess) {
        free(buffer);
        return NULL;
    }

    buffer->count = count;
    hipMemset(buffer->device_ptr, 0, count * sizeof(float));
    return (ROCmBufferRef)buffer;
}

void rocm_backend_destroy_buffer(ROCmBackendRef backend_ref, ROCmBufferRef buffer_ref) {
    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)backend_ref;
    ZigNNROCmBuffer* buffer = (ZigNNROCmBuffer*)buffer_ref;
    if (buffer == NULL) {
        return;
    }

    if (rocm_set_context(backend) && buffer->device_ptr != NULL) {
        hipFree(buffer->device_ptr);
    }
    free(buffer);
}

int rocm_buffer_upload_f32(ROCmBackendRef backend_ref, ROCmBufferRef buffer_ref, const float* source, unsigned long count) {
    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)backend_ref;
    ZigNNROCmBuffer* buffer = (ZigNNROCmBuffer*)buffer_ref;
    if (!rocm_set_context(backend) || buffer == NULL || source == NULL || count > buffer->count) {
        return 0;
    }

    return hipMemcpy(buffer->device_ptr, source, count * sizeof(float), hipMemcpyHostToDevice) == hipSuccess;
}

int rocm_buffer_download_f32(ROCmBackendRef backend_ref, ROCmBufferRef buffer_ref, float* destination, unsigned long count) {
    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)backend_ref;
    ZigNNROCmBuffer* buffer = (ZigNNROCmBuffer*)buffer_ref;
    if (!rocm_set_context(backend) || buffer == NULL || destination == NULL || count > buffer->count) {
        return 0;
    }

    return hipMemcpy(destination, buffer->device_ptr, count * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess;
}

static int rocm_launch_1d(ZigNNROCmBackend* backend, hipFunction_t function, unsigned int width, void** args) {
    if (!rocm_set_context(backend) || function == NULL) {
        return 0;
    }

    unsigned int block_x = 256;
    unsigned int grid_x = (width + block_x - 1) / block_x;
    if (grid_x == 0) {
        grid_x = 1;
    }

    hipError_t result = hipModuleLaunchKernel(function, grid_x, 1, 1, block_x, 1, 1, 0, NULL, args, NULL);
    if (result != hipSuccess) {
        return 0;
    }

    return 1;
}

static int rocm_launch_2d(ZigNNROCmBackend* backend, hipFunction_t function, unsigned int width, unsigned int height, void** args) {
    if (!rocm_set_context(backend) || function == NULL) {
        return 0;
    }

    unsigned int block_x = 16;
    unsigned int block_y = 16;
    unsigned int grid_x = (width + block_x - 1) / block_x;
    unsigned int grid_y = (height + block_y - 1) / block_y;
    if (grid_x == 0) {
        grid_x = 1;
    }
    if (grid_y == 0) {
        grid_y = 1;
    }

    hipError_t result = hipModuleLaunchKernel(function, grid_x, grid_y, 1, block_x, block_y, 1, 0, NULL, args, NULL);
    if (result != hipSuccess) {
        return 0;
    }

    return 1;
}

int rocm_launch_matrix_multiply(ROCmBackendRef backend_ref, ROCmBufferRef a_ref, ROCmBufferRef b_ref, ROCmBufferRef result_ref, unsigned int a_rows, unsigned int a_cols, unsigned int b_cols) {
    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)backend_ref;
    ZigNNROCmBuffer* a = (ZigNNROCmBuffer*)a_ref;
    ZigNNROCmBuffer* b = (ZigNNROCmBuffer*)b_ref;
    ZigNNROCmBuffer* result = (ZigNNROCmBuffer*)result_ref;
    if (a == NULL || b == NULL || result == NULL) {
        return 0;
    }

    if (!rocm_set_context(backend) || backend->rocblas == NULL) {
        return 0;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float* a_ptr = (const float*)(uintptr_t)a->device_ptr;
    const float* b_ptr = (const float*)(uintptr_t)b->device_ptr;
    float* result_ptr = (float*)(uintptr_t)result->device_ptr;

    rocblas_status status = rocblas_sgemm(
        backend->rocblas,
        rocblas_operation_none,
        rocblas_operation_none,
        (int)b_cols,
        (int)a_rows,
        (int)a_cols,
        &alpha,
        b_ptr,
        (int)b_cols,
        a_ptr,
        (int)a_cols,
        &beta,
        result_ptr,
        (int)b_cols
    );
    if (status == rocblas_status_success) {
        return 2;
    }

    void* raw_a = a->device_ptr;
    void* raw_b = b->device_ptr;
    void* raw_result = result->device_ptr;
    void* args[] = { &raw_a, &raw_b, &raw_result, &a_rows, &a_cols, &b_cols };
    return rocm_launch_2d(backend, backend->matrix_multiply, b_cols, a_rows, args);
}

int rocm_launch_binary_kernel(ROCmBackendRef backend_ref, int kernel_id, ROCmBufferRef a_ref, ROCmBufferRef b_ref, ROCmBufferRef result_ref, unsigned int rows, unsigned int cols) {
    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)backend_ref;
    ZigNNROCmBuffer* a = (ZigNNROCmBuffer*)a_ref;
    ZigNNROCmBuffer* b = (ZigNNROCmBuffer*)b_ref;
    ZigNNROCmBuffer* result = (ZigNNROCmBuffer*)result_ref;
    if (a == NULL || b == NULL || result == NULL) {
        return 0;
    }

    void* a_ptr = a->device_ptr;
    void* b_ptr = b->device_ptr;
    void* result_ptr = result->device_ptr;
    void* args[] = { &a_ptr, &b_ptr, &result_ptr, &rows, &cols };
    return rocm_launch_2d(backend, rocm_function_for_kernel_id(backend, kernel_id), cols, rows, args);
}

int rocm_launch_scale(ROCmBackendRef backend_ref, ROCmBufferRef input_ref, ROCmBufferRef result_ref, float scalar, unsigned int rows, unsigned int cols) {
    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)backend_ref;
    ZigNNROCmBuffer* input = (ZigNNROCmBuffer*)input_ref;
    ZigNNROCmBuffer* result = (ZigNNROCmBuffer*)result_ref;
    if (input == NULL || result == NULL) {
        return 0;
    }

    void* input_ptr = input->device_ptr;
    void* result_ptr = result->device_ptr;
    void* args[] = { &input_ptr, &result_ptr, &scalar, &rows, &cols };
    return rocm_launch_2d(backend, backend != NULL ? backend->matrix_scale : NULL, cols, rows, args);
}

int rocm_launch_transpose(ROCmBackendRef backend_ref, ROCmBufferRef input_ref, ROCmBufferRef result_ref, unsigned int rows, unsigned int cols) {
    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)backend_ref;
    ZigNNROCmBuffer* input = (ZigNNROCmBuffer*)input_ref;
    ZigNNROCmBuffer* result = (ZigNNROCmBuffer*)result_ref;
    if (input == NULL || result == NULL) {
        return 0;
    }

    void* input_ptr = input->device_ptr;
    void* result_ptr = result->device_ptr;
    void* args[] = { &input_ptr, &result_ptr, &rows, &cols };
    return rocm_launch_2d(backend, backend != NULL ? backend->matrix_transpose : NULL, rows, cols, args);
}

int rocm_launch_sum_rows(ROCmBackendRef backend_ref, ROCmBufferRef input_ref, ROCmBufferRef result_ref, unsigned int rows, unsigned int cols) {
    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)backend_ref;
    ZigNNROCmBuffer* input = (ZigNNROCmBuffer*)input_ref;
    ZigNNROCmBuffer* result = (ZigNNROCmBuffer*)result_ref;
    if (input == NULL || result == NULL) {
        return 0;
    }

    void* input_ptr = input->device_ptr;
    void* result_ptr = result->device_ptr;
    void* args[] = { &input_ptr, &result_ptr, &rows, &cols };
    return rocm_launch_1d(backend, backend != NULL ? backend->matrix_sum_rows : NULL, cols, args);
}

int rocm_launch_extract_batch(ROCmBackendRef backend_ref, ROCmBufferRef input_ref, ROCmBufferRef result_ref, unsigned int start, unsigned int batch_rows, unsigned int cols) {
    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)backend_ref;
    ZigNNROCmBuffer* input = (ZigNNROCmBuffer*)input_ref;
    ZigNNROCmBuffer* result = (ZigNNROCmBuffer*)result_ref;
    if (input == NULL || result == NULL) {
        return 0;
    }

    void* input_ptr = input->device_ptr;
    void* result_ptr = result->device_ptr;
    void* args[] = { &input_ptr, &result_ptr, &start, &batch_rows, &cols };
    return rocm_launch_2d(backend, backend != NULL ? backend->matrix_extract_batch : NULL, cols, batch_rows, args);
}

int rocm_launch_unary_kernel(ROCmBackendRef backend_ref, int kernel_id, ROCmBufferRef input_ref, ROCmBufferRef result_ref, unsigned int size) {
    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)backend_ref;
    ZigNNROCmBuffer* input = (ZigNNROCmBuffer*)input_ref;
    ZigNNROCmBuffer* result = (ZigNNROCmBuffer*)result_ref;
    if (input == NULL || result == NULL) {
        return 0;
    }

    void* input_ptr = input->device_ptr;
    void* result_ptr = result->device_ptr;
    void* args[] = { &input_ptr, &result_ptr, &size };
    return rocm_launch_1d(backend, rocm_function_for_kernel_id(backend, kernel_id), size, args);
}

int rocm_launch_softmax_rows(ROCmBackendRef backend_ref, ROCmBufferRef input_ref, ROCmBufferRef result_ref, unsigned int rows, unsigned int cols) {
    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)backend_ref;
    ZigNNROCmBuffer* input = (ZigNNROCmBuffer*)input_ref;
    ZigNNROCmBuffer* result = (ZigNNROCmBuffer*)result_ref;
    if (input == NULL || result == NULL) {
        return 0;
    }

    void* input_ptr = input->device_ptr;
    void* result_ptr = result->device_ptr;
    void* args[] = { &input_ptr, &result_ptr, &rows, &cols };
    return rocm_launch_1d(backend, backend != NULL ? backend->softmax_rows : NULL, rows, args);
}

int rocm_launch_gated_kernel(ROCmBackendRef backend_ref, int kernel_id, ROCmBufferRef linear_ref, ROCmBufferRef gating_ref, ROCmBufferRef result_ref, unsigned int size) {
    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)backend_ref;
    ZigNNROCmBuffer* linear = (ZigNNROCmBuffer*)linear_ref;
    ZigNNROCmBuffer* gating = (ZigNNROCmBuffer*)gating_ref;
    ZigNNROCmBuffer* result = (ZigNNROCmBuffer*)result_ref;
    if (linear == NULL || gating == NULL || result == NULL) {
        return 0;
    }

    void* linear_ptr = linear->device_ptr;
    void* gating_ptr = gating->device_ptr;
    void* result_ptr = result->device_ptr;
    void* args[] = { &linear_ptr, &gating_ptr, &result_ptr, &size };
    return rocm_launch_1d(backend, rocm_function_for_kernel_id(backend, kernel_id), size, args);
}

int rocm_launch_layer_norm(ROCmBackendRef backend_ref, ROCmBufferRef input_ref, ROCmBufferRef gamma_ref, ROCmBufferRef beta_ref, ROCmBufferRef result_ref, unsigned int rows, unsigned int cols, float epsilon) {
    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)backend_ref;
    ZigNNROCmBuffer* input = (ZigNNROCmBuffer*)input_ref;
    ZigNNROCmBuffer* gamma = (ZigNNROCmBuffer*)gamma_ref;
    ZigNNROCmBuffer* beta = (ZigNNROCmBuffer*)beta_ref;
    ZigNNROCmBuffer* result = (ZigNNROCmBuffer*)result_ref;
    if (input == NULL || gamma == NULL || beta == NULL || result == NULL) {
        return 0;
    }

    void* input_ptr = input->device_ptr;
    void* gamma_ptr = gamma->device_ptr;
    void* beta_ptr = beta->device_ptr;
    void* result_ptr = result->device_ptr;
    void* args[] = { &input_ptr, &gamma_ptr, &beta_ptr, &result_ptr, &rows, &cols, &epsilon };
    return rocm_launch_1d(backend, backend != NULL ? backend->matrix_layer_norm : NULL, rows, args);
}

int rocm_launch_layer_norm_backward(ROCmBackendRef backend_ref, ROCmBufferRef input_ref, ROCmBufferRef gamma_ref, ROCmBufferRef output_gradient_ref, ROCmBufferRef input_gradient_ref, ROCmBufferRef gamma_gradient_ref, ROCmBufferRef beta_gradient_ref, unsigned int rows, unsigned int cols, float epsilon) {
    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)backend_ref;
    ZigNNROCmBuffer* input = (ZigNNROCmBuffer*)input_ref;
    ZigNNROCmBuffer* gamma = (ZigNNROCmBuffer*)gamma_ref;
    ZigNNROCmBuffer* output_gradient = (ZigNNROCmBuffer*)output_gradient_ref;
    ZigNNROCmBuffer* input_gradient = (ZigNNROCmBuffer*)input_gradient_ref;
    ZigNNROCmBuffer* gamma_gradient = (ZigNNROCmBuffer*)gamma_gradient_ref;
    ZigNNROCmBuffer* beta_gradient = (ZigNNROCmBuffer*)beta_gradient_ref;
    if (input == NULL || gamma == NULL || output_gradient == NULL || input_gradient == NULL || gamma_gradient == NULL || beta_gradient == NULL) {
        return 0;
    }

    void* input_ptr = input->device_ptr;
    void* gamma_ptr = gamma->device_ptr;
    void* output_gradient_ptr = output_gradient->device_ptr;
    void* input_gradient_ptr = input_gradient->device_ptr;
    void* gamma_gradient_ptr = gamma_gradient->device_ptr;
    void* beta_gradient_ptr = beta_gradient->device_ptr;
    void* args[] = { &input_ptr, &gamma_ptr, &output_gradient_ptr, &input_gradient_ptr, &gamma_gradient_ptr, &beta_gradient_ptr, &rows, &cols, &epsilon };
    unsigned int width = rows > cols ? rows : cols;
    return rocm_launch_1d(backend, backend != NULL ? backend->matrix_layer_norm_backward : NULL, width, args);
}

int rocm_launch_causal_self_attention(ROCmBackendRef backend_ref, ROCmBufferRef query_ref, ROCmBufferRef key_ref, ROCmBufferRef value_ref, ROCmBufferRef result_ref, unsigned int tokens, unsigned int channels, unsigned int heads) {
    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)backend_ref;
    ZigNNROCmBuffer* query = (ZigNNROCmBuffer*)query_ref;
    ZigNNROCmBuffer* key = (ZigNNROCmBuffer*)key_ref;
    ZigNNROCmBuffer* value = (ZigNNROCmBuffer*)value_ref;
    ZigNNROCmBuffer* result = (ZigNNROCmBuffer*)result_ref;
    if (query == NULL || key == NULL || value == NULL || result == NULL) {
        return 0;
    }

    void* query_ptr = query->device_ptr;
    void* key_ptr = key->device_ptr;
    void* value_ptr = value->device_ptr;
    void* result_ptr = result->device_ptr;
    void* args[] = { &query_ptr, &key_ptr, &value_ptr, &result_ptr, &tokens, &channels, &heads };
    return rocm_launch_2d(backend, backend != NULL ? backend->causal_self_attention : NULL, tokens, heads, args);
}

int rocm_launch_causal_attention_probabilities_backward(ROCmBackendRef backend_ref, ROCmBufferRef query_ref, ROCmBufferRef key_ref, ROCmBufferRef value_ref, ROCmBufferRef output_gradient_ref, ROCmBufferRef probabilities_ref, ROCmBufferRef score_gradients_ref, unsigned int tokens, unsigned int channels, unsigned int heads) {
    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)backend_ref;
    ZigNNROCmBuffer* query = (ZigNNROCmBuffer*)query_ref;
    ZigNNROCmBuffer* key = (ZigNNROCmBuffer*)key_ref;
    ZigNNROCmBuffer* value = (ZigNNROCmBuffer*)value_ref;
    ZigNNROCmBuffer* output_gradient = (ZigNNROCmBuffer*)output_gradient_ref;
    ZigNNROCmBuffer* probabilities = (ZigNNROCmBuffer*)probabilities_ref;
    ZigNNROCmBuffer* score_gradients = (ZigNNROCmBuffer*)score_gradients_ref;
    if (query == NULL || key == NULL || value == NULL || output_gradient == NULL || probabilities == NULL || score_gradients == NULL) {
        return 0;
    }
    void* query_ptr = query->device_ptr;
    void* key_ptr = key->device_ptr;
    void* value_ptr = value->device_ptr;
    void* output_gradient_ptr = output_gradient->device_ptr;
    void* probabilities_ptr = probabilities->device_ptr;
    void* score_gradients_ptr = score_gradients->device_ptr;
    void* args[] = { &query_ptr, &key_ptr, &value_ptr, &output_gradient_ptr, &probabilities_ptr, &score_gradients_ptr, &tokens, &channels, &heads };
    return rocm_launch_2d(backend, backend != NULL ? backend->causal_attention_probabilities_backward : NULL, tokens, heads, args);
}

int rocm_launch_causal_attention_input_gradients(ROCmBackendRef backend_ref, ROCmBufferRef query_ref, ROCmBufferRef key_ref, ROCmBufferRef output_gradient_ref, ROCmBufferRef probabilities_ref, ROCmBufferRef score_gradients_ref, ROCmBufferRef query_gradient_ref, ROCmBufferRef key_gradient_ref, ROCmBufferRef value_gradient_ref, unsigned int tokens, unsigned int channels, unsigned int heads) {
    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)backend_ref;
    ZigNNROCmBuffer* query = (ZigNNROCmBuffer*)query_ref;
    ZigNNROCmBuffer* key = (ZigNNROCmBuffer*)key_ref;
    ZigNNROCmBuffer* output_gradient = (ZigNNROCmBuffer*)output_gradient_ref;
    ZigNNROCmBuffer* probabilities = (ZigNNROCmBuffer*)probabilities_ref;
    ZigNNROCmBuffer* score_gradients = (ZigNNROCmBuffer*)score_gradients_ref;
    ZigNNROCmBuffer* query_gradient = (ZigNNROCmBuffer*)query_gradient_ref;
    ZigNNROCmBuffer* key_gradient = (ZigNNROCmBuffer*)key_gradient_ref;
    ZigNNROCmBuffer* value_gradient = (ZigNNROCmBuffer*)value_gradient_ref;
    if (query == NULL || key == NULL || output_gradient == NULL || probabilities == NULL || score_gradients == NULL || query_gradient == NULL || key_gradient == NULL || value_gradient == NULL) {
        return 0;
    }
    void* query_ptr = query->device_ptr;
    void* key_ptr = key->device_ptr;
    void* output_gradient_ptr = output_gradient->device_ptr;
    void* probabilities_ptr = probabilities->device_ptr;
    void* score_gradients_ptr = score_gradients->device_ptr;
    void* query_gradient_ptr = query_gradient->device_ptr;
    void* key_gradient_ptr = key_gradient->device_ptr;
    void* value_gradient_ptr = value_gradient->device_ptr;
    void* args[] = { &query_ptr, &key_ptr, &output_gradient_ptr, &probabilities_ptr, &score_gradients_ptr, &query_gradient_ptr, &key_gradient_ptr, &value_gradient_ptr, &tokens, &channels, &heads };
    return rocm_launch_2d(backend, backend != NULL ? backend->causal_attention_input_gradients : NULL, channels, tokens, args);
}

int rocm_launch_embedding_lookup(ROCmBackendRef backend_ref, ROCmBufferRef table_ref, ROCmBufferRef indices_ref, ROCmBufferRef result_ref, unsigned int vocabulary_size, unsigned int tokens, unsigned int channels) {
    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)backend_ref;
    ZigNNROCmBuffer* table = (ZigNNROCmBuffer*)table_ref;
    ZigNNROCmBuffer* indices = (ZigNNROCmBuffer*)indices_ref;
    ZigNNROCmBuffer* result = (ZigNNROCmBuffer*)result_ref;
    if (table == NULL || indices == NULL || result == NULL) return 0;
    void* table_ptr = table->device_ptr;
    void* indices_ptr = indices->device_ptr;
    void* result_ptr = result->device_ptr;
    void* args[] = { &table_ptr, &indices_ptr, &result_ptr, &vocabulary_size, &tokens, &channels };
    return rocm_launch_2d(backend, backend != NULL ? backend->embedding_lookup : NULL, channels, tokens, args);
}

int rocm_launch_embedding_gradient(ROCmBackendRef backend_ref, ROCmBufferRef indices_ref, ROCmBufferRef output_gradient_ref, ROCmBufferRef result_ref, unsigned int vocabulary_size, unsigned int tokens, unsigned int channels) {
    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)backend_ref;
    ZigNNROCmBuffer* indices = (ZigNNROCmBuffer*)indices_ref;
    ZigNNROCmBuffer* output_gradient = (ZigNNROCmBuffer*)output_gradient_ref;
    ZigNNROCmBuffer* result = (ZigNNROCmBuffer*)result_ref;
    if (indices == NULL || output_gradient == NULL || result == NULL) return 0;
    void* indices_ptr = indices->device_ptr;
    void* output_gradient_ptr = output_gradient->device_ptr;
    void* result_ptr = result->device_ptr;
    void* args[] = { &indices_ptr, &output_gradient_ptr, &result_ptr, &vocabulary_size, &tokens, &channels };
    return rocm_launch_2d(backend, backend != NULL ? backend->embedding_gradient : NULL, channels, vocabulary_size, args);
}

int rocm_launch_cached_self_attention(ROCmBackendRef backend_ref, ROCmBufferRef query_ref, ROCmBufferRef key_ref, ROCmBufferRef value_ref, ROCmBufferRef key_cache_ref, ROCmBufferRef value_cache_ref, ROCmBufferRef result_ref, unsigned int position, unsigned int channels, unsigned int heads) {
    ZigNNROCmBackend* backend = (ZigNNROCmBackend*)backend_ref;
    ZigNNROCmBuffer* query = (ZigNNROCmBuffer*)query_ref;
    ZigNNROCmBuffer* key = (ZigNNROCmBuffer*)key_ref;
    ZigNNROCmBuffer* value = (ZigNNROCmBuffer*)value_ref;
    ZigNNROCmBuffer* key_cache = (ZigNNROCmBuffer*)key_cache_ref;
    ZigNNROCmBuffer* value_cache = (ZigNNROCmBuffer*)value_cache_ref;
    ZigNNROCmBuffer* result = (ZigNNROCmBuffer*)result_ref;
    if (query == NULL || key == NULL || value == NULL || key_cache == NULL || value_cache == NULL || result == NULL) return 0;
    void* query_ptr = query->device_ptr;
    void* key_ptr = key->device_ptr;
    void* value_ptr = value->device_ptr;
    void* key_cache_ptr = key_cache->device_ptr;
    void* value_cache_ptr = value_cache->device_ptr;
    void* result_ptr = result->device_ptr;
    void* args[] = { &query_ptr, &key_ptr, &value_ptr, &key_cache_ptr, &value_cache_ptr, &result_ptr, &position, &channels, &heads };
    return rocm_launch_1d(backend, backend != NULL ? backend->cached_self_attention : NULL, heads, args);
}
