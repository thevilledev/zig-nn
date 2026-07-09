#include "cuda_wrapper.h"

#include <cuda.h>
#include <cublas_v2.h>
#include <nvrtc.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct ZigNNCUDABuffer {
    CUdeviceptr device_ptr;
    unsigned long count;
} ZigNNCUDABuffer;

typedef struct ZigNNCUDABackend {
    CUdevice device;
    CUcontext context;
    CUmodule module;
    cublasHandle_t cublas;
    CUfunction matrix_multiply;
    CUfunction matrix_add;
    CUfunction matrix_add_row_bias;
    CUfunction matrix_subtract;
    CUfunction matrix_element_wise_multiply;
    CUfunction matrix_scale;
    CUfunction matrix_transpose;
    CUfunction matrix_sum_rows;
    CUfunction matrix_extract_batch;
    CUfunction apply_linear;
    CUfunction apply_linear_derivative;
    CUfunction apply_sigmoid;
    CUfunction apply_sigmoid_derivative;
    CUfunction apply_relu;
    CUfunction apply_relu_derivative;
    CUfunction apply_tanh;
    CUfunction apply_tanh_derivative;
    CUfunction apply_swish;
    CUfunction apply_swish_derivative;
    CUfunction softmax_rows;
    CUfunction apply_glu;
    CUfunction apply_swiglu;
    CUfunction apply_gelu;
    CUfunction apply_gelu_derivative;
    CUfunction matrix_layer_norm;
    CUfunction matrix_layer_norm_backward;
    CUfunction causal_self_attention;
    CUfunction causal_attention_probabilities_backward;
    CUfunction causal_attention_input_gradients;
    char device_name[256];
} ZigNNCUDABackend;

static void cuda_copy_message(char* buffer, unsigned long buffer_len, const char* message) {
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

static void cuda_copy_driver_error(char* buffer, unsigned long buffer_len, const char* prefix, CUresult result) {
    const char* name = NULL;
    const char* description = NULL;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &description);

    char message[1024];
    snprintf(message, sizeof(message), "%s: %s (%s)", prefix, description != NULL ? description : "unknown CUDA driver error", name != NULL ? name : "unknown");
    cuda_copy_message(buffer, buffer_len, message);
}

static int cuda_check(CUresult result, char* buffer, unsigned long buffer_len, const char* prefix) {
    if (result == CUDA_SUCCESS) {
        return 1;
    }
    cuda_copy_driver_error(buffer, buffer_len, prefix, result);
    return 0;
}

static int cuda_check_nvrtc(nvrtcResult result, char* buffer, unsigned long buffer_len, const char* prefix) {
    if (result == NVRTC_SUCCESS) {
        return 1;
    }

    char message[1024];
    snprintf(message, sizeof(message), "%s: %s", prefix, nvrtcGetErrorString(result));
    cuda_copy_message(buffer, buffer_len, message);
    return 0;
}

static int cuda_compile_module(ZigNNCUDABackend* backend, const char* kernel_source, int compute_major, int compute_minor, char* error_buffer, unsigned long error_buffer_len) {
    if (kernel_source == NULL || kernel_source[0] == '\0') {
        cuda_copy_message(error_buffer, error_buffer_len, "missing CUDA kernel source");
        return 0;
    }

    nvrtcProgram program = NULL;
    nvrtcResult nvrtc_status = nvrtcCreateProgram(
        &program,
        kernel_source,
        "kernels.cu",
        0,
        NULL,
        NULL
    );
    if (!cuda_check_nvrtc(nvrtc_status, error_buffer, error_buffer_len, "NVRTC program creation failed")) {
        return 0;
    }

    char arch_option[64];
    snprintf(arch_option, sizeof(arch_option), "--gpu-architecture=compute_%d%d", compute_major, compute_minor);
    const char* options[] = {
        arch_option,
        "--std=c++11",
        "--use_fast_math",
    };

    nvrtc_status = nvrtcCompileProgram(program, 3, options);
    if (nvrtc_status != NVRTC_SUCCESS) {
        if (nvrtc_status == NVRTC_ERROR_INVALID_OPTION && compute_minor != 0) {
            nvrtcDestroyProgram(&program);
            return cuda_compile_module(backend, kernel_source, compute_major, 0, error_buffer, error_buffer_len);
        }

        size_t log_size = 0;
        nvrtcGetProgramLogSize(program, &log_size);
        if (log_size > 1) {
            char* log = (char*)malloc(log_size);
            if (log != NULL) {
                nvrtcGetProgramLog(program, log);
                cuda_copy_message(error_buffer, error_buffer_len, log);
                free(log);
            } else {
                cuda_check_nvrtc(nvrtc_status, error_buffer, error_buffer_len, "NVRTC compile failed");
            }
        } else {
            cuda_check_nvrtc(nvrtc_status, error_buffer, error_buffer_len, "NVRTC compile failed");
        }
        nvrtcDestroyProgram(&program);
        return 0;
    }

    size_t ptx_size = 0;
    if (!cuda_check_nvrtc(nvrtcGetPTXSize(program, &ptx_size), error_buffer, error_buffer_len, "NVRTC PTX size query failed")) {
        nvrtcDestroyProgram(&program);
        return 0;
    }

    char* ptx = (char*)malloc(ptx_size);
    if (ptx == NULL) {
        cuda_copy_message(error_buffer, error_buffer_len, "failed to allocate PTX buffer");
        nvrtcDestroyProgram(&program);
        return 0;
    }

    if (!cuda_check_nvrtc(nvrtcGetPTX(program, ptx), error_buffer, error_buffer_len, "NVRTC PTX extraction failed")) {
        free(ptx);
        nvrtcDestroyProgram(&program);
        return 0;
    }

    if (!cuda_check(cuModuleLoadData(&backend->module, ptx), error_buffer, error_buffer_len, "CUDA module load failed")) {
        free(ptx);
        nvrtcDestroyProgram(&program);
        return 0;
    }

    free(ptx);
    nvrtcDestroyProgram(&program);
    return 1;
}

static int cuda_get_function(CUmodule module, CUfunction* function, const char* name, char* error_buffer, unsigned long error_buffer_len) {
    CUresult result = cuModuleGetFunction(function, module, name);
    if (result == CUDA_SUCCESS) {
        return 1;
    }

    char message[512];
    snprintf(message, sizeof(message), "CUDA kernel function not found: %s", name);
    cuda_copy_message(error_buffer, error_buffer_len, message);
    return 0;
}

static int cuda_load_functions(ZigNNCUDABackend* backend, char* error_buffer, unsigned long error_buffer_len) {
    return cuda_get_function(backend->module, &backend->matrix_multiply, "matrix_multiply", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->matrix_add, "matrix_add", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->matrix_add_row_bias, "matrix_add_row_bias", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->matrix_subtract, "matrix_subtract", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->matrix_element_wise_multiply, "matrix_element_wise_multiply", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->matrix_scale, "matrix_scale", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->matrix_transpose, "matrix_transpose", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->matrix_sum_rows, "matrix_sum_rows", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->matrix_extract_batch, "matrix_extract_batch", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->apply_linear, "apply_linear", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->apply_linear_derivative, "apply_linear_derivative", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->apply_sigmoid, "apply_sigmoid", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->apply_sigmoid_derivative, "apply_sigmoid_derivative", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->apply_relu, "apply_relu", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->apply_relu_derivative, "apply_relu_derivative", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->apply_tanh, "apply_tanh", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->apply_tanh_derivative, "apply_tanh_derivative", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->apply_swish, "apply_swish", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->apply_swish_derivative, "apply_swish_derivative", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->softmax_rows, "softmax_rows", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->apply_glu, "apply_glu", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->apply_swiglu, "apply_swiglu", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->apply_gelu, "apply_gelu", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->apply_gelu_derivative, "apply_gelu_derivative", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->matrix_layer_norm, "matrix_layer_norm", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->matrix_layer_norm_backward, "matrix_layer_norm_backward", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->causal_self_attention, "causal_self_attention", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->causal_attention_probabilities_backward, "causal_attention_probabilities_backward", error_buffer, error_buffer_len) &&
        cuda_get_function(backend->module, &backend->causal_attention_input_gradients, "causal_attention_input_gradients", error_buffer, error_buffer_len);
}

static CUfunction cuda_function_for_kernel_id(ZigNNCUDABackend* backend, int kernel_id) {
    if (backend == NULL) {
        return NULL;
    }

    switch (kernel_id) {
        case ZIG_NN_CUDA_KERNEL_MATRIX_ADD:
            return backend->matrix_add;
        case ZIG_NN_CUDA_KERNEL_MATRIX_SUBTRACT:
            return backend->matrix_subtract;
        case ZIG_NN_CUDA_KERNEL_MATRIX_ELEMENT_WISE_MULTIPLY:
            return backend->matrix_element_wise_multiply;
        case ZIG_NN_CUDA_KERNEL_APPLY_SIGMOID:
            return backend->apply_sigmoid;
        case ZIG_NN_CUDA_KERNEL_APPLY_RELU:
            return backend->apply_relu;
        case ZIG_NN_CUDA_KERNEL_APPLY_TANH:
            return backend->apply_tanh;
        case ZIG_NN_CUDA_KERNEL_APPLY_SWISH:
            return backend->apply_swish;
        case ZIG_NN_CUDA_KERNEL_APPLY_GLU:
            return backend->apply_glu;
        case ZIG_NN_CUDA_KERNEL_APPLY_SWIGLU:
            return backend->apply_swiglu;
        case ZIG_NN_CUDA_KERNEL_APPLY_LINEAR:
            return backend->apply_linear;
        case ZIG_NN_CUDA_KERNEL_APPLY_LINEAR_DERIVATIVE:
            return backend->apply_linear_derivative;
        case ZIG_NN_CUDA_KERNEL_APPLY_SIGMOID_DERIVATIVE:
            return backend->apply_sigmoid_derivative;
        case ZIG_NN_CUDA_KERNEL_APPLY_RELU_DERIVATIVE:
            return backend->apply_relu_derivative;
        case ZIG_NN_CUDA_KERNEL_APPLY_TANH_DERIVATIVE:
            return backend->apply_tanh_derivative;
        case ZIG_NN_CUDA_KERNEL_APPLY_SWISH_DERIVATIVE:
            return backend->apply_swish_derivative;
        case ZIG_NN_CUDA_KERNEL_MATRIX_ADD_ROW_BIAS:
            return backend->matrix_add_row_bias;
        case ZIG_NN_CUDA_KERNEL_APPLY_GELU:
            return backend->apply_gelu;
        case ZIG_NN_CUDA_KERNEL_APPLY_GELU_DERIVATIVE:
            return backend->apply_gelu_derivative;
        default:
            return NULL;
    }
}

static int cuda_set_context(ZigNNCUDABackend* backend) {
    if (backend == NULL || backend->context == NULL) {
        return 0;
    }
    return cuCtxSetCurrent(backend->context) == CUDA_SUCCESS;
}

CUDABackendRef cuda_backend_create(const char* kernel_source, char* error_buffer, unsigned long error_buffer_len) {
    cuda_copy_message(error_buffer, error_buffer_len, NULL);

    if (!cuda_check(cuInit(0), error_buffer, error_buffer_len, "CUDA initialization failed")) {
        return NULL;
    }

    int device_count = 0;
    if (!cuda_check(cuDeviceGetCount(&device_count), error_buffer, error_buffer_len, "CUDA device count query failed")) {
        return NULL;
    }
    if (device_count <= 0) {
        cuda_copy_message(error_buffer, error_buffer_len, "no CUDA devices found");
        return NULL;
    }

    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)calloc(1, sizeof(ZigNNCUDABackend));
    if (backend == NULL) {
        cuda_copy_message(error_buffer, error_buffer_len, "failed to allocate CUDA backend");
        return NULL;
    }

    if (!cuda_check(cuDeviceGet(&backend->device, 0), error_buffer, error_buffer_len, "CUDA device selection failed")) {
        free(backend);
        return NULL;
    }
    cuDeviceGetName(backend->device_name, (int)sizeof(backend->device_name), backend->device);

    int compute_major = 0;
    int compute_minor = 0;
    if (!cuda_check(cuDeviceGetAttribute(&compute_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, backend->device), error_buffer, error_buffer_len, "CUDA compute capability major query failed")) {
        free(backend);
        return NULL;
    }
    if (!cuda_check(cuDeviceGetAttribute(&compute_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, backend->device), error_buffer, error_buffer_len, "CUDA compute capability minor query failed")) {
        free(backend);
        return NULL;
    }

#if CUDA_VERSION >= 13000
    CUctxCreateParams context_params;
    memset(&context_params, 0, sizeof(context_params));
    if (!cuda_check(cuCtxCreate(&backend->context, &context_params, 0, backend->device), error_buffer, error_buffer_len, "CUDA context creation failed")) {
        free(backend);
        return NULL;
    }
#else
    if (!cuda_check(cuCtxCreate(&backend->context, 0, backend->device), error_buffer, error_buffer_len, "CUDA context creation failed")) {
        free(backend);
        return NULL;
    }
#endif

    if (!cuda_compile_module(backend, kernel_source, compute_major, compute_minor, error_buffer, error_buffer_len)) {
        cuCtxDestroy(backend->context);
        free(backend);
        return NULL;
    }

    if (!cuda_load_functions(backend, error_buffer, error_buffer_len)) {
        cuModuleUnload(backend->module);
        cuCtxDestroy(backend->context);
        free(backend);
        return NULL;
    }

    if (cublasCreate(&backend->cublas) != CUBLAS_STATUS_SUCCESS) {
        cuda_copy_message(error_buffer, error_buffer_len, "cuBLAS handle creation failed");
        cuModuleUnload(backend->module);
        cuCtxDestroy(backend->context);
        free(backend);
        return NULL;
    }

    return (CUDABackendRef)backend;
}

void cuda_backend_destroy(CUDABackendRef backend_ref) {
    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)backend_ref;
    if (backend == NULL) {
        return;
    }

    cuda_set_context(backend);
    if (backend->cublas != NULL) {
        cublasDestroy(backend->cublas);
    }
    if (backend->module != NULL) {
        cuModuleUnload(backend->module);
    }
    if (backend->context != NULL) {
        cuCtxDestroy(backend->context);
    }
    free(backend);
}

int cuda_backend_device_name(CUDABackendRef backend_ref, char* output, unsigned long output_len) {
    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)backend_ref;
    if (backend == NULL || output == NULL || output_len == 0) {
        return 0;
    }

    cuda_copy_message(output, output_len, backend->device_name);
    return 1;
}

int cuda_backend_synchronize(CUDABackendRef backend_ref) {
    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)backend_ref;
    if (!cuda_set_context(backend)) {
        return 0;
    }
    return cuCtxSynchronize() == CUDA_SUCCESS;
}

CUDABufferRef cuda_backend_create_buffer(CUDABackendRef backend_ref, unsigned long count) {
    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)backend_ref;
    if (!cuda_set_context(backend)) {
        return NULL;
    }

    ZigNNCUDABuffer* buffer = (ZigNNCUDABuffer*)calloc(1, sizeof(ZigNNCUDABuffer));
    if (buffer == NULL) {
        return NULL;
    }

    CUresult result = cuMemAlloc(&buffer->device_ptr, count * sizeof(float));
    if (result != CUDA_SUCCESS) {
        free(buffer);
        return NULL;
    }

    buffer->count = count;
    cuMemsetD8(buffer->device_ptr, 0, count * sizeof(float));
    return (CUDABufferRef)buffer;
}

void cuda_backend_destroy_buffer(CUDABackendRef backend_ref, CUDABufferRef buffer_ref) {
    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)backend_ref;
    ZigNNCUDABuffer* buffer = (ZigNNCUDABuffer*)buffer_ref;
    if (buffer == NULL) {
        return;
    }

    if (cuda_set_context(backend) && buffer->device_ptr != 0) {
        cuMemFree(buffer->device_ptr);
    }
    free(buffer);
}

int cuda_buffer_upload_f32(CUDABackendRef backend_ref, CUDABufferRef buffer_ref, const float* source, unsigned long count) {
    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)backend_ref;
    ZigNNCUDABuffer* buffer = (ZigNNCUDABuffer*)buffer_ref;
    if (!cuda_set_context(backend) || buffer == NULL || source == NULL || count > buffer->count) {
        return 0;
    }

    return cuMemcpyHtoD(buffer->device_ptr, source, count * sizeof(float)) == CUDA_SUCCESS;
}

int cuda_buffer_download_f32(CUDABackendRef backend_ref, CUDABufferRef buffer_ref, float* destination, unsigned long count) {
    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)backend_ref;
    ZigNNCUDABuffer* buffer = (ZigNNCUDABuffer*)buffer_ref;
    if (!cuda_set_context(backend) || buffer == NULL || destination == NULL || count > buffer->count) {
        return 0;
    }

    return cuMemcpyDtoH(destination, buffer->device_ptr, count * sizeof(float)) == CUDA_SUCCESS;
}

static int cuda_launch_1d(ZigNNCUDABackend* backend, CUfunction function, unsigned int width, void** args) {
    if (!cuda_set_context(backend) || function == NULL) {
        return 0;
    }

    unsigned int block_x = 256;
    unsigned int grid_x = (width + block_x - 1) / block_x;
    if (grid_x == 0) {
        grid_x = 1;
    }

    CUresult result = cuLaunchKernel(function, grid_x, 1, 1, block_x, 1, 1, 0, NULL, args, NULL);
    if (result != CUDA_SUCCESS) {
        return 0;
    }

    return 1;
}

static int cuda_launch_2d(ZigNNCUDABackend* backend, CUfunction function, unsigned int width, unsigned int height, void** args) {
    if (!cuda_set_context(backend) || function == NULL) {
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

    CUresult result = cuLaunchKernel(function, grid_x, grid_y, 1, block_x, block_y, 1, 0, NULL, args, NULL);
    if (result != CUDA_SUCCESS) {
        return 0;
    }

    return 1;
}

int cuda_launch_matrix_multiply(CUDABackendRef backend_ref, CUDABufferRef a_ref, CUDABufferRef b_ref, CUDABufferRef result_ref, unsigned int a_rows, unsigned int a_cols, unsigned int b_cols) {
    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)backend_ref;
    ZigNNCUDABuffer* a = (ZigNNCUDABuffer*)a_ref;
    ZigNNCUDABuffer* b = (ZigNNCUDABuffer*)b_ref;
    ZigNNCUDABuffer* result = (ZigNNCUDABuffer*)result_ref;
    if (a == NULL || b == NULL || result == NULL) {
        return 0;
    }

    if (!cuda_set_context(backend) || backend->cublas == NULL) {
        return 0;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float* a_ptr = (const float*)(uintptr_t)a->device_ptr;
    const float* b_ptr = (const float*)(uintptr_t)b->device_ptr;
    float* result_ptr = (float*)(uintptr_t)result->device_ptr;

    cublasStatus_t status = cublasSgemm(
        backend->cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
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
    if (status == CUBLAS_STATUS_SUCCESS) {
        return 2;
    }

    CUdeviceptr raw_a = a->device_ptr;
    CUdeviceptr raw_b = b->device_ptr;
    CUdeviceptr raw_result = result->device_ptr;
    void* args[] = { &raw_a, &raw_b, &raw_result, &a_rows, &a_cols, &b_cols };
    return cuda_launch_2d(backend, backend->matrix_multiply, b_cols, a_rows, args);
}

int cuda_launch_binary_kernel(CUDABackendRef backend_ref, int kernel_id, CUDABufferRef a_ref, CUDABufferRef b_ref, CUDABufferRef result_ref, unsigned int rows, unsigned int cols) {
    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)backend_ref;
    ZigNNCUDABuffer* a = (ZigNNCUDABuffer*)a_ref;
    ZigNNCUDABuffer* b = (ZigNNCUDABuffer*)b_ref;
    ZigNNCUDABuffer* result = (ZigNNCUDABuffer*)result_ref;
    if (a == NULL || b == NULL || result == NULL) {
        return 0;
    }

    CUdeviceptr a_ptr = a->device_ptr;
    CUdeviceptr b_ptr = b->device_ptr;
    CUdeviceptr result_ptr = result->device_ptr;
    void* args[] = { &a_ptr, &b_ptr, &result_ptr, &rows, &cols };
    return cuda_launch_2d(backend, cuda_function_for_kernel_id(backend, kernel_id), cols, rows, args);
}

int cuda_launch_scale(CUDABackendRef backend_ref, CUDABufferRef input_ref, CUDABufferRef result_ref, float scalar, unsigned int rows, unsigned int cols) {
    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)backend_ref;
    ZigNNCUDABuffer* input = (ZigNNCUDABuffer*)input_ref;
    ZigNNCUDABuffer* result = (ZigNNCUDABuffer*)result_ref;
    if (input == NULL || result == NULL) {
        return 0;
    }

    CUdeviceptr input_ptr = input->device_ptr;
    CUdeviceptr result_ptr = result->device_ptr;
    void* args[] = { &input_ptr, &result_ptr, &scalar, &rows, &cols };
    return cuda_launch_2d(backend, backend != NULL ? backend->matrix_scale : NULL, cols, rows, args);
}

int cuda_launch_transpose(CUDABackendRef backend_ref, CUDABufferRef input_ref, CUDABufferRef result_ref, unsigned int rows, unsigned int cols) {
    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)backend_ref;
    ZigNNCUDABuffer* input = (ZigNNCUDABuffer*)input_ref;
    ZigNNCUDABuffer* result = (ZigNNCUDABuffer*)result_ref;
    if (input == NULL || result == NULL) {
        return 0;
    }

    CUdeviceptr input_ptr = input->device_ptr;
    CUdeviceptr result_ptr = result->device_ptr;
    void* args[] = { &input_ptr, &result_ptr, &rows, &cols };
    return cuda_launch_2d(backend, backend != NULL ? backend->matrix_transpose : NULL, rows, cols, args);
}

int cuda_launch_sum_rows(CUDABackendRef backend_ref, CUDABufferRef input_ref, CUDABufferRef result_ref, unsigned int rows, unsigned int cols) {
    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)backend_ref;
    ZigNNCUDABuffer* input = (ZigNNCUDABuffer*)input_ref;
    ZigNNCUDABuffer* result = (ZigNNCUDABuffer*)result_ref;
    if (input == NULL || result == NULL) {
        return 0;
    }

    CUdeviceptr input_ptr = input->device_ptr;
    CUdeviceptr result_ptr = result->device_ptr;
    void* args[] = { &input_ptr, &result_ptr, &rows, &cols };
    return cuda_launch_1d(backend, backend != NULL ? backend->matrix_sum_rows : NULL, cols, args);
}

int cuda_launch_extract_batch(CUDABackendRef backend_ref, CUDABufferRef input_ref, CUDABufferRef result_ref, unsigned int start, unsigned int batch_rows, unsigned int cols) {
    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)backend_ref;
    ZigNNCUDABuffer* input = (ZigNNCUDABuffer*)input_ref;
    ZigNNCUDABuffer* result = (ZigNNCUDABuffer*)result_ref;
    if (input == NULL || result == NULL) {
        return 0;
    }

    CUdeviceptr input_ptr = input->device_ptr;
    CUdeviceptr result_ptr = result->device_ptr;
    void* args[] = { &input_ptr, &result_ptr, &start, &batch_rows, &cols };
    return cuda_launch_2d(backend, backend != NULL ? backend->matrix_extract_batch : NULL, cols, batch_rows, args);
}

int cuda_launch_unary_kernel(CUDABackendRef backend_ref, int kernel_id, CUDABufferRef input_ref, CUDABufferRef result_ref, unsigned int size) {
    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)backend_ref;
    ZigNNCUDABuffer* input = (ZigNNCUDABuffer*)input_ref;
    ZigNNCUDABuffer* result = (ZigNNCUDABuffer*)result_ref;
    if (input == NULL || result == NULL) {
        return 0;
    }

    CUdeviceptr input_ptr = input->device_ptr;
    CUdeviceptr result_ptr = result->device_ptr;
    void* args[] = { &input_ptr, &result_ptr, &size };
    return cuda_launch_1d(backend, cuda_function_for_kernel_id(backend, kernel_id), size, args);
}

int cuda_launch_softmax_rows(CUDABackendRef backend_ref, CUDABufferRef input_ref, CUDABufferRef result_ref, unsigned int rows, unsigned int cols) {
    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)backend_ref;
    ZigNNCUDABuffer* input = (ZigNNCUDABuffer*)input_ref;
    ZigNNCUDABuffer* result = (ZigNNCUDABuffer*)result_ref;
    if (input == NULL || result == NULL) {
        return 0;
    }

    CUdeviceptr input_ptr = input->device_ptr;
    CUdeviceptr result_ptr = result->device_ptr;
    void* args[] = { &input_ptr, &result_ptr, &rows, &cols };
    return cuda_launch_1d(backend, backend != NULL ? backend->softmax_rows : NULL, rows, args);
}

int cuda_launch_gated_kernel(CUDABackendRef backend_ref, int kernel_id, CUDABufferRef linear_ref, CUDABufferRef gating_ref, CUDABufferRef result_ref, unsigned int size) {
    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)backend_ref;
    ZigNNCUDABuffer* linear = (ZigNNCUDABuffer*)linear_ref;
    ZigNNCUDABuffer* gating = (ZigNNCUDABuffer*)gating_ref;
    ZigNNCUDABuffer* result = (ZigNNCUDABuffer*)result_ref;
    if (linear == NULL || gating == NULL || result == NULL) {
        return 0;
    }

    CUdeviceptr linear_ptr = linear->device_ptr;
    CUdeviceptr gating_ptr = gating->device_ptr;
    CUdeviceptr result_ptr = result->device_ptr;
    void* args[] = { &linear_ptr, &gating_ptr, &result_ptr, &size };
    return cuda_launch_1d(backend, cuda_function_for_kernel_id(backend, kernel_id), size, args);
}

int cuda_launch_layer_norm(CUDABackendRef backend_ref, CUDABufferRef input_ref, CUDABufferRef gamma_ref, CUDABufferRef beta_ref, CUDABufferRef result_ref, unsigned int rows, unsigned int cols, float epsilon) {
    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)backend_ref;
    ZigNNCUDABuffer* input = (ZigNNCUDABuffer*)input_ref;
    ZigNNCUDABuffer* gamma = (ZigNNCUDABuffer*)gamma_ref;
    ZigNNCUDABuffer* beta = (ZigNNCUDABuffer*)beta_ref;
    ZigNNCUDABuffer* result = (ZigNNCUDABuffer*)result_ref;
    if (input == NULL || gamma == NULL || beta == NULL || result == NULL) {
        return 0;
    }

    CUdeviceptr input_ptr = input->device_ptr;
    CUdeviceptr gamma_ptr = gamma->device_ptr;
    CUdeviceptr beta_ptr = beta->device_ptr;
    CUdeviceptr result_ptr = result->device_ptr;
    void* args[] = { &input_ptr, &gamma_ptr, &beta_ptr, &result_ptr, &rows, &cols, &epsilon };
    return cuda_launch_1d(backend, backend != NULL ? backend->matrix_layer_norm : NULL, rows, args);
}

int cuda_launch_layer_norm_backward(CUDABackendRef backend_ref, CUDABufferRef input_ref, CUDABufferRef gamma_ref, CUDABufferRef output_gradient_ref, CUDABufferRef input_gradient_ref, CUDABufferRef gamma_gradient_ref, CUDABufferRef beta_gradient_ref, unsigned int rows, unsigned int cols, float epsilon) {
    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)backend_ref;
    ZigNNCUDABuffer* input = (ZigNNCUDABuffer*)input_ref;
    ZigNNCUDABuffer* gamma = (ZigNNCUDABuffer*)gamma_ref;
    ZigNNCUDABuffer* output_gradient = (ZigNNCUDABuffer*)output_gradient_ref;
    ZigNNCUDABuffer* input_gradient = (ZigNNCUDABuffer*)input_gradient_ref;
    ZigNNCUDABuffer* gamma_gradient = (ZigNNCUDABuffer*)gamma_gradient_ref;
    ZigNNCUDABuffer* beta_gradient = (ZigNNCUDABuffer*)beta_gradient_ref;
    if (input == NULL || gamma == NULL || output_gradient == NULL || input_gradient == NULL || gamma_gradient == NULL || beta_gradient == NULL) {
        return 0;
    }

    CUdeviceptr input_ptr = input->device_ptr;
    CUdeviceptr gamma_ptr = gamma->device_ptr;
    CUdeviceptr output_gradient_ptr = output_gradient->device_ptr;
    CUdeviceptr input_gradient_ptr = input_gradient->device_ptr;
    CUdeviceptr gamma_gradient_ptr = gamma_gradient->device_ptr;
    CUdeviceptr beta_gradient_ptr = beta_gradient->device_ptr;
    void* args[] = { &input_ptr, &gamma_ptr, &output_gradient_ptr, &input_gradient_ptr, &gamma_gradient_ptr, &beta_gradient_ptr, &rows, &cols, &epsilon };
    unsigned int width = rows > cols ? rows : cols;
    return cuda_launch_1d(backend, backend != NULL ? backend->matrix_layer_norm_backward : NULL, width, args);
}

int cuda_launch_causal_self_attention(CUDABackendRef backend_ref, CUDABufferRef query_ref, CUDABufferRef key_ref, CUDABufferRef value_ref, CUDABufferRef result_ref, unsigned int tokens, unsigned int channels, unsigned int heads) {
    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)backend_ref;
    ZigNNCUDABuffer* query = (ZigNNCUDABuffer*)query_ref;
    ZigNNCUDABuffer* key = (ZigNNCUDABuffer*)key_ref;
    ZigNNCUDABuffer* value = (ZigNNCUDABuffer*)value_ref;
    ZigNNCUDABuffer* result = (ZigNNCUDABuffer*)result_ref;
    if (query == NULL || key == NULL || value == NULL || result == NULL) {
        return 0;
    }

    CUdeviceptr query_ptr = query->device_ptr;
    CUdeviceptr key_ptr = key->device_ptr;
    CUdeviceptr value_ptr = value->device_ptr;
    CUdeviceptr result_ptr = result->device_ptr;
    void* args[] = { &query_ptr, &key_ptr, &value_ptr, &result_ptr, &tokens, &channels, &heads };
    return cuda_launch_2d(backend, backend != NULL ? backend->causal_self_attention : NULL, tokens, heads, args);
}

int cuda_launch_causal_attention_probabilities_backward(CUDABackendRef backend_ref, CUDABufferRef query_ref, CUDABufferRef key_ref, CUDABufferRef value_ref, CUDABufferRef output_gradient_ref, CUDABufferRef probabilities_ref, CUDABufferRef score_gradients_ref, unsigned int tokens, unsigned int channels, unsigned int heads) {
    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)backend_ref;
    ZigNNCUDABuffer* query = (ZigNNCUDABuffer*)query_ref;
    ZigNNCUDABuffer* key = (ZigNNCUDABuffer*)key_ref;
    ZigNNCUDABuffer* value = (ZigNNCUDABuffer*)value_ref;
    ZigNNCUDABuffer* output_gradient = (ZigNNCUDABuffer*)output_gradient_ref;
    ZigNNCUDABuffer* probabilities = (ZigNNCUDABuffer*)probabilities_ref;
    ZigNNCUDABuffer* score_gradients = (ZigNNCUDABuffer*)score_gradients_ref;
    if (query == NULL || key == NULL || value == NULL || output_gradient == NULL || probabilities == NULL || score_gradients == NULL) {
        return 0;
    }
    CUdeviceptr query_ptr = query->device_ptr;
    CUdeviceptr key_ptr = key->device_ptr;
    CUdeviceptr value_ptr = value->device_ptr;
    CUdeviceptr output_gradient_ptr = output_gradient->device_ptr;
    CUdeviceptr probabilities_ptr = probabilities->device_ptr;
    CUdeviceptr score_gradients_ptr = score_gradients->device_ptr;
    void* args[] = { &query_ptr, &key_ptr, &value_ptr, &output_gradient_ptr, &probabilities_ptr, &score_gradients_ptr, &tokens, &channels, &heads };
    return cuda_launch_2d(backend, backend != NULL ? backend->causal_attention_probabilities_backward : NULL, tokens, heads, args);
}

int cuda_launch_causal_attention_input_gradients(CUDABackendRef backend_ref, CUDABufferRef query_ref, CUDABufferRef key_ref, CUDABufferRef output_gradient_ref, CUDABufferRef probabilities_ref, CUDABufferRef score_gradients_ref, CUDABufferRef query_gradient_ref, CUDABufferRef key_gradient_ref, CUDABufferRef value_gradient_ref, unsigned int tokens, unsigned int channels, unsigned int heads) {
    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)backend_ref;
    ZigNNCUDABuffer* query = (ZigNNCUDABuffer*)query_ref;
    ZigNNCUDABuffer* key = (ZigNNCUDABuffer*)key_ref;
    ZigNNCUDABuffer* output_gradient = (ZigNNCUDABuffer*)output_gradient_ref;
    ZigNNCUDABuffer* probabilities = (ZigNNCUDABuffer*)probabilities_ref;
    ZigNNCUDABuffer* score_gradients = (ZigNNCUDABuffer*)score_gradients_ref;
    ZigNNCUDABuffer* query_gradient = (ZigNNCUDABuffer*)query_gradient_ref;
    ZigNNCUDABuffer* key_gradient = (ZigNNCUDABuffer*)key_gradient_ref;
    ZigNNCUDABuffer* value_gradient = (ZigNNCUDABuffer*)value_gradient_ref;
    if (query == NULL || key == NULL || output_gradient == NULL || probabilities == NULL || score_gradients == NULL || query_gradient == NULL || key_gradient == NULL || value_gradient == NULL) {
        return 0;
    }
    CUdeviceptr query_ptr = query->device_ptr;
    CUdeviceptr key_ptr = key->device_ptr;
    CUdeviceptr output_gradient_ptr = output_gradient->device_ptr;
    CUdeviceptr probabilities_ptr = probabilities->device_ptr;
    CUdeviceptr score_gradients_ptr = score_gradients->device_ptr;
    CUdeviceptr query_gradient_ptr = query_gradient->device_ptr;
    CUdeviceptr key_gradient_ptr = key_gradient->device_ptr;
    CUdeviceptr value_gradient_ptr = value_gradient->device_ptr;
    void* args[] = { &query_ptr, &key_ptr, &output_gradient_ptr, &probabilities_ptr, &score_gradients_ptr, &query_gradient_ptr, &key_gradient_ptr, &value_gradient_ptr, &tokens, &channels, &heads };
    return cuda_launch_2d(backend, backend != NULL ? backend->causal_attention_input_gradients : NULL, channels, tokens, args);
}
