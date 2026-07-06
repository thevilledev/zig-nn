#include "cuda_wrapper.h"

#include <cuda.h>
#include <nvrtc.h>
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
    CUfunction matrix_multiply;
    CUfunction matrix_add;
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
    char device_name[256];
} ZigNNCUDABackend;

static const char* cuda_kernel_source =
"extern \"C\" __global__ void matrix_multiply(\n"
"    const float* A,\n"
"    const float* B,\n"
"    float* C,\n"
"    unsigned int A_rows,\n"
"    unsigned int A_cols,\n"
"    unsigned int B_cols\n"
") {\n"
"    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;\n"
"    if (col >= B_cols || row >= A_rows) return;\n"
"    float sum = 0.0f;\n"
"    for (unsigned int k = 0; k < A_cols; k++) {\n"
"        sum += A[row * A_cols + k] * B[k * B_cols + col];\n"
"    }\n"
"    C[row * B_cols + col] = sum;\n"
"}\n"
"\n"
"extern \"C\" __global__ void matrix_add(\n"
"    const float* A,\n"
"    const float* B,\n"
"    float* C,\n"
"    unsigned int rows,\n"
"    unsigned int cols\n"
") {\n"
"    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;\n"
"    if (col >= cols || row >= rows) return;\n"
"    unsigned int index = row * cols + col;\n"
"    C[index] = A[index] + B[index];\n"
"}\n"
"\n"
"extern \"C\" __global__ void matrix_subtract(\n"
"    const float* A,\n"
"    const float* B,\n"
"    float* C,\n"
"    unsigned int rows,\n"
"    unsigned int cols\n"
") {\n"
"    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;\n"
"    if (col >= cols || row >= rows) return;\n"
"    unsigned int index = row * cols + col;\n"
"    C[index] = A[index] - B[index];\n"
"}\n"
"\n"
"extern \"C\" __global__ void matrix_element_wise_multiply(\n"
"    const float* A,\n"
"    const float* B,\n"
"    float* C,\n"
"    unsigned int rows,\n"
"    unsigned int cols\n"
") {\n"
"    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;\n"
"    if (col >= cols || row >= rows) return;\n"
"    unsigned int index = row * cols + col;\n"
"    C[index] = A[index] * B[index];\n"
"}\n"
"\n"
"extern \"C\" __global__ void matrix_scale(\n"
"    const float* A,\n"
"    float* B,\n"
"    float scalar,\n"
"    unsigned int rows,\n"
"    unsigned int cols\n"
") {\n"
"    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;\n"
"    if (col >= cols || row >= rows) return;\n"
"    unsigned int index = row * cols + col;\n"
"    B[index] = A[index] * scalar;\n"
"}\n"
"\n"
"extern \"C\" __global__ void matrix_transpose(\n"
"    const float* A,\n"
"    float* B,\n"
"    unsigned int A_rows,\n"
"    unsigned int A_cols\n"
") {\n"
"    unsigned int out_col = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    unsigned int out_row = blockIdx.y * blockDim.y + threadIdx.y;\n"
"    if (out_col >= A_rows || out_row >= A_cols) return;\n"
"    B[out_row * A_rows + out_col] = A[out_col * A_cols + out_row];\n"
"}\n"
"\n"
"extern \"C\" __global__ void matrix_sum_rows(\n"
"    const float* A,\n"
"    float* B,\n"
"    unsigned int rows,\n"
"    unsigned int cols\n"
") {\n"
"    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (col >= cols) return;\n"
"    float sum = 0.0f;\n"
"    for (unsigned int row = 0; row < rows; row++) {\n"
"        sum += A[row * cols + col];\n"
"    }\n"
"    B[col] = sum;\n"
"}\n"
"\n"
"extern \"C\" __global__ void matrix_extract_batch(\n"
"    const float* A,\n"
"    float* B,\n"
"    unsigned int start,\n"
"    unsigned int batch_rows,\n"
"    unsigned int cols\n"
") {\n"
"    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;\n"
"    if (col >= cols || row >= batch_rows) return;\n"
"    B[row * cols + col] = A[(start + row) * cols + col];\n"
"}\n"
"\n"
"extern \"C\" __global__ void apply_linear(\n"
"    const float* A,\n"
"    float* B,\n"
"    unsigned int size\n"
") {\n"
"    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (index >= size) return;\n"
"    B[index] = A[index];\n"
"}\n"
"\n"
"extern \"C\" __global__ void apply_linear_derivative(\n"
"    const float* A,\n"
"    float* B,\n"
"    unsigned int size\n"
") {\n"
"    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (index >= size) return;\n"
"    B[index] = 1.0f;\n"
"}\n"
"\n"
"extern \"C\" __global__ void apply_sigmoid(\n"
"    const float* A,\n"
"    float* B,\n"
"    unsigned int size\n"
") {\n"
"    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (index >= size) return;\n"
"    B[index] = 1.0f / (1.0f + __expf(-A[index]));\n"
"}\n"
"\n"
"extern \"C\" __global__ void apply_sigmoid_derivative(\n"
"    const float* A,\n"
"    float* B,\n"
"    unsigned int size\n"
") {\n"
"    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (index >= size) return;\n"
"    float sigmoid = 1.0f / (1.0f + __expf(-A[index]));\n"
"    B[index] = sigmoid * (1.0f - sigmoid);\n"
"}\n"
"\n"
"extern \"C\" __global__ void apply_relu(\n"
"    const float* A,\n"
"    float* B,\n"
"    unsigned int size\n"
") {\n"
"    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (index >= size) return;\n"
"    float value = A[index];\n"
"    B[index] = value > 0.0f ? value : 0.0f;\n"
"}\n"
"\n"
"extern \"C\" __global__ void apply_relu_derivative(\n"
"    const float* A,\n"
"    float* B,\n"
"    unsigned int size\n"
") {\n"
"    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (index >= size) return;\n"
"    B[index] = A[index] > 0.0f ? 1.0f : 0.0f;\n"
"}\n"
"\n"
"extern \"C\" __global__ void apply_tanh(\n"
"    const float* A,\n"
"    float* B,\n"
"    unsigned int size\n"
") {\n"
"    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (index >= size) return;\n"
"    B[index] = tanhf(A[index]);\n"
"}\n"
"\n"
"extern \"C\" __global__ void apply_tanh_derivative(\n"
"    const float* A,\n"
"    float* B,\n"
"    unsigned int size\n"
") {\n"
"    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (index >= size) return;\n"
"    float tanh_value = tanhf(A[index]);\n"
"    B[index] = 1.0f - tanh_value * tanh_value;\n"
"}\n"
"\n"
"extern \"C\" __global__ void apply_swish(\n"
"    const float* A,\n"
"    float* B,\n"
"    unsigned int size\n"
") {\n"
"    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (index >= size) return;\n"
"    float sigmoid = 1.0f / (1.0f + __expf(-A[index]));\n"
"    B[index] = A[index] * sigmoid;\n"
"}\n"
"\n"
"extern \"C\" __global__ void apply_swish_derivative(\n"
"    const float* A,\n"
"    float* B,\n"
"    unsigned int size\n"
") {\n"
"    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (index >= size) return;\n"
"    float value = A[index];\n"
"    float sigmoid = 1.0f / (1.0f + __expf(-value));\n"
"    B[index] = sigmoid + value * sigmoid * (1.0f - sigmoid);\n"
"}\n"
"\n"
"extern \"C\" __global__ void apply_glu(\n"
"    const float* linear,\n"
"    const float* gating,\n"
"    float* result,\n"
"    unsigned int size\n"
") {\n"
"    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (index >= size) return;\n"
"    float sigmoid = 1.0f / (1.0f + __expf(-gating[index]));\n"
"    result[index] = linear[index] * sigmoid;\n"
"}\n"
"\n"
"extern \"C\" __global__ void apply_swiglu(\n"
"    const float* linear,\n"
"    const float* gating,\n"
"    float* result,\n"
"    unsigned int size\n"
") {\n"
"    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (index >= size) return;\n"
"    float gate = gating[index];\n"
"    float sigmoid = 1.0f / (1.0f + __expf(-gate));\n"
"    result[index] = linear[index] * gate * sigmoid;\n"
"}\n"
"\n"
"extern \"C\" __global__ void softmax_rows(\n"
"    const float* A,\n"
"    float* result,\n"
"    unsigned int rows,\n"
"    unsigned int cols\n"
") {\n"
"    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (row >= rows) return;\n"
"    float max_value = -3.4028234663852886e38f;\n"
"    for (unsigned int col = 0; col < cols; col++) {\n"
"        float value = A[row * cols + col];\n"
"        max_value = value > max_value ? value : max_value;\n"
"    }\n"
"    float sum = 0.0f;\n"
"    for (unsigned int col = 0; col < cols; col++) {\n"
"        unsigned int index = row * cols + col;\n"
"        float exp_value = __expf(A[index] - max_value);\n"
"        result[index] = exp_value;\n"
"        sum += exp_value;\n"
"    }\n"
"    for (unsigned int col = 0; col < cols; col++) {\n"
"        unsigned int index = row * cols + col;\n"
"        result[index] = result[index] / sum;\n"
"    }\n"
"}\n";

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

static int cuda_compile_module(ZigNNCUDABackend* backend, int compute_major, int compute_minor, char* error_buffer, unsigned long error_buffer_len) {
    nvrtcProgram program = NULL;
    nvrtcResult nvrtc_status = nvrtcCreateProgram(
        &program,
        cuda_kernel_source,
        "zig_nn_cuda_kernels.cu",
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
            return cuda_compile_module(backend, compute_major, 0, error_buffer, error_buffer_len);
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
        cuda_get_function(backend->module, &backend->apply_swiglu, "apply_swiglu", error_buffer, error_buffer_len);
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

CUDABackendRef cuda_backend_create(char* error_buffer, unsigned long error_buffer_len) {
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

    if (!cuda_compile_module(backend, compute_major, compute_minor, error_buffer, error_buffer_len)) {
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

    return (CUDABackendRef)backend;
}

void cuda_backend_destroy(CUDABackendRef backend_ref) {
    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)backend_ref;
    if (backend == NULL) {
        return;
    }

    cuda_set_context(backend);
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

int cuda_buffer_upload_f64(CUDABackendRef backend_ref, CUDABufferRef buffer_ref, const double* source, unsigned long count) {
    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)backend_ref;
    ZigNNCUDABuffer* buffer = (ZigNNCUDABuffer*)buffer_ref;
    if (!cuda_set_context(backend) || buffer == NULL || source == NULL || count > buffer->count) {
        return 0;
    }

    float* converted = (float*)malloc(count * sizeof(float));
    if (converted == NULL) {
        return 0;
    }

    for (unsigned long i = 0; i < count; i++) {
        converted[i] = (float)source[i];
    }

    CUresult result = cuMemcpyHtoD(buffer->device_ptr, converted, count * sizeof(float));
    free(converted);
    return result == CUDA_SUCCESS;
}

int cuda_buffer_download_f64(CUDABackendRef backend_ref, CUDABufferRef buffer_ref, double* destination, unsigned long count) {
    ZigNNCUDABackend* backend = (ZigNNCUDABackend*)backend_ref;
    ZigNNCUDABuffer* buffer = (ZigNNCUDABuffer*)buffer_ref;
    if (!cuda_set_context(backend) || buffer == NULL || destination == NULL || count > buffer->count) {
        return 0;
    }

    float* converted = (float*)malloc(count * sizeof(float));
    if (converted == NULL) {
        return 0;
    }

    CUresult result = cuMemcpyDtoH(converted, buffer->device_ptr, count * sizeof(float));
    if (result == CUDA_SUCCESS) {
        for (unsigned long i = 0; i < count; i++) {
            destination[i] = (double)converted[i];
        }
    }

    free(converted);
    return result == CUDA_SUCCESS;
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

    return cuCtxSynchronize() == CUDA_SUCCESS;
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

    return cuCtxSynchronize() == CUDA_SUCCESS;
}

int cuda_launch_matrix_multiply(CUDABackendRef backend_ref, CUDABufferRef a_ref, CUDABufferRef b_ref, CUDABufferRef result_ref, unsigned int a_rows, unsigned int a_cols, unsigned int b_cols) {
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
    void* args[] = { &a_ptr, &b_ptr, &result_ptr, &a_rows, &a_cols, &b_cols };
    return cuda_launch_2d(backend, backend != NULL ? backend->matrix_multiply : NULL, b_cols, a_rows, args);
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
