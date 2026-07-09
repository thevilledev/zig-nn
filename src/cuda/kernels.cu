extern "C" __global__ void matrix_multiply(
    const float* A,
    const float* B,
    float* C,
    unsigned int A_rows,
    unsigned int A_cols,
    unsigned int B_cols
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= B_cols || row >= A_rows) return;
    float sum = 0.0f;
    for (unsigned int k = 0; k < A_cols; k++) {
        sum += A[row * A_cols + k] * B[k * B_cols + col];
    }
    C[row * B_cols + col] = sum;
}

extern "C" __global__ void matrix_add(
    const float* A,
    const float* B,
    float* C,
    unsigned int rows,
    unsigned int cols
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= cols || row >= rows) return;
    unsigned int index = row * cols + col;
    C[index] = A[index] + B[index];
}

extern "C" __global__ void matrix_add_row_bias(
    const float* input,
    const float* bias,
    float* result,
    unsigned int rows,
    unsigned int cols
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= cols || row >= rows) return;
    unsigned int index = row * cols + col;
    result[index] = input[index] + bias[col];
}

extern "C" __global__ void matrix_subtract(
    const float* A,
    const float* B,
    float* C,
    unsigned int rows,
    unsigned int cols
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= cols || row >= rows) return;
    unsigned int index = row * cols + col;
    C[index] = A[index] - B[index];
}

extern "C" __global__ void matrix_element_wise_multiply(
    const float* A,
    const float* B,
    float* C,
    unsigned int rows,
    unsigned int cols
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= cols || row >= rows) return;
    unsigned int index = row * cols + col;
    C[index] = A[index] * B[index];
}

extern "C" __global__ void matrix_scale(
    const float* A,
    float* B,
    float scalar,
    unsigned int rows,
    unsigned int cols
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= cols || row >= rows) return;
    unsigned int index = row * cols + col;
    B[index] = A[index] * scalar;
}

extern "C" __global__ void matrix_transpose(
    const float* A,
    float* B,
    unsigned int A_rows,
    unsigned int A_cols
) {
    unsigned int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    if (out_col >= A_rows || out_row >= A_cols) return;
    B[out_row * A_rows + out_col] = A[out_col * A_cols + out_row];
}

extern "C" __global__ void matrix_sum_rows(
    const float* A,
    float* B,
    unsigned int rows,
    unsigned int cols
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    float sum = 0.0f;
    for (unsigned int row = 0; row < rows; row++) {
        sum += A[row * cols + col];
    }
    B[col] = sum;
}

extern "C" __global__ void matrix_extract_batch(
    const float* A,
    float* B,
    unsigned int start,
    unsigned int batch_rows,
    unsigned int cols
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= cols || row >= batch_rows) return;
    B[row * cols + col] = A[(start + row) * cols + col];
}

extern "C" __global__ void apply_linear(
    const float* A,
    float* B,
    unsigned int size
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    B[index] = A[index];
}

extern "C" __global__ void apply_linear_derivative(
    const float* A,
    float* B,
    unsigned int size
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    B[index] = 1.0f;
}

extern "C" __global__ void apply_sigmoid(
    const float* A,
    float* B,
    unsigned int size
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    B[index] = 1.0f / (1.0f + __expf(-A[index]));
}

extern "C" __global__ void apply_sigmoid_derivative(
    const float* A,
    float* B,
    unsigned int size
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    float sigmoid = 1.0f / (1.0f + __expf(-A[index]));
    B[index] = sigmoid * (1.0f - sigmoid);
}

extern "C" __global__ void apply_relu(
    const float* A,
    float* B,
    unsigned int size
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    float value = A[index];
    B[index] = value > 0.0f ? value : 0.0f;
}

extern "C" __global__ void apply_relu_derivative(
    const float* A,
    float* B,
    unsigned int size
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    B[index] = A[index] > 0.0f ? 1.0f : 0.0f;
}

extern "C" __global__ void apply_tanh(
    const float* A,
    float* B,
    unsigned int size
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    B[index] = tanhf(A[index]);
}

extern "C" __global__ void apply_tanh_derivative(
    const float* A,
    float* B,
    unsigned int size
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    float tanh_value = tanhf(A[index]);
    B[index] = 1.0f - tanh_value * tanh_value;
}

extern "C" __global__ void apply_swish(
    const float* A,
    float* B,
    unsigned int size
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    float sigmoid = 1.0f / (1.0f + __expf(-A[index]));
    B[index] = A[index] * sigmoid;
}

extern "C" __global__ void apply_swish_derivative(
    const float* A,
    float* B,
    unsigned int size
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    float value = A[index];
    float sigmoid = 1.0f / (1.0f + __expf(-value));
    B[index] = sigmoid + value * sigmoid * (1.0f - sigmoid);
}

extern "C" __global__ void apply_gelu(
    const float* input,
    float* result,
    unsigned int size
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    float x = input[index];
    const float coefficient = 0.7978845608028654f;
    float inner = coefficient * (x + 0.044715f * x * x * x);
    result[index] = 0.5f * x * (1.0f + tanhf(inner));
}

extern "C" __global__ void matrix_layer_norm(
    const float* input,
    const float* gamma,
    const float* beta,
    float* result,
    unsigned int rows,
    unsigned int cols,
    float epsilon
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    unsigned int offset = row * cols;
    float mean = 0.0f;
    for (unsigned int col = 0; col < cols; col++) mean += input[offset + col];
    mean /= (float)cols;

    float variance = 0.0f;
    for (unsigned int col = 0; col < cols; col++) {
        float centered = input[offset + col] - mean;
        variance += centered * centered;
    }
    variance /= (float)cols;
    float inverse_std = rsqrtf(variance + epsilon);
    for (unsigned int col = 0; col < cols; col++) {
        float normalized = (input[offset + col] - mean) * inverse_std;
        result[offset + col] = normalized * gamma[col] + beta[col];
    }
}

extern "C" __global__ void causal_self_attention(
    const float* query,
    const float* key,
    const float* value,
    float* result,
    unsigned int tokens,
    unsigned int channels,
    unsigned int heads
) {
    unsigned int query_position = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int head = blockIdx.y * blockDim.y + threadIdx.y;
    if (query_position >= tokens || head >= heads) return;

    unsigned int head_width = channels / heads;
    unsigned int channel_offset = head * head_width;
    float scale = rsqrtf((float)head_width);
    float max_score = -INFINITY;
    for (unsigned int key_position = 0; key_position <= query_position; key_position++) {
        float score = 0.0f;
        for (unsigned int channel = 0; channel < head_width; channel++) {
            score += query[query_position * channels + channel_offset + channel] *
                key[key_position * channels + channel_offset + channel];
        }
        max_score = fmaxf(max_score, score * scale);
    }

    float denominator = 0.0f;
    for (unsigned int key_position = 0; key_position <= query_position; key_position++) {
        float score = 0.0f;
        for (unsigned int channel = 0; channel < head_width; channel++) {
            score += query[query_position * channels + channel_offset + channel] *
                key[key_position * channels + channel_offset + channel];
        }
        denominator += __expf(score * scale - max_score);
    }

    for (unsigned int output_channel = 0; output_channel < head_width; output_channel++) {
        float output = 0.0f;
        for (unsigned int key_position = 0; key_position <= query_position; key_position++) {
            float score = 0.0f;
            for (unsigned int channel = 0; channel < head_width; channel++) {
                score += query[query_position * channels + channel_offset + channel] *
                    key[key_position * channels + channel_offset + channel];
            }
            float probability = __expf(score * scale - max_score) / denominator;
            output += probability * value[key_position * channels + channel_offset + output_channel];
        }
        result[query_position * channels + channel_offset + output_channel] = output;
    }
}

extern "C" __global__ void apply_glu(
    const float* linear,
    const float* gating,
    float* result,
    unsigned int size
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    float sigmoid = 1.0f / (1.0f + __expf(-gating[index]));
    result[index] = linear[index] * sigmoid;
}

extern "C" __global__ void apply_swiglu(
    const float* linear,
    const float* gating,
    float* result,
    unsigned int size
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    float gate = gating[index];
    float sigmoid = 1.0f / (1.0f + __expf(-gate));
    result[index] = linear[index] * gate * sigmoid;
}

extern "C" __global__ void softmax_rows(
    const float* A,
    float* result,
    unsigned int rows,
    unsigned int cols
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float max_value = -3.4028234663852886e38f;
    for (unsigned int col = 0; col < cols; col++) {
        float value = A[row * cols + col];
        max_value = value > max_value ? value : max_value;
    }
    float sum = 0.0f;
    for (unsigned int col = 0; col < cols; col++) {
        unsigned int index = row * cols + col;
        float exp_value = __expf(A[index] - max_value);
        result[index] = exp_value;
        sum += exp_value;
    }
    for (unsigned int col = 0; col < cols; col++) {
        unsigned int index = row * cols + col;
        result[index] = result[index] / sum;
    }
}
