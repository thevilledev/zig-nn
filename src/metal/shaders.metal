#include <metal_stdlib>
using namespace metal;

// Matrix multiplication compute kernel
// C = A * B
kernel void matrix_multiply(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& A_rows [[buffer(3)]],
    constant uint& A_cols [[buffer(4)]],
    constant uint& B_cols [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]
) {
    // Check if within bounds
    if (position.x >= B_cols || position.y >= A_rows) {
        return;
    }

    // Calculate result for this thread's position
    float sum = 0.0f;
    for (uint k = 0; k < A_cols; k++) {
        sum += A[position.y * A_cols + k] * B[k * B_cols + position.x];
    }

    // Write result
    C[position.y * B_cols + position.x] = sum;
}

// Element-wise addition compute kernel
// C = A + B
kernel void matrix_add(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 position [[thread_position_in_grid]]
) {
    // Check if within bounds
    if (position.x >= cols || position.y >= rows) {
        return;
    }

    // Calculate index
    uint index = position.y * cols + position.x;

    // Write result
    C[index] = A[index] + B[index];
}

kernel void matrix_add_row_bias(
    device const float* input [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 position [[thread_position_in_grid]]
) {
    if (position.x >= cols || position.y >= rows) return;
    uint index = position.y * cols + position.x;
    result[index] = input[index] + bias[position.x];
}

// Element-wise subtraction compute kernel
// C = A - B
kernel void matrix_subtract(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 position [[thread_position_in_grid]]
) {
    // Check if within bounds
    if (position.x >= cols || position.y >= rows) {
        return;
    }

    // Calculate index
    uint index = position.y * cols + position.x;

    // Write result
    C[index] = A[index] - B[index];
}

// Element-wise multiplication compute kernel (Hadamard product)
// C = A ⊙ B
kernel void matrix_element_wise_multiply(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 position [[thread_position_in_grid]]
) {
    // Check if within bounds
    if (position.x >= cols || position.y >= rows) {
        return;
    }

    // Calculate index
    uint index = position.y * cols + position.x;

    // Write result
    C[index] = A[index] * B[index];
}

// Matrix scaling compute kernel
// B = A * scalar
kernel void matrix_scale(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 position [[thread_position_in_grid]]
) {
    // Check if within bounds
    if (position.x >= cols || position.y >= rows) {
        return;
    }

    // Calculate index
    uint index = position.y * cols + position.x;

    // Write result
    B[index] = A[index] * scalar;
}

// Matrix transpose compute kernel
// B = A^T
kernel void matrix_transpose(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& A_rows [[buffer(2)]],
    constant uint& A_cols [[buffer(3)]],
    uint2 position [[thread_position_in_grid]]
) {
    // Note: threads positions align with output B dimensions
    // If A is rows x cols, B is cols x rows

    // Check if within bounds (of B)
    if (position.x >= A_rows || position.y >= A_cols) {
        return;
    }

    // Calculate indices
    uint A_index = position.x * A_cols + position.y;
    uint B_index = position.y * A_rows + position.x;

    // Write result
    B[B_index] = A[A_index];
}

// Column-wise sum
// B[j] = sum_i A[i, j]
kernel void matrix_sum_rows(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint col [[thread_position_in_grid]]
) {
    if (col >= cols) {
        return;
    }

    float sum = 0.0f;
    for (uint row = 0; row < rows; row++) {
        sum += A[row * cols + col];
    }

    B[col] = sum;
}

// Copy a contiguous row batch
// B[row, col] = A[start + row, col]
kernel void matrix_extract_batch(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& start [[buffer(2)]],
    constant uint& batch_rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint col = position.x;
    uint row = position.y;

    if (row >= batch_rows || col >= cols) {
        return;
    }

    B[row * cols + col] = A[(start + row) * cols + col];
}

// Sigmoid activation function
// B[i] = 1 / (1 + exp(-A[i]))
kernel void apply_linear(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint position [[thread_position_in_grid]]
) {
    if (position >= size) {
        return;
    }

    B[position] = A[position];
}

// Linear derivative
// B[i] = 1
kernel void apply_linear_derivative(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint position [[thread_position_in_grid]]
) {
    if (position >= size) {
        return;
    }

    B[position] = 1.0f;
}

kernel void apply_sigmoid(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint position [[thread_position_in_grid]]
) {
    // Check if within bounds
    if (position >= size) {
        return;
    }

    // Apply sigmoid
    B[position] = 1.0f / (1.0f + exp(-A[position]));
}

// Sigmoid derivative
// B[i] = sigmoid(A[i]) * (1 - sigmoid(A[i]))
kernel void apply_sigmoid_derivative(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint position [[thread_position_in_grid]]
) {
    if (position >= size) {
        return;
    }

    float sigmoid_val = 1.0f / (1.0f + exp(-A[position]));
    B[position] = sigmoid_val * (1.0f - sigmoid_val);
}

// ReLU activation function
// B[i] = max(0, A[i])
kernel void apply_relu(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint position [[thread_position_in_grid]]
) {
    // Check if within bounds
    if (position >= size) {
        return;
    }

    // Apply ReLU
    B[position] = max(0.0f, A[position]);
}

// ReLU derivative
// B[i] = A[i] > 0 ? 1 : 0
kernel void apply_relu_derivative(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint position [[thread_position_in_grid]]
) {
    if (position >= size) {
        return;
    }

    B[position] = A[position] > 0.0f ? 1.0f : 0.0f;
}

// Tanh activation function
// B[i] = tanh(A[i])
kernel void apply_tanh(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint position [[thread_position_in_grid]]
) {
    // Check if within bounds
    if (position >= size) {
        return;
    }

    // Apply tanh
    B[position] = tanh(A[position]);
}

// Tanh derivative
// B[i] = 1 - tanh(A[i])^2
kernel void apply_tanh_derivative(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint position [[thread_position_in_grid]]
) {
    if (position >= size) {
        return;
    }

    float tanh_val = tanh(A[position]);
    B[position] = 1.0f - tanh_val * tanh_val;
}

// Swish activation function
// B[i] = A[i] * sigmoid(A[i])
kernel void apply_swish(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint position [[thread_position_in_grid]]
) {
    // Check if within bounds
    if (position >= size) {
        return;
    }

    // Apply Swish
    float sigmoid_val = 1.0f / (1.0f + exp(-A[position]));
    B[position] = A[position] * sigmoid_val;
}

// Swish derivative
// B[i] = sigmoid(A[i]) + A[i] * sigmoid(A[i]) * (1 - sigmoid(A[i]))
kernel void apply_swish_derivative(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint position [[thread_position_in_grid]]
) {
    if (position >= size) {
        return;
    }

    float value = A[position];
    float sigmoid_val = 1.0f / (1.0f + exp(-value));
    B[position] = sigmoid_val + value * sigmoid_val * (1.0f - sigmoid_val);
}

kernel void apply_gelu(
    device const float* input [[buffer(0)]],
    device float* result [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint position [[thread_position_in_grid]]
) {
    if (position >= size) return;
    float x = input[position];
    const float coefficient = 0.7978845608028654f;
    float inner = coefficient * (x + 0.044715f * x * x * x);
    result[position] = 0.5f * x * (1.0f + tanh(inner));
}

kernel void matrix_layer_norm(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* result [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) return;
    uint offset = row * cols;
    float mean = 0.0f;
    for (uint col = 0; col < cols; col++) mean += input[offset + col];
    mean /= float(cols);

    float variance = 0.0f;
    for (uint col = 0; col < cols; col++) {
        float centered = input[offset + col] - mean;
        variance += centered * centered;
    }
    variance /= float(cols);
    float inverse_std = rsqrt(variance + epsilon);
    for (uint col = 0; col < cols; col++) {
        float normalized = (input[offset + col] - mean) * inverse_std;
        result[offset + col] = normalized * gamma[col] + beta[col];
    }
}

kernel void causal_self_attention(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* result [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint query_position = position.x;
    uint head = position.y;
    if (query_position >= tokens || head >= heads) return;

    uint head_width = channels / heads;
    uint channel_offset = head * head_width;
    float scale = rsqrt(float(head_width));
    float max_score = -INFINITY;
    for (uint key_position = 0; key_position <= query_position; key_position++) {
        float score = 0.0f;
        for (uint channel = 0; channel < head_width; channel++) {
            score += query[query_position * channels + channel_offset + channel] *
                key[key_position * channels + channel_offset + channel];
        }
        max_score = max(max_score, score * scale);
    }

    float denominator = 0.0f;
    for (uint key_position = 0; key_position <= query_position; key_position++) {
        float score = 0.0f;
        for (uint channel = 0; channel < head_width; channel++) {
            score += query[query_position * channels + channel_offset + channel] *
                key[key_position * channels + channel_offset + channel];
        }
        denominator += exp(score * scale - max_score);
    }

    for (uint output_channel = 0; output_channel < head_width; output_channel++) {
        float output = 0.0f;
        for (uint key_position = 0; key_position <= query_position; key_position++) {
            float score = 0.0f;
            for (uint channel = 0; channel < head_width; channel++) {
                score += query[query_position * channels + channel_offset + channel] *
                    key[key_position * channels + channel_offset + channel];
            }
            float probability = exp(score * scale - max_score) / denominator;
            output += probability * value[key_position * channels + channel_offset + output_channel];
        }
        result[query_position * channels + channel_offset + output_channel] = output;
    }
}

// Gated Linear Unit
// result[i] = linear[i] * sigmoid(gating[i])
kernel void apply_glu(
    device const float* linear [[buffer(0)]],
    device const float* gating [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint position [[thread_position_in_grid]]
) {
    if (position >= size) {
        return;
    }

    float sigmoid_val = 1.0f / (1.0f + exp(-gating[position]));
    result[position] = linear[position] * sigmoid_val;
}

// Swish Gated Linear Unit
// result[i] = linear[i] * swish(gating[i])
kernel void apply_swiglu(
    device const float* linear [[buffer(0)]],
    device const float* gating [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint position [[thread_position_in_grid]]
) {
    if (position >= size) {
        return;
    }

    float gate = gating[position];
    float sigmoid_val = 1.0f / (1.0f + exp(-gate));
    result[position] = linear[position] * gate * sigmoid_val;
}

// This is the first step of softmax: finding the maximum value in each row
kernel void softmax_find_max(
    device const float* A [[buffer(0)]],
    device float* row_maxes [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row_idx [[thread_position_in_grid]]
) {
    // Check if within bounds
    if (row_idx >= rows) {
        return;
    }

    // Find max value in this row
    float max_val = -INFINITY;
    for (uint j = 0; j < cols; j++) {
        max_val = max(max_val, A[row_idx * cols + j]);
    }

    // Store max value for this row
    row_maxes[row_idx] = max_val;
}

// This is the second step of softmax: compute exp(x_i - max) and sum
kernel void softmax_exp_and_sum(
    device const float* A [[buffer(0)]],
    device const float* row_maxes [[buffer(1)]],
    device float* exp_values [[buffer(2)]],
    device float* row_sums [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }

    float sum = 0.0f;
    for (uint col = 0; col < cols; col++) {
        uint idx = row * cols + col;
        float exp_val = exp(A[idx] - row_maxes[row]);
        exp_values[idx] = exp_val;
        sum += exp_val;
    }

    row_sums[row] = sum;
}

// This is the final step of softmax: divide by sum
kernel void softmax_normalize(
    device const float* exp_values [[buffer(0)]],
    device const float* row_sums [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint row = position.y;
    uint col = position.x;

    // Check if within bounds
    if (row >= rows || col >= cols) {
        return;
    }

    // Normalize by row sum
    uint idx = row * cols + col;
    result[idx] = exp_values[idx] / row_sums[row];
}

// Complete row-wise softmax in one simple kernel.
kernel void softmax_rows(
    device const float* A [[buffer(0)]],
    device float* result [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }

    float max_val = -INFINITY;
    for (uint col = 0; col < cols; col++) {
        max_val = max(max_val, A[row * cols + col]);
    }

    float sum = 0.0f;
    for (uint col = 0; col < cols; col++) {
        uint idx = row * cols + col;
        float exp_val = exp(A[idx] - max_val);
        result[idx] = exp_val;
        sum += exp_val;
    }

    for (uint col = 0; col < cols; col++) {
        uint idx = row * cols + col;
        result[idx] = result[idx] / sum;
    }
}
