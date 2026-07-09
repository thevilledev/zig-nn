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

kernel void apply_gelu_derivative(
    device const float* input [[buffer(0)]],
    device float* result [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint position [[thread_position_in_grid]]
) {
    if (position >= size) return;
    float x = input[position];
    const float coefficient = 0.7978845608028654f;
    float inner = coefficient * (x + 0.044715f * x * x * x);
    float tanh_inner = tanh(inner);
    float inner_derivative = coefficient * (1.0f + 3.0f * 0.044715f * x * x);
    result[position] = 0.5f * (1.0f + tanh_inner) +
        0.5f * x * (1.0f - tanh_inner * tanh_inner) * inner_derivative;
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

kernel void matrix_layer_norm_backward(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* output_gradient [[buffer(2)]],
    device float* input_gradient [[buffer(3)]],
    device float* gamma_gradient [[buffer(4)]],
    device float* beta_gradient [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& cols [[buffer(7)]],
    constant float& epsilon [[buffer(8)]],
    uint position [[thread_position_in_grid]]
) {
    if (position < rows) {
        uint offset = position * cols;
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
        float sum_gradient = 0.0f;
        float sum_gradient_normalized = 0.0f;
        for (uint col = 0; col < cols; col++) {
            float normalized = (input[offset + col] - mean) * inverse_std;
            float gradient = output_gradient[offset + col] * gamma[col];
            sum_gradient += gradient;
            sum_gradient_normalized += gradient * normalized;
        }
        for (uint col = 0; col < cols; col++) {
            float normalized = (input[offset + col] - mean) * inverse_std;
            float gradient = output_gradient[offset + col] * gamma[col];
            input_gradient[offset + col] = inverse_std / float(cols) *
                (float(cols) * gradient - sum_gradient - normalized * sum_gradient_normalized);
        }
    }

    if (position < cols) {
        float gamma_sum = 0.0f;
        float beta_sum = 0.0f;
        for (uint row = 0; row < rows; row++) {
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
            float normalized = (input[offset + position] - mean) * rsqrt(variance + epsilon);
            float gradient = output_gradient[offset + position];
            gamma_sum += gradient * normalized;
            beta_sum += gradient;
        }
        gamma_gradient[position] = gamma_sum;
        beta_gradient[position] = beta_sum;
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

kernel void causal_attention_probabilities_backward(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device const float* output_gradient [[buffer(3)]],
    device float* probabilities [[buffer(4)]],
    device float* score_gradients [[buffer(5)]],
    constant uint& tokens [[buffer(6)]],
    constant uint& channels [[buffer(7)]],
    constant uint& heads [[buffer(8)]],
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
    float weighted_probability_gradient = 0.0f;
    for (uint key_position = 0; key_position <= query_position; key_position++) {
        float score = 0.0f;
        float probability_gradient = 0.0f;
        for (uint channel = 0; channel < head_width; channel++) {
            score += query[query_position * channels + channel_offset + channel] *
                key[key_position * channels + channel_offset + channel];
            probability_gradient += output_gradient[query_position * channels + channel_offset + channel] *
                value[key_position * channels + channel_offset + channel];
        }
        float probability = exp(score * scale - max_score) / denominator;
        uint index = (head * tokens + query_position) * tokens + key_position;
        probabilities[index] = probability;
        score_gradients[index] = probability_gradient;
        weighted_probability_gradient += probability * probability_gradient;
    }
    for (uint key_position = 0; key_position < tokens; key_position++) {
        uint index = (head * tokens + query_position) * tokens + key_position;
        if (key_position <= query_position) {
            score_gradients[index] = probabilities[index] *
                (score_gradients[index] - weighted_probability_gradient);
        } else {
            probabilities[index] = 0.0f;
            score_gradients[index] = 0.0f;
        }
    }
}

kernel void causal_attention_input_gradients(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* output_gradient [[buffer(2)]],
    device const float* probabilities [[buffer(3)]],
    device const float* score_gradients [[buffer(4)]],
    device float* query_gradient [[buffer(5)]],
    device float* key_gradient [[buffer(6)]],
    device float* value_gradient [[buffer(7)]],
    constant uint& tokens [[buffer(8)]],
    constant uint& channels [[buffer(9)]],
    constant uint& heads [[buffer(10)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint channel = position.x;
    uint token = position.y;
    if (channel >= channels || token >= tokens) return;
    uint head_width = channels / heads;
    uint head = channel / head_width;
    float scale = rsqrt(float(head_width));
    float query_sum = 0.0f;
    for (uint key_position = 0; key_position <= token; key_position++) {
        uint index = (head * tokens + token) * tokens + key_position;
        query_sum += score_gradients[index] * key[key_position * channels + channel] * scale;
    }
    float key_sum = 0.0f;
    float value_sum = 0.0f;
    for (uint query_position = token; query_position < tokens; query_position++) {
        uint index = (head * tokens + query_position) * tokens + token;
        key_sum += score_gradients[index] * query[query_position * channels + channel] * scale;
        value_sum += probabilities[index] * output_gradient[query_position * channels + channel];
    }
    uint output_index = token * channels + channel;
    query_gradient[output_index] = query_sum;
    key_gradient[output_index] = key_sum;
    value_gradient[output_index] = value_sum;
}

kernel void embedding_lookup(
    device const float* table [[buffer(0)]],
    device const float* indices [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& vocabulary_size [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint channel = position.x;
    uint token = position.y;
    if (channel >= channels || token >= tokens) return;
    float raw_index = indices[token];
    if (raw_index < 0.0f || raw_index >= float(vocabulary_size) || floor(raw_index) != raw_index) {
        result[token * channels + channel] = 0.0f;
        return;
    }
    uint index = uint(raw_index);
    result[token * channels + channel] = table[index * channels + channel];
}

kernel void embedding_gradient(
    device const float* indices [[buffer(0)]],
    device const float* output_gradient [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& vocabulary_size [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint channel = position.x;
    uint vocabulary_index = position.y;
    if (channel >= channels || vocabulary_index >= vocabulary_size) return;
    float sum = 0.0f;
    for (uint token = 0; token < tokens; token++) {
        float raw_index = indices[token];
        if (raw_index >= 0.0f && floor(raw_index) == raw_index && uint(raw_index) == vocabulary_index) {
            sum += output_gradient[token * channels + channel];
        }
    }
    result[vocabulary_index * channels + channel] = sum;
}

kernel void cached_self_attention(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* key_cache [[buffer(3)]],
    device float* value_cache [[buffer(4)]],
    device float* result [[buffer(5)]],
    constant uint& position [[buffer(6)]],
    constant uint& channels [[buffer(7)]],
    constant uint& heads [[buffer(8)]],
    uint head [[thread_position_in_grid]]
) {
    if (head >= heads) return;
    uint head_width = channels / heads;
    uint offset = head * head_width;
    for (uint channel = 0; channel < head_width; channel++) {
        key_cache[position * channels + offset + channel] = key[offset + channel];
        value_cache[position * channels + offset + channel] = value[offset + channel];
    }
    float scale = rsqrt(float(head_width));
    float max_score = -INFINITY;
    for (uint key_position = 0; key_position <= position; key_position++) {
        float score = 0.0f;
        for (uint channel = 0; channel < head_width; channel++) {
            score += query[offset + channel] * key_cache[key_position * channels + offset + channel];
        }
        max_score = max(max_score, score * scale);
    }
    float denominator = 0.0f;
    for (uint key_position = 0; key_position <= position; key_position++) {
        float score = 0.0f;
        for (uint channel = 0; channel < head_width; channel++) score += query[offset + channel] * key_cache[key_position * channels + offset + channel];
        denominator += exp(score * scale - max_score);
    }
    for (uint output_channel = 0; output_channel < head_width; output_channel++) {
        float output = 0.0f;
        for (uint key_position = 0; key_position <= position; key_position++) {
            float score = 0.0f;
            for (uint channel = 0; channel < head_width; channel++) score += query[offset + channel] * key_cache[key_position * channels + offset + channel];
            output += exp(score * scale - max_score) / denominator * value_cache[key_position * channels + offset + output_channel];
        }
        result[offset + output_channel] = output;
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
