#define ZIG_NN_CUDA_INFINITY __int_as_float(0x7f800000)

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

extern "C" __global__ void batched_matrix_multiply(
    const float* A,
    const float* B,
    float* C,
    unsigned int batch,
    unsigned int A_rows,
    unsigned int A_cols,
    unsigned int B_rows,
    unsigned int B_cols,
    unsigned int transpose_a,
    unsigned int transpose_b
) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int flat_row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int output_rows = transpose_a ? A_cols : A_rows;
    unsigned int output_cols = transpose_b ? B_rows : B_cols;
    if (col >= output_cols || flat_row >= batch * output_rows) return;
    unsigned int batch_index = flat_row / output_rows;
    unsigned int row = flat_row % output_rows;
    unsigned int inner_size = transpose_a ? A_rows : A_cols;
    unsigned int a_offset = batch_index * A_rows * A_cols;
    unsigned int b_offset = batch_index * B_rows * B_cols;
    float sum = 0.0f;
    for (unsigned int inner = 0; inner < inner_size; inner++) {
        unsigned int a_index = a_offset + (transpose_a ? inner * A_cols + row : row * A_cols + inner);
        unsigned int b_index = b_offset + (transpose_b ? col * B_cols + inner : inner * B_cols + col);
        sum += A[a_index] * B[b_index];
    }
    C[flat_row * output_cols + col] = sum;
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

extern "C" __global__ void optimizer_update(
    float* parameter,
    const float* gradient,
    float* first_moment,
    float* second_moment,
    const float* total_squares,
    unsigned int kind,
    unsigned int size,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    float max_gradient_norm
) {
    unsigned int position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= size) return;

    float clip_scale = 1.0f;
    float max_norm_squared = max_gradient_norm * max_gradient_norm;
    if (max_gradient_norm > 0.0f && total_squares[0] > max_norm_squared) {
        clip_scale = max_gradient_norm / sqrtf(total_squares[0]);
    }
    float clipped_gradient = gradient[position] * clip_scale;
    float direction = clipped_gradient;
    if (kind == 1) {
        first_moment[position] = beta1 * first_moment[position] + clipped_gradient;
        direction = first_moment[position];
    } else if (kind == 2) {
        first_moment[position] = beta1 * first_moment[position] +
            (1.0f - beta1) * clipped_gradient;
        second_moment[position] = beta2 * second_moment[position] +
            (1.0f - beta2) * clipped_gradient * clipped_gradient;
        float corrected_first = first_moment[position] / bias_correction1;
        float corrected_second = second_moment[position] / bias_correction2;
        direction = corrected_first / (sqrtf(corrected_second) + epsilon);
    }
    float decay = 1.0f - learning_rate * weight_decay;
    parameter[position] = parameter[position] * decay - learning_rate * direction;
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

extern "C" __global__ void apply_gelu_derivative(
    const float* input,
    float* result,
    unsigned int size
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    float x = input[index];
    const float coefficient = 0.7978845608028654f;
    float inner = coefficient * (x + 0.044715f * x * x * x);
    float tanh_inner = tanhf(inner);
    float inner_derivative = coefficient * (1.0f + 3.0f * 0.044715f * x * x);
    result[index] = 0.5f * (1.0f + tanh_inner) +
        0.5f * x * (1.0f - tanh_inner * tanh_inner) * inner_derivative;
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

extern "C" __global__ void matrix_layer_norm_backward(
    const float* input,
    const float* gamma,
    const float* output_gradient,
    float* input_gradient,
    float* gamma_gradient,
    float* beta_gradient,
    unsigned int rows,
    unsigned int cols,
    float epsilon
) {
    unsigned int position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position < rows) {
        unsigned int offset = position * cols;
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
        float sum_gradient = 0.0f;
        float sum_gradient_normalized = 0.0f;
        for (unsigned int col = 0; col < cols; col++) {
            float normalized = (input[offset + col] - mean) * inverse_std;
            float gradient = output_gradient[offset + col] * gamma[col];
            sum_gradient += gradient;
            sum_gradient_normalized += gradient * normalized;
        }
        for (unsigned int col = 0; col < cols; col++) {
            float normalized = (input[offset + col] - mean) * inverse_std;
            float gradient = output_gradient[offset + col] * gamma[col];
            input_gradient[offset + col] = inverse_std / (float)cols *
                ((float)cols * gradient - sum_gradient - normalized * sum_gradient_normalized);
        }
    }

    if (position < cols) {
        float gamma_sum = 0.0f;
        float beta_sum = 0.0f;
        for (unsigned int row = 0; row < rows; row++) {
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
            float normalized = (input[offset + position] - mean) * rsqrtf(variance + epsilon);
            float gradient = output_gradient[offset + position];
            gamma_sum += gradient * normalized;
            beta_sum += gradient;
        }
        gamma_gradient[position] = gamma_sum;
        beta_gradient[position] = beta_sum;
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
    float max_score = -ZIG_NN_CUDA_INFINITY;
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

extern "C" __global__ void causal_attention_probabilities_backward(
    const float* query,
    const float* key,
    const float* value,
    const float* output_gradient,
    float* probabilities,
    float* score_gradients,
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
    float max_score = -ZIG_NN_CUDA_INFINITY;
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
    float weighted_probability_gradient = 0.0f;
    for (unsigned int key_position = 0; key_position <= query_position; key_position++) {
        float score = 0.0f;
        float probability_gradient = 0.0f;
        for (unsigned int channel = 0; channel < head_width; channel++) {
            score += query[query_position * channels + channel_offset + channel] *
                key[key_position * channels + channel_offset + channel];
            probability_gradient += output_gradient[query_position * channels + channel_offset + channel] *
                value[key_position * channels + channel_offset + channel];
        }
        float probability = __expf(score * scale - max_score) / denominator;
        unsigned int index = (head * tokens + query_position) * tokens + key_position;
        probabilities[index] = probability;
        score_gradients[index] = probability_gradient;
        weighted_probability_gradient += probability * probability_gradient;
    }
    for (unsigned int key_position = 0; key_position < tokens; key_position++) {
        unsigned int index = (head * tokens + query_position) * tokens + key_position;
        if (key_position <= query_position) {
            score_gradients[index] = probabilities[index] *
                (score_gradients[index] - weighted_probability_gradient);
        } else {
            probabilities[index] = 0.0f;
            score_gradients[index] = 0.0f;
        }
    }
}

extern "C" __global__ void causal_attention_input_gradients(
    const float* query,
    const float* key,
    const float* output_gradient,
    const float* probabilities,
    const float* score_gradients,
    float* query_gradient,
    float* key_gradient,
    float* value_gradient,
    unsigned int tokens,
    unsigned int channels,
    unsigned int heads
) {
    unsigned int channel = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int token = blockIdx.y * blockDim.y + threadIdx.y;
    if (channel >= channels || token >= tokens) return;
    unsigned int head_width = channels / heads;
    unsigned int head = channel / head_width;
    float scale = rsqrtf((float)head_width);
    float query_sum = 0.0f;
    for (unsigned int key_position = 0; key_position <= token; key_position++) {
        unsigned int index = (head * tokens + token) * tokens + key_position;
        query_sum += score_gradients[index] * key[key_position * channels + channel] * scale;
    }
    float key_sum = 0.0f;
    float value_sum = 0.0f;
    for (unsigned int query_position = token; query_position < tokens; query_position++) {
        unsigned int index = (head * tokens + query_position) * tokens + token;
        key_sum += score_gradients[index] * query[query_position * channels + channel] * scale;
        value_sum += probabilities[index] * output_gradient[query_position * channels + channel];
    }
    unsigned int output_index = token * channels + channel;
    query_gradient[output_index] = query_sum;
    key_gradient[output_index] = key_sum;
    value_gradient[output_index] = value_sum;
}

extern "C" __global__ void embedding_lookup(
    const float* table,
    const float* indices,
    float* result,
    unsigned int vocabulary_size,
    unsigned int tokens,
    unsigned int channels
) {
    unsigned int channel = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int token = blockIdx.y * blockDim.y + threadIdx.y;
    if (channel >= channels || token >= tokens) return;
    float raw_index = indices[token];
    if (raw_index < 0.0f || raw_index >= (float)vocabulary_size || floorf(raw_index) != raw_index) {
        result[token * channels + channel] = 0.0f;
        return;
    }
    unsigned int index = (unsigned int)raw_index;
    result[token * channels + channel] = table[index * channels + channel];
}

extern "C" __global__ void embedding_gradient(
    const float* indices,
    const float* output_gradient,
    float* result,
    unsigned int vocabulary_size,
    unsigned int tokens,
    unsigned int channels
) {
    unsigned int channel = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int vocabulary_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (channel >= channels || vocabulary_index >= vocabulary_size) return;
    float sum = 0.0f;
    for (unsigned int token = 0; token < tokens; token++) {
        float raw_index = indices[token];
        if (raw_index >= 0.0f && floorf(raw_index) == raw_index && (unsigned int)raw_index == vocabulary_index) {
            sum += output_gradient[token * channels + channel];
        }
    }
    result[vocabulary_index * channels + channel] = sum;
}

extern "C" __global__ void cached_self_attention(
    const float* query,
    const float* key,
    const float* value,
    float* key_cache,
    float* value_cache,
    float* result,
    unsigned int position,
    unsigned int channels,
    unsigned int heads
) {
    unsigned int head = blockIdx.x * blockDim.x + threadIdx.x;
    if (head >= heads) return;
    unsigned int head_width = channels / heads;
    unsigned int offset = head * head_width;
    for (unsigned int channel = 0; channel < head_width; channel++) {
        key_cache[position * channels + offset + channel] = key[offset + channel];
        value_cache[position * channels + offset + channel] = value[offset + channel];
    }
    float scale = rsqrtf((float)head_width);
    float max_score = -ZIG_NN_CUDA_INFINITY;
    for (unsigned int key_position = 0; key_position <= position; key_position++) {
        float score = 0.0f;
        for (unsigned int channel = 0; channel < head_width; channel++) score += query[offset + channel] * key_cache[key_position * channels + offset + channel];
        max_score = fmaxf(max_score, score * scale);
    }
    float denominator = 0.0f;
    for (unsigned int key_position = 0; key_position <= position; key_position++) {
        float score = 0.0f;
        for (unsigned int channel = 0; channel < head_width; channel++) score += query[offset + channel] * key_cache[key_position * channels + offset + channel];
        denominator += __expf(score * scale - max_score);
    }
    for (unsigned int output_channel = 0; output_channel < head_width; output_channel++) {
        float output = 0.0f;
        for (unsigned int key_position = 0; key_position <= position; key_position++) {
            float score = 0.0f;
            for (unsigned int channel = 0; channel < head_width; channel++) score += query[offset + channel] * key_cache[key_position * channels + offset + channel];
            output += __expf(score * scale - max_score) / denominator * value_cache[key_position * channels + offset + output_channel];
        }
        result[offset + output_channel] = output;
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
