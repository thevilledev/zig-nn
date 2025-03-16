/// Regression Example using Neural Networks
/// This example demonstrates how to use the neural network to approximate
/// a nonlinear continuous function: f(x) = x² * sin(x)
///
/// The network architecture:
/// - Input layer: 1 neuron (x value)
/// - Hidden layer 1: 32 neurons with ReLU activation
/// - Hidden layer 2: 16 neurons with ReLU activation
/// - Output layer: 1 neuron with linear activation (for regression)
///
/// Training details:
/// - 1000 random samples in range [-4, 4]
/// - Mini-batch gradient descent with batch size 100
/// - Mean Squared Error loss function
/// - Learning rate 0.001
/// - 500 epochs of training
const std = @import("std");
const nn = @import("nn");
const Network = nn.Network;
const Layer = nn.Layer;
const Matrix = nn.Matrix;
const Activation = nn.Activation;
const LossFunction = nn.LossFunction;
const math = std.math;

pub fn main() !void {
    // Initialize memory allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    // Create matrices for training data
    // x_data: Input values
    // y_data: Target values (x² * sin(x))
    const num_samples = 1000;
    var x_data = try Matrix.init(allocator, num_samples, 1);
    var y_data = try Matrix.init(allocator, num_samples, 1);
    defer x_data.deinit();
    defer y_data.deinit();

    // Generate training data points
    // For each point:
    // 1. Generate random x in [-4, 4]
    // 2. Calculate y = x² * sin(x)
    // Use deterministic seed for reproducible results
    var rng = std.Random.DefaultPrng.init(42);
    var i: usize = 0;
    while (i < num_samples) : (i += 1) {
        // Generate more samples at the edges of the range
        const u = rng.random().float(f64); // Uniform in [0, 1]
        // Apply inverse transform sampling to get more samples at edges
        const x = if (u < 0.4) // Increased edge sampling
            if (rng.random().boolean())
                -4.0 + rng.random().float(f64) * 1.5 // [-4, -2.5]
            else
                2.5 + rng.random().float(f64) * 1.5 // [2.5, 4]
        else
            -2.5 + rng.random().float(f64) * 5.0; // [-2.5, 2.5]
        x_data.set(i, 0, x);
        y_data.set(i, 0, x * x * @sin(x));
    }

    // Initialize neural network
    // - learning_rate: Increased for faster convergence
    // - MeanSquaredError: Standard loss function for regression tasks
    const learning_rate = 0.001;
    var network = Network.init(allocator, learning_rate, .MeanSquaredError);
    defer network.deinit();

    // Build network architecture
    // Layer 1: 1 → 32 (Input → Hidden)
    // Layer 2: 32 → 16 (Hidden → Hidden)
    // Layer 3: 16 → 1 (Hidden → Output)
    // ReLU activation for hidden layers helps learn nonlinear patterns
    // Linear activation for output layer allows any real number output
    try network.addLayer(1, 32, Activation.relu, Activation.relu_derivative);
    try network.addLayer(32, 16, Activation.relu, Activation.relu_derivative);
    try network.addLayer(16, 1, Activation.linear, Activation.linear_derivative);

    // Training parameters
    // - num_epochs: Reduced for faster training
    // - batch_size: Increased for better hardware utilization
    const num_epochs = 500;
    const batch_size = 100;

    // Training loop
    // For each epoch:
    // 1. Split data into mini-batches
    // 2. Train network on each batch
    // 3. Calculate and display average loss every 50 epochs
    const stdout = std.io.getStdOut().writer();
    var epoch: usize = 0;
    while (epoch < num_epochs) : (epoch += 1) {
        var total_loss: f64 = 0.0;
        var batch: usize = 0;
        while (batch < num_samples / batch_size) : (batch += 1) {
            // Extract current batch from training data
            const start_idx = batch * batch_size;
            const end_idx = @min(start_idx + batch_size, num_samples);

            var batch_x = try x_data.extractBatch(start_idx, end_idx, allocator);
            var batch_y = try y_data.extractBatch(start_idx, end_idx, allocator);
            defer batch_x.deinit();
            defer batch_y.deinit();

            // Train on current batch and accumulate loss
            const loss = try network.trainBatch(batch_x, batch_y);
            total_loss += loss;
        }

        // Display training progress every 50 epochs
        if (epoch % 50 == 0) {
            try stdout.print("Epoch {d}: Average Loss = {d:.6}\n", .{ epoch, total_loss / @as(f64, @floatFromInt(num_samples / batch_size)) });
        }
    }

    // Test the trained network
    // Evaluate network on a set of test points to visualize the learned function
    try stdout.print("\nTesting the network:\n", .{});
    // Test more points for better visualization
    const test_points = [_]f64{ -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0 };
    for (test_points) |x| {
        // Create input matrix with single test point
        var input = try Matrix.init(allocator, 1, 1);
        defer input.deinit();
        input.set(0, 0, x);

        // Get network's prediction
        var prediction = try network.forward(input);
        defer prediction.deinit();

        // Calculate actual value and display comparison
        const true_value = x * x * @sin(x);
        const err = @abs(prediction.get(0, 0) - true_value);
        try stdout.print("x = {d:6.3}: Predicted = {d:8.3}, Actual = {d:8.3}, Error = {d:8.3}\n", .{ x, prediction.get(0, 0), true_value, err });
    }
}
