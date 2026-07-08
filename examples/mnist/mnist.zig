const std = @import("std");
const nn = @import("nn");
const Network = nn.Network;
const Matrix = nn.Matrix;
const Activation = nn.Activation;

const image_size = 28 * 28;
const digit_count = 10;
const mnist_mean = 0.1307;
const mnist_stddev = 0.3081;

/// Structure to hold MNIST dataset
/// images: Raw pixel data for all images (flattened 28x28 pixels)
/// labels: Ground truth labels (0-9) for each image
/// num_images: Total number of images in the dataset
const MnistData = struct {
    images: []u8,
    labels: []u8,
    num_images: usize,

    pub fn deinit(self: *const MnistData, allocator: std.mem.Allocator) void {
        allocator.free(self.images);
        allocator.free(self.labels);
    }
};

/// Reads MNIST dataset from IDX format files
/// The IDX file format is a simple format for vectors and multidimensional matrices of various numerical types
/// Format reference: http://yann.lecun.com/exdb/mnist/
/// Parameters:
///   - allocator: Memory allocator for dataset
///   - images_path: Path to IDX file containing image data
///   - labels_path: Path to IDX file containing label data
fn readMnistData(allocator: std.mem.Allocator, images_path: []const u8, labels_path: []const u8) !MnistData {
    const io = std.Options.debug_io;

    // Open image file
    const images_file = try std.Io.Dir.cwd().openFile(io, images_path, .{});
    defer images_file.close(io);

    // Open labels file
    const labels_file = try std.Io.Dir.cwd().openFile(io, labels_path, .{});
    defer labels_file.close(io);

    var images_reader_buffer: [4096]u8 = undefined;
    var labels_reader_buffer: [4096]u8 = undefined;
    var images_reader = images_file.readerStreaming(io, &images_reader_buffer);
    var labels_reader = labels_file.readerStreaming(io, &labels_reader_buffer);
    const images_in = &images_reader.interface;
    const labels_in = &labels_reader.interface;

    // Read image file header
    var header: [16]u8 = undefined;
    try images_in.readSliceAll(header[0..]);

    // Read header values in big-endian format
    const magic_number = std.mem.bytesToValue(u32, header[0..4]);
    const num_images = std.mem.bytesToValue(u32, header[4..8]);
    const num_rows = std.mem.bytesToValue(u32, header[8..12]);
    const num_cols = std.mem.bytesToValue(u32, header[12..16]);

    // Convert from little-endian to big-endian
    const magic_number_be = @byteSwap(magic_number);
    const num_images_be = @byteSwap(num_images);
    const num_rows_be = @byteSwap(num_rows);
    const num_cols_be = @byteSwap(num_cols);

    if (magic_number_be != 0x803) {
        return error.InvalidMagicNumber;
    }

    // Read label file header
    var label_header: [8]u8 = undefined;
    try labels_in.readSliceAll(label_header[0..]);

    // Read label header values in big-endian format
    const label_magic = std.mem.bytesToValue(u32, label_header[0..4]);
    const num_labels = std.mem.bytesToValue(u32, label_header[4..8]);

    // Convert from little-endian to big-endian
    const label_magic_be = @byteSwap(label_magic);
    const num_labels_be = @byteSwap(num_labels);

    if (label_magic_be != 0x801 or num_labels_be != num_images_be) {
        return error.InvalidLabelFile;
    }

    // Read image data
    const idx_image_size = num_rows_be * num_cols_be;
    const images = try allocator.alloc(u8, num_images_be * idx_image_size);
    try images_in.readSliceAll(images);

    // Read label data
    const labels = try allocator.alloc(u8, num_images_be);
    try labels_in.readSliceAll(labels);

    return MnistData{
        .images = images,
        .labels = labels,
        .num_images = num_images_be,
    };
}

/// Converts raw MNIST data into network-ready format
/// Parameters:
///   - allocator: Memory allocator for matrices
///   - data: Raw MNIST data structure
///   - start_idx: Starting index in the dataset (for batch processing)
///   - batch_size: Number of images to process in this batch
/// Returns:
///   - Input matrix: Normalized pixel values using MNIST mean/std
///   - Target matrix: One-hot encoded labels (e.g., [0,1,0,0,0,0,0,0,0,0] for digit 1)
fn prepareData(allocator: std.mem.Allocator, data: MnistData, start_idx: usize, batch_size: usize) !struct { Matrix, Matrix } {
    var input = try Matrix.init(allocator, batch_size, image_size);
    var target = try Matrix.init(allocator, batch_size, digit_count);

    var i: usize = 0;
    while (i < batch_size) : (i += 1) {
        const idx = start_idx + i;
        if (idx >= data.num_images) break;

        var j: usize = 0;
        while (j < image_size) : (j += 1) {
            const pixel = @as(f64, @floatFromInt(data.images[idx * image_size + j])) / 255.0;
            try input.set(i, j, (pixel - mnist_mean) / mnist_stddev);
        }

        // Create one-hot encoded target
        const label = data.labels[idx];
        var k: usize = 0;
        while (k < digit_count) : (k += 1) {
            try target.set(i, k, if (k == label) 1.0 else 0.0);
        }
    }

    return .{ input, target };
}

const EvaluationResult = struct {
    loss: f64,
    accuracy: f64,
};

fn scheduledLearningRate(epoch: usize) f64 {
    if (epoch < 3) return 0.001;
    if (epoch < 6) return 0.0001;
    return 0.00001;
}

fn gradientNorm(matrix: Matrix) f64 {
    var squared_sum: f64 = 0.0;
    for (matrix.data) |value| {
        squared_sum += value * value;
    }
    return @sqrt(squared_sum);
}

fn backwardClipped(network: *Network, predicted: Matrix, target: Matrix, max_norm: f64) !void {
    if (network.layers.items.len == 0) {
        return error.EmptyNetwork;
    }

    var gradient = try network.calculateLossGradient(predicted, target);
    errdefer gradient.deinit();

    const norm = gradientNorm(gradient);
    if (max_norm > 0.0 and norm > max_norm) {
        const scale = max_norm / (norm + 1e-12);
        const clipped = try gradient.scale(scale, network.allocator);
        gradient.deinit();
        gradient = clipped;
    }

    var i = network.layers.items.len;
    while (i > 0) {
        i -= 1;
        const new_gradient = try network.layers.items[i].backward(gradient, network.learning_rate);
        gradient.deinit();
        if (i > 0) {
            gradient = new_gradient;
        } else {
            new_gradient.deinit();
        }
    }
}

fn evaluate(
    allocator: std.mem.Allocator,
    network: *Network,
    data: MnistData,
    start_idx: usize,
    sample_size: usize,
    batch_size: usize,
) !EvaluationResult {
    const batches = sample_size / batch_size;
    if (batches == 0) return error.EmptyInput;

    var total_loss: f64 = 0.0;
    var correct: usize = 0;
    var total: usize = 0;

    var batch: usize = 0;
    while (batch < batches) : (batch += 1) {
        const batch_data = try prepareData(allocator, data, start_idx + batch * batch_size, batch_size);
        const input = batch_data[0];
        const target = batch_data[1];
        defer input.deinit();
        defer target.deinit();

        const output = try network.forward(input);
        defer output.deinit();

        total_loss += try network.calculateLoss(output, target);

        var row: usize = 0;
        while (row < output.rows) : (row += 1) {
            var max_idx: usize = 0;
            var max_val: f64 = try output.get(row, 0);
            var col: usize = 1;
            while (col < digit_count) : (col += 1) {
                const value = try output.get(row, col);
                if (value > max_val) {
                    max_val = value;
                    max_idx = col;
                }
            }

            if (data.labels[start_idx + batch * batch_size + row] == max_idx) {
                correct += 1;
            }
            total += 1;
        }
    }

    return .{
        .loss = total_loss / @as(f64, @floatFromInt(batches)),
        .accuracy = @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(total)),
    };
}

pub fn main() !void {
    // Initialize memory allocator
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create neural network with CrossEntropy loss. The epoch loop updates the
    // learning rate with a simple schedule.
    var network = Network.init(allocator, scheduledLearningRate(0), .CrossEntropy);
    defer network.deinit();

    // Network architecture:
    // 1. Input layer: 784 neurons (28x28 pixels flattened)
    // 2. Hidden layer 1: 256 neurons with ReLU activation
    // 3. Hidden layer 2: 128 neurons with ReLU activation
    // 4. Output layer: 10 neurons (one per digit) with softmax activation
    try network.addLayer(image_size, 256, Activation.relu, Activation.relu_derivative);
    try network.addLayer(256, 128, Activation.relu, Activation.relu_derivative);
    try network.addLayer(128, digit_count, Activation.softmax, Activation.softmax_derivative);

    // Training configuration
    // - Using subset of data for faster training/testing
    // - Batch size 32 provides good balance between:
    //   * Training stability (larger batches = better gradient estimates)
    //   * Training speed (smaller batches = more weight updates)
    //   * Memory usage
    const validation_sample_size: usize = 5000;
    const training_sample_size: usize = 50000; // Out of 60000 total training images
    const test_sample_size: usize = 10000; // Full test set
    const batch_size = 32;
    const num_epochs = 10;
    const gradient_clip_norm = 5.0;

    // Load MNIST training and test datasets
    var train_data = try readMnistData(
        allocator,
        "data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte",
    );
    defer train_data.deinit(allocator);

    var test_data = try readMnistData(
        allocator,
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte",
    );
    defer test_data.deinit(allocator);

    // Calculate effective dataset sizes and batches per epoch
    const effective_validation_size = @min(validation_sample_size, train_data.num_images);
    const train_start_idx = effective_validation_size;
    const available_train_size = train_data.num_images - effective_validation_size;
    const effective_train_size = @min(training_sample_size, available_train_size);
    const effective_test_size = @min(test_sample_size, test_data.num_images);
    const batches_per_epoch = effective_train_size / batch_size;

    std.debug.print("Training on {d} samples, validating on {d} samples, testing on {d} samples\n", .{
        effective_train_size,
        effective_validation_size,
        effective_test_size,
    });
    std.debug.print("Will run {d} batches per epoch\n\n", .{batches_per_epoch});

    // Training loop
    // For each epoch:
    // 1. Train on batches of data
    // 2. Track and report loss
    // 3. Evaluate on test set
    // 4. Report accuracy
    var epoch: usize = 0;
    while (epoch < num_epochs) : (epoch += 1) {
        var total_loss: f64 = 0.0;
        var batch: usize = 0;
        network.learning_rate = scheduledLearningRate(epoch);

        std.debug.print("Epoch {d}/{d} - learning rate {d:.6}:\n", .{ epoch + 1, num_epochs, network.learning_rate });

        // Progress tracking - show updates every 10% of epoch
        const progress_interval = @max(batches_per_epoch / 10, 1);

        // Training loop for single epoch
        while (batch < batches_per_epoch) : (batch += 1) {
            const start_idx = train_start_idx + batch * batch_size;
            const batch_data = try prepareData(allocator, train_data, start_idx, batch_size);
            const input = batch_data[0];
            const target = batch_data[1];
            defer input.deinit();
            defer target.deinit();

            // Forward pass through network
            const output = try network.forward(input);
            defer output.deinit();

            // Numerical stability checks
            // Detect NaN or Inf values which indicate training problems
            var has_invalid = false;
            for (0..output.rows) |i| {
                for (0..output.cols) |j| {
                    const val = try output.get(i, j);
                    if (std.math.isNan(val) or std.math.isInf(val)) {
                        has_invalid = true;
                        std.debug.print("Invalid output at batch {d}, row {d}, col {d}: {d}\n", .{ batch, i, j, val });
                        break;
                    }
                }
                if (has_invalid) break;
            }

            // Calculate loss for this batch
            const loss = try network.calculateLoss(output, target);

            // Additional numerical stability checks for loss
            if (std.math.isNan(loss) or std.math.isInf(loss)) {
                std.debug.print("Invalid loss at batch {d}: {d}\n", .{ batch, loss });
                std.debug.print("Output values:\n", .{});
                for (0..@min(output.rows, 3)) |i| {
                    for (0..output.cols) |j| {
                        std.debug.print("{d:.4} ", .{try output.get(i, j)});
                    }
                    std.debug.print("\n", .{});
                }
                std.debug.print("Target values:\n", .{});
                for (0..@min(target.rows, 3)) |i| {
                    for (0..target.cols) |j| {
                        std.debug.print("{d:.4} ", .{try target.get(i, j)});
                    }
                    std.debug.print("\n", .{});
                }
                return error.NumericalInstability;
            }

            total_loss += loss;

            // Progress reporting
            if ((batch + 1) % progress_interval == 0) {
                const progress = @as(f64, @floatFromInt(batch + 1)) / @as(f64, @floatFromInt(batches_per_epoch)) * 100.0;
                const running_avg_loss = total_loss / @as(f64, @floatFromInt(batch + 1));
                std.debug.print("  Progress: {d:>3.0}% - Batch {d}/{d} - Running avg loss: {d:.4}\n", .{ progress, batch + 1, batches_per_epoch, running_avg_loss });
            }

            try backwardClipped(&network, output, target, gradient_clip_norm);
        }

        // Report epoch results
        const avg_loss = total_loss / @as(f64, @floatFromInt(batches_per_epoch));
        std.debug.print("\nEpoch {d}/{d} completed - Final avg loss: {d:.4}\n", .{ epoch + 1, num_epochs, avg_loss });

        const validation = try evaluate(allocator, &network, train_data, 0, effective_validation_size, batch_size);
        const test_result = try evaluate(allocator, &network, test_data, 0, effective_test_size, batch_size);

        std.debug.print("Validation loss: {d:.4}, accuracy: {d:.2}%\n", .{ validation.loss, validation.accuracy * 100.0 });
        std.debug.print("Test loss: {d:.4}, accuracy: {d:.2}%\n\n", .{ test_result.loss, test_result.accuracy * 100.0 });
    }
}

test "scheduled learning rate decays every three epochs" {
    try std.testing.expectApproxEqAbs(@as(f64, 0.001), scheduledLearningRate(0), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.001), scheduledLearningRate(2), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0001), scheduledLearningRate(3), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.00001), scheduledLearningRate(6), 1e-12);
}

test "prepareData applies MNIST mean and stddev normalization" {
    const allocator = std.testing.allocator;

    var images: [image_size]u8 = undefined;
    @memset(images[0..], 0);
    images[0] = 255;
    var labels = [_]u8{3};
    const data = MnistData{
        .images = images[0..],
        .labels = labels[0..],
        .num_images = 1,
    };

    const batch_data = try prepareData(allocator, data, 0, 1);
    const input = batch_data[0];
    const target = batch_data[1];
    defer input.deinit();
    defer target.deinit();

    try std.testing.expectApproxEqAbs((1.0 - mnist_mean) / mnist_stddev, try input.get(0, 0), 1e-12);
    try std.testing.expectApproxEqAbs((0.0 - mnist_mean) / mnist_stddev, try input.get(0, 1), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try target.get(0, 3), 1e-12);
}
