const std = @import("std");
const nn = @import("nn");
const Network = nn.Network;
const Matrix = nn.Matrix;
const Activation = nn.Activation;

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
    // Open image file
    const images_file = try std.fs.cwd().openFile(images_path, .{});
    defer images_file.close();

    // Open labels file
    const labels_file = try std.fs.cwd().openFile(labels_path, .{});
    defer labels_file.close();

    // Read image file header
    var header: [16]u8 = undefined;
    _ = try images_file.read(header[0..]);
    
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
    _ = try labels_file.read(label_header[0..]);
    
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
    const image_size = num_rows_be * num_cols_be;
    const images = try allocator.alloc(u8, num_images_be * image_size);
    _ = try images_file.read(images);

    // Read label data
    const labels = try allocator.alloc(u8, num_images_be);
    _ = try labels_file.read(labels);

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
///   - Input matrix: Normalized pixel values (-0.15 to 0.15)
///   - Target matrix: One-hot encoded labels (e.g., [0,1,0,0,0,0,0,0,0,0] for digit 1)
fn prepareData(allocator: std.mem.Allocator, data: MnistData, start_idx: usize, batch_size: usize) !struct { Matrix, Matrix } {
    const image_size = 28 * 28;
    var input = try Matrix.init(allocator, batch_size, image_size);
    var target = try Matrix.init(allocator, batch_size, 10);

    var i: usize = 0;
    while (i < batch_size) : (i += 1) {
        const idx = start_idx + i;
        if (idx >= data.num_images) break;

        // Normalize inputs considering weight initialization scale
        var j: usize = 0;
        while (j < image_size) : (j += 1) {
            const pixel = @as(f64, @floatFromInt(data.images[idx * image_size + j])) / 255.0;
            // Keep inputs small but not too small, matching initialization scale
            input.set(i, j, (pixel - 0.5) * 0.3);
        }

        // Create one-hot encoded target
        const label = data.labels[idx];
        var k: usize = 0;
        while (k < 10) : (k += 1) {
            target.set(i, k, if (k == label) 1.0 else 0.0);
        }
    }

    return .{ input, target };
}

pub fn main() !void {
    // Initialize memory allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create neural network with CrossEntropy loss
    // Learning rate 0.0001 chosen empirically for stable training
    var network = Network.init(allocator, 0.0001, .CrossEntropy);
    defer network.deinit();

    // Network architecture:
    // 1. Input layer: 784 neurons (28x28 pixels flattened)
    // 2. Hidden layer 1: 256 neurons with tanh activation
    // 3. Hidden layer 2: 128 neurons with tanh activation
    // 4. Hidden layer 3: 64 neurons with tanh activation
    // 5. Output layer: 10 neurons (one per digit) with softmax activation
    // 
    // Architecture design choices:
    // - Gradually decreasing layer sizes help manage gradient flow
    // - tanh activation provides good gradient properties
    // - Final softmax layer outputs probability distribution over digits
    try network.addLayer(784, 256, Activation.tanh, Activation.tanh_derivative);
    try network.addLayer(256, 128, Activation.tanh, Activation.tanh_derivative);
    try network.addLayer(128, 64, Activation.tanh, Activation.tanh_derivative);
    try network.addLayer(64, 10, Activation.softmax, Activation.softmax_derivative);

    // Training configuration
    // - Using subset of data for faster training/testing
    // - Batch size 32 provides good balance between:
    //   * Training stability (larger batches = better gradient estimates)
    //   * Training speed (smaller batches = more weight updates)
    //   * Memory usage
    const training_sample_size: usize = 50000;  // Out of 60000 total training images
    const test_sample_size: usize = 10000;      // Full test set
    const batch_size = 32;
    const num_epochs = 10;

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
    const effective_train_size = @min(training_sample_size, train_data.num_images);
    const effective_test_size = @min(test_sample_size, test_data.num_images);
    const batches_per_epoch = effective_train_size / batch_size;

    std.debug.print("Training on {d} samples, testing on {d} samples\n", 
        .{ effective_train_size, effective_test_size });
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

        std.debug.print("Epoch {d}/{d}:\n", .{ epoch + 1, num_epochs });
        
        // Progress tracking - show updates every 10% of epoch
        const progress_interval = @max(batches_per_epoch / 10, 1);
        
        // Training loop for single epoch
        while (batch < batches_per_epoch) : (batch += 1) {
            const start_idx = batch * batch_size;
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
                    const val = output.get(i, j);
                    if (std.math.isNan(val) or std.math.isInf(val)) {
                        has_invalid = true;
                        std.debug.print("Invalid output at batch {d}, row {d}, col {d}: {d}\n", 
                            .{ batch, i, j, val });
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
                        std.debug.print("{d:.4} ", .{output.get(i, j)});
                    }
                    std.debug.print("\n", .{});
                }
                std.debug.print("Target values:\n", .{});
                for (0..@min(target.rows, 3)) |i| {
                    for (0..target.cols) |j| {
                        std.debug.print("{d:.4} ", .{target.get(i, j)});
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
                std.debug.print("  Progress: {d:>3.0}% - Batch {d}/{d} - Running avg loss: {d:.4}\n", 
                    .{ progress, batch + 1, batches_per_epoch, running_avg_loss });
            }

            // Backward pass - update network weights
            _ = try network.backward(output, target);
        }

        // Report epoch results
        const avg_loss = total_loss / @as(f64, @floatFromInt(batches_per_epoch));
        std.debug.print("\nEpoch {d}/{d} completed - Final avg loss: {d:.4}\n", 
            .{ epoch + 1, num_epochs, avg_loss });

        // Evaluation phase
        // Test network performance on held-out test set
        std.debug.print("Evaluating on test set...\n", .{});
        var correct: usize = 0;
        var total: usize = 0;

        // Process test set in batches
        const test_batches = effective_test_size / batch_size;
        var test_batch: usize = 0;
        while (test_batch < test_batches) : (test_batch += 1) {
            const test_batch_data = try prepareData(allocator, test_data, test_batch * batch_size, batch_size);
            const test_input = test_batch_data[0];
            const test_target = test_batch_data[1];
            defer test_input.deinit();
            defer test_target.deinit();

            // Forward pass only (no training on test set)
            const test_output = try network.forward(test_input);
            defer test_output.deinit();

            // Calculate accuracy
            // For each image, find the digit with highest probability
            // Compare with true label
            var i: usize = 0;
            while (i < test_output.rows) : (i += 1) {
                var max_idx: usize = 0;
                var max_val: f64 = test_output.get(i, 0);
                var j: usize = 1;
                while (j < 10) : (j += 1) {
                    const val = test_output.get(i, j);
                    if (val > max_val) {
                        max_val = val;
                        max_idx = j;
                    }
                }

                if (test_data.labels[test_batch * batch_size + i] == max_idx) {
                    correct += 1;
                }
                total += 1;
            }
        }

        // Report test set accuracy
        const accuracy = @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(total));
        std.debug.print("Test accuracy: {d:.2}%\n\n", .{accuracy * 100.0});
    }
} 
