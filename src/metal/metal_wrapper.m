#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "metal_wrapper.h"
#include <string.h>

static void metal_copy_message(char* buffer, unsigned long buffer_len, NSString* message) {
    if (buffer == NULL || buffer_len == 0) {
        return;
    }

    buffer[0] = '\0';
    if (message == nil) {
        return;
    }

    const char* utf8 = [message UTF8String];
    if (utf8 == NULL) {
        return;
    }

    strncpy(buffer, utf8, buffer_len - 1);
    buffer[buffer_len - 1] = '\0';
}

static unsigned long metal_min_ulong(unsigned long a, unsigned long b) {
    return a < b ? a : b;
}

MTLDeviceRef metal_create_system_default_device(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return (void*)device;
}

MTLCommandQueueRef metal_device_create_command_queue(MTLDeviceRef device_ref) {
    id<MTLDevice> device = (id<MTLDevice>)device_ref;
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    return (void*)commandQueue;
}

MTLCommandBufferRef metal_command_queue_create_command_buffer(MTLCommandQueueRef command_queue_ref) {
    id<MTLCommandQueue> commandQueue = (id<MTLCommandQueue>)command_queue_ref;
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    if (commandBuffer == nil) {
        return NULL;
    }

    [commandBuffer retain];
    return (void*)commandBuffer;
}

void metal_command_buffer_commit(MTLCommandBufferRef command_buffer_ref) {
    id<MTLCommandBuffer> commandBuffer = (id<MTLCommandBuffer>)command_buffer_ref;
    [commandBuffer commit];
}

int metal_command_buffer_wait_until_completed(MTLCommandBufferRef command_buffer_ref, char* error_buffer, unsigned long error_buffer_len) {
    id<MTLCommandBuffer> commandBuffer = (id<MTLCommandBuffer>)command_buffer_ref;
    [commandBuffer waitUntilCompleted];

    if ([commandBuffer status] == MTLCommandBufferStatusCompleted) {
        metal_copy_message(error_buffer, error_buffer_len, nil);
        return 1;
    }

    NSError* error = [commandBuffer error];
    metal_copy_message(error_buffer, error_buffer_len, error != nil ? [error localizedDescription] : @"Metal command buffer failed");
    return 0;
}

MTLBufferRef metal_device_create_buffer(MTLDeviceRef device_ref, unsigned long length, unsigned int options) {
    id<MTLDevice> device = (id<MTLDevice>)device_ref;
    id<MTLBuffer> buffer = [device newBufferWithLength:length options:options];
    return (void*)buffer;
}

unsigned long metal_buffer_length(MTLBufferRef buffer_ref) {
    id<MTLBuffer> buffer = (id<MTLBuffer>)buffer_ref;
    return [buffer length];
}

int metal_buffer_upload_f32(MTLBufferRef buffer_ref, const float* source, unsigned long count) {
    id<MTLBuffer> buffer = (id<MTLBuffer>)buffer_ref;
    if (buffer == nil || source == NULL) {
        return 0;
    }

    unsigned long required_len = count * sizeof(float);
    if ([buffer length] < required_len) {
        return 0;
    }

    float* destination = (float*)[buffer contents];
    if (destination == NULL) {
        return 0;
    }

    memcpy(destination, source, required_len);

    return 1;
}

int metal_buffer_download_f32(MTLBufferRef buffer_ref, float* destination, unsigned long count) {
    id<MTLBuffer> buffer = (id<MTLBuffer>)buffer_ref;
    if (buffer == nil || destination == NULL) {
        return 0;
    }

    unsigned long required_len = count * sizeof(float);
    if ([buffer length] < required_len) {
        return 0;
    }

    const float* source = (const float*)[buffer contents];
    if (source == NULL) {
        return 0;
    }

    memcpy(destination, source, required_len);

    return 1;
}

MTLLibraryRef metal_device_create_library_from_source(MTLDeviceRef device_ref, const char* source, char* error_buffer, unsigned long error_buffer_len) {
    id<MTLDevice> device = (id<MTLDevice>)device_ref;
    if (device == nil || source == NULL) {
        metal_copy_message(error_buffer, error_buffer_len, @"Missing Metal device or shader source");
        return NULL;
    }

    NSString* sourceString = [[NSString alloc] initWithUTF8String:source];
    if (sourceString == nil) {
        metal_copy_message(error_buffer, error_buffer_len, @"Metal shader source is not valid UTF-8");
        return NULL;
    }

    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:sourceString options:nil error:&error];
    [sourceString release];

    if (library == nil) {
        metal_copy_message(error_buffer, error_buffer_len, error != nil ? [error localizedDescription] : @"Metal shader compilation failed");
        return NULL;
    }

    metal_copy_message(error_buffer, error_buffer_len, nil);
    return (void*)library;
}

MTLFunctionRef metal_library_create_function(MTLLibraryRef library_ref, const char* function_name) {
    id<MTLLibrary> library = (id<MTLLibrary>)library_ref;
    if (library == nil || function_name == NULL) {
        return NULL;
    }

    NSString* functionName = [[NSString alloc] initWithUTF8String:function_name];
    if (functionName == nil) {
        return NULL;
    }

    id<MTLFunction> function = [library newFunctionWithName:functionName];
    [functionName release];
    return (void*)function;
}

MTLComputePipelineStateRef metal_device_create_compute_pipeline_state(MTLDeviceRef device_ref, MTLFunctionRef function_ref, char* error_buffer, unsigned long error_buffer_len) {
    id<MTLDevice> device = (id<MTLDevice>)device_ref;
    id<MTLFunction> function = (id<MTLFunction>)function_ref;
    if (device == nil || function == nil) {
        metal_copy_message(error_buffer, error_buffer_len, @"Missing Metal device or shader function");
        return NULL;
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
    if (pipelineState == nil) {
        metal_copy_message(error_buffer, error_buffer_len, error != nil ? [error localizedDescription] : @"Metal pipeline creation failed");
        return NULL;
    }

    metal_copy_message(error_buffer, error_buffer_len, nil);
    return (void*)pipelineState;
}

MTLComputeCommandEncoderRef metal_command_buffer_create_compute_command_encoder(MTLCommandBufferRef command_buffer_ref) {
    id<MTLCommandBuffer> commandBuffer = (id<MTLCommandBuffer>)command_buffer_ref;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        return NULL;
    }

    [encoder retain];
    return (void*)encoder;
}

void metal_compute_encoder_set_pipeline_state(MTLComputeCommandEncoderRef encoder_ref, MTLComputePipelineStateRef pipeline_state_ref) {
    id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)encoder_ref;
    id<MTLComputePipelineState> pipelineState = (id<MTLComputePipelineState>)pipeline_state_ref;
    [encoder setComputePipelineState:pipelineState];
}

void metal_compute_encoder_set_buffer(MTLComputeCommandEncoderRef encoder_ref, MTLBufferRef buffer_ref, unsigned long offset, unsigned int index) {
    id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)encoder_ref;
    id<MTLBuffer> buffer = (id<MTLBuffer>)buffer_ref;
    [encoder setBuffer:buffer offset:offset atIndex:index];
}

void metal_compute_encoder_set_bytes(MTLComputeCommandEncoderRef encoder_ref, const void* bytes, unsigned long length, unsigned int index) {
    id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)encoder_ref;
    [encoder setBytes:bytes length:length atIndex:index];
}

void metal_compute_encoder_dispatch_threads(MTLComputeCommandEncoderRef encoder_ref, MTLComputePipelineStateRef pipeline_state_ref, unsigned long width, unsigned long height, unsigned long depth) {
    id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)encoder_ref;
    id<MTLComputePipelineState> pipelineState = (id<MTLComputePipelineState>)pipeline_state_ref;

    unsigned long grid_width = width == 0 ? 1 : width;
    unsigned long grid_height = height == 0 ? 1 : height;
    unsigned long grid_depth = depth == 0 ? 1 : depth;

    unsigned long execution_width = [pipelineState threadExecutionWidth];
    if (execution_width == 0) {
        execution_width = 1;
    }

    unsigned long max_threads = [pipelineState maxTotalThreadsPerThreadgroup];
    if (max_threads == 0) {
        max_threads = execution_width;
    }

    unsigned long tg_width = metal_min_ulong(execution_width, grid_width);
    if (tg_width == 0) {
        tg_width = 1;
    }

    unsigned long tg_height_limit = max_threads / tg_width;
    if (tg_height_limit == 0) {
        tg_height_limit = 1;
    }
    unsigned long tg_height = metal_min_ulong(tg_height_limit, grid_height);
    if (tg_height == 0) {
        tg_height = 1;
    }

    MTLSize gridSize = MTLSizeMake(grid_width, grid_height, grid_depth);
    MTLSize threadgroupSize = MTLSizeMake(tg_width, tg_height, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
}

void metal_compute_encoder_end_encoding(MTLComputeCommandEncoderRef encoder_ref) {
    id<MTLComputeCommandEncoder> encoder = (id<MTLComputeCommandEncoder>)encoder_ref;
    [encoder endEncoding];
}

void metal_release(void* object) {
    if (object != NULL) {
        [(id)object release];
    }
}
