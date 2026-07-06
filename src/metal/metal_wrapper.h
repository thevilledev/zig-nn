#ifndef METAL_WRAPPER_H
#define METAL_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

// Opaque types for Metal objects
typedef void* MTLDeviceRef;
typedef void* MTLCommandQueueRef;
typedef void* MTLLibraryRef;
typedef void* MTLFunctionRef;
typedef void* MTLBufferRef;
typedef void* MTLComputePipelineStateRef;
typedef void* MTLCommandBufferRef;
typedef void* MTLComputeCommandEncoderRef;

// Device creation
MTLDeviceRef metal_create_system_default_device(void);

// Command queue
MTLCommandQueueRef metal_device_create_command_queue(MTLDeviceRef device);
MTLCommandBufferRef metal_command_queue_create_command_buffer(MTLCommandQueueRef command_queue);

// Command buffer synchronization
void metal_command_buffer_commit(MTLCommandBufferRef command_buffer);
int metal_command_buffer_wait_until_completed(MTLCommandBufferRef command_buffer, char* error_buffer, unsigned long error_buffer_len);

// Buffer creation
MTLBufferRef metal_device_create_buffer(MTLDeviceRef device, unsigned long length, unsigned int options);
unsigned long metal_buffer_length(MTLBufferRef buffer);
int metal_buffer_upload_f64(MTLBufferRef buffer, const double* source, unsigned long count);
int metal_buffer_download_f64(MTLBufferRef buffer, double* destination, unsigned long count);

// Shader library and pipeline creation
MTLLibraryRef metal_device_create_library_from_source(MTLDeviceRef device, const char* source, char* error_buffer, unsigned long error_buffer_len);
MTLFunctionRef metal_library_create_function(MTLLibraryRef library, const char* function_name);
MTLComputePipelineStateRef metal_device_create_compute_pipeline_state(MTLDeviceRef device, MTLFunctionRef function, char* error_buffer, unsigned long error_buffer_len);

// Compute encoder
MTLComputeCommandEncoderRef metal_command_buffer_create_compute_command_encoder(MTLCommandBufferRef command_buffer);
void metal_compute_encoder_set_pipeline_state(MTLComputeCommandEncoderRef encoder, MTLComputePipelineStateRef pipeline_state);
void metal_compute_encoder_set_buffer(MTLComputeCommandEncoderRef encoder, MTLBufferRef buffer, unsigned long offset, unsigned int index);
void metal_compute_encoder_set_bytes(MTLComputeCommandEncoderRef encoder, const void* bytes, unsigned long length, unsigned int index);
void metal_compute_encoder_dispatch_threads(MTLComputeCommandEncoderRef encoder, MTLComputePipelineStateRef pipeline_state, unsigned long width, unsigned long height, unsigned long depth);
void metal_compute_encoder_end_encoding(MTLComputeCommandEncoderRef encoder);

// Retained Objective-C object release
void metal_release(void* object);

#ifdef __cplusplus
}
#endif

#endif /* METAL_WRAPPER_H */
