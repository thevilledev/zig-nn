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

// Device creation
MTLDeviceRef metal_create_system_default_device(void);

// Command queue
MTLCommandQueueRef metal_device_create_command_queue(MTLDeviceRef device);

// Buffer creation
MTLBufferRef metal_device_create_buffer(MTLDeviceRef device, unsigned long length, unsigned int options);

#ifdef __cplusplus
}
#endif

#endif /* METAL_WRAPPER_H */ 
