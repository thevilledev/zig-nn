#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "metal_wrapper.h"

MTLDeviceRef metal_create_system_default_device(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return (__bridge_retained void*)device;
}

MTLCommandQueueRef metal_device_create_command_queue(MTLDeviceRef device_ref) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_ref;
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    return (__bridge_retained void*)commandQueue;
}

MTLBufferRef metal_device_create_buffer(MTLDeviceRef device_ref, unsigned long length, unsigned int options) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_ref;
    id<MTLBuffer> buffer = [device newBufferWithLength:length options:options];
    return (__bridge_retained void*)buffer;
} 
