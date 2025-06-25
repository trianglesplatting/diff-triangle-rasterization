/*
 * Metal backend implementation for triangle rasterization on Apple Silicon
 * Copyright (C) 2024 - Mac M1 Port
 */

#include "rasterizer_metal.h"
#include <iostream>

#ifdef __APPLE__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

namespace MetalRasterizer {

MetalRasterizerImpl::MetalRasterizerImpl() {
#ifdef __APPLE__
    device = nil;
    commandQueue = nil;
    library = nil;
    rasterizePipeline = nil;
    backwardPipeline = nil;
#endif
}

MetalRasterizerImpl::~MetalRasterizerImpl() {
#ifdef __APPLE__
    if (rasterizePipeline) {
        [rasterizePipeline release];
    }
    if (backwardPipeline) {
        [backwardPipeline release];
    }
    if (library) {
        [library release];
    }
    if (commandQueue) {
        [commandQueue release];
    }
    if (device) {
        [device release];
    }
#endif
}

bool MetalRasterizerImpl::initialize() {
#ifdef __APPLE__
    // Get default Metal device
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Metal is not supported on this device" << std::endl;
        return false;
    }
    
    // Create command queue
    commandQueue = [device newCommandQueue];
    if (!commandQueue) {
        std::cerr << "Failed to create Metal command queue" << std::endl;
        return false;
    }
    
    // Load Metal shaders (would need .metal files)
    NSError* error = nil;
    NSString* shaderSource = @R"(
        #include <metal_stdlib>
        using namespace metal;
        
        struct RasterizeParams {
            int P, D, M, R, H, W;
            float focal_x, focal_y;
            float tan_fovx, tan_fovy;
        };
        
        kernel void rasterize_triangles_kernel(
            device float* means3D [[buffer(0)]],
            device float* colors [[buffer(1)]],
            device float* opacity [[buffer(2)]],
            device float* out_color [[buffer(3)]],
            device float* out_depth [[buffer(4)]],
            constant RasterizeParams& params [[buffer(5)]],
            uint3 gid [[thread_position_in_grid]]
        ) {
            // Basic rasterization kernel - simplified version
            uint idx = gid.x;
            if (idx >= params.P) return;
            
            // TODO: Implement full rasterization logic
            // This is a placeholder that copies input to output
            if (idx < params.P * 3) {
                out_color[idx] = colors[idx];
            }
        }
        
        kernel void rasterize_backward_kernel(
            device float* grad_out_color [[buffer(0)]],
            device float* grad_means3D [[buffer(1)]],
            device float* grad_colors [[buffer(2)]],
            constant RasterizeParams& params [[buffer(3)]],
            uint3 gid [[thread_position_in_grid]]
        ) {
            // Basic backward pass - simplified version
            uint idx = gid.x;
            if (idx >= params.P) return;
            
            // TODO: Implement full backward logic
            if (idx < params.P * 3) {
                grad_colors[idx] = grad_out_color[idx];
            }
        }
    )";
    
    library = [device newLibraryWithSource:shaderSource 
                                   options:nil 
                                     error:&error];
    if (!library) {
        std::cerr << "Failed to create Metal library: " 
                  << [[error localizedDescription] UTF8String] << std::endl;
        return false;
    }
    
    // Create compute pipeline states
    id<MTLFunction> rasterizeFunction = [library newFunctionWithName:@"rasterize_triangles_kernel"];
    id<MTLFunction> backwardFunction = [library newFunctionWithName:@"rasterize_backward_kernel"];
    
    rasterizePipeline = [device newComputePipelineStateWithFunction:rasterizeFunction error:&error];
    if (!rasterizePipeline) {
        std::cerr << "Failed to create rasterize pipeline: " 
                  << [[error localizedDescription] UTF8String] << std::endl;
        return false;
    }
    
    backwardPipeline = [device newComputePipelineStateWithFunction:backwardFunction error:&error];
    if (!backwardPipeline) {
        std::cerr << "Failed to create backward pipeline: " 
                  << [[error localizedDescription] UTF8String] << std::endl;
        return false;
    }
    
    return true;
#else
    std::cerr << "Metal backend not available on non-Apple platforms" << std::endl;
    return false;
#endif
}

int MetalRasterizerImpl::rasterize_triangles(
    const RasterizeArgs& args,
    float* out_color,
    float* out_depth,
    int* out_alpha,
    float* out_cum_alpha,
    int* radii,
    float* geomBuffer,
    float* binningBuffer,
    float* imgBuffer
) {
#ifdef __APPLE__
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:rasterizePipeline];
        
        // Create Metal buffers
        size_t means3D_size = args.P * 3 * sizeof(float);
        size_t colors_size = args.P * args.D * sizeof(float);
        size_t output_size = args.H * args.W * 3 * sizeof(float);
        
        id<MTLBuffer> means3DBuffer = [device newBufferWithBytes:args.means3D 
                                                          length:means3D_size 
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> colorsBuffer = [device newBufferWithBytes:args.colors 
                                                         length:colors_size 
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:output_size 
                                                         options:MTLResourceStorageModeShared];
        
        [encoder setBuffer:means3DBuffer offset:0 atIndex:0];
        [encoder setBuffer:colorsBuffer offset:0 atIndex:1];
        [encoder setBuffer:outputBuffer offset:0 atIndex:3];
        
        // Set compute parameters
        struct {
            int P, D, M, R, H, W;
            float focal_x, focal_y;
            float tan_fovx, tan_fovy;
        } params = {
            args.P, args.D, args.M, args.R, args.H, args.W,
            args.focal_x, args.focal_y, args.tan_fovx, args.tan_fovy
        };
        
        [encoder setBytes:&params length:sizeof(params) atIndex:5];
        
        // Dispatch threads
        MTLSize gridSize = MTLSizeMake(args.P, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(std::min(args.P, 256), 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy results back
        memcpy(out_color, [outputBuffer contents], output_size);
        
        return 0;
    }
#else
    std::cerr << "Metal backend not available" << std::endl;
    return -1;
#endif
}

int MetalRasterizerImpl::rasterize_triangles_backward(
    const RasterizeArgs& args,
    const float* grad_out_color,
    const float* grad_out_depth,
    const int* radii,
    const float* geomBuffer,
    const float* binningBuffer,
    const float* imgBuffer,
    float* grad_means3D,
    float* grad_colors,
    float* grad_opacity,
    float* grad_scales,
    float* grad_rotations,
    float* grad_cov3D_precomp
) {
#ifdef __APPLE__
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:backwardPipeline];
        
        // Create Metal buffers for backward pass
        size_t grad_size = args.H * args.W * 3 * sizeof(float);
        size_t output_grad_size = args.P * 3 * sizeof(float);
        
        id<MTLBuffer> gradInputBuffer = [device newBufferWithBytes:grad_out_color 
                                                            length:grad_size 
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> gradOutputBuffer = [device newBufferWithLength:output_grad_size 
                                                             options:MTLResourceStorageModeShared];
        
        [encoder setBuffer:gradInputBuffer offset:0 atIndex:0];
        [encoder setBuffer:gradOutputBuffer offset:0 atIndex:1];
        
        // Set compute parameters
        struct {
            int P, D, M, R, H, W;
            float focal_x, focal_y;
            float tan_fovx, tan_fovy;
        } params = {
            args.P, args.D, args.M, args.R, args.H, args.W,
            args.focal_x, args.focal_y, args.tan_fovx, args.tan_fovy
        };
        
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        
        // Dispatch threads
        MTLSize gridSize = MTLSizeMake(args.P, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(std::min(args.P, 256), 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy results back
        memcpy(grad_means3D, [gradOutputBuffer contents], output_grad_size);
        
        return 0;
    }
#else
    std::cerr << "Metal backend not available" << std::endl;
    return -1;
#endif
}

} // namespace MetalRasterizer