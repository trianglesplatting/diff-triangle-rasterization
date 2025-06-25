/*
 * Metal backend for triangle rasterization on Apple Silicon
 * Copyright (C) 2024 - Mac M1 Port
 */

#pragma once

#ifdef __APPLE__
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

#include <glm/glm.hpp>
#include <memory>

namespace MetalRasterizer {

struct RasterizeArgs {
    int P;
    int D; 
    int M;
    int R;
    int H;
    int W;
    float focal_x, focal_y;
    float tan_fovx, tan_fovy;
    float* background;
    float* means3D;
    float* colors;
    float* opacity;
    float* scales;
    float* rotations;
    float* cov3D_precomp;
    float* viewmatrix;
    float* projmatrix;
    float* cam_pos;
    bool prefiltered;
};

class MetalRasterizerImpl {
private:
#ifdef __APPLE__
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> rasterizePipeline;
    id<MTLComputePipelineState> backwardPipeline;
#endif
    
public:
    MetalRasterizerImpl();
    ~MetalRasterizerImpl();
    
    bool initialize();
    
    int rasterize_triangles(
        const RasterizeArgs& args,
        float* out_color,
        float* out_depth,
        int* out_alpha,
        float* out_cum_alpha,
        int* radii,
        float* geomBuffer,
        float* binningBuffer,
        float* imgBuffer
    );
    
    int rasterize_triangles_backward(
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
    );
};

} // namespace MetalRasterizer