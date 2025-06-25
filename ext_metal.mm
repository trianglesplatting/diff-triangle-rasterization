/*
 * PyTorch extension for Metal backend
 * Copyright (C) 2024 - Mac M1 Port
 */

#include <torch/extension.h>
#ifdef __APPLE__
#import <Foundation/Foundation.h>
#endif
#include "rasterizer_metal.h"

using namespace MetalRasterizer;

static std::unique_ptr<MetalRasterizer::MetalRasterizerImpl> g_metal_rasterizer;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizetrianglesCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const torch::Tensor& cam_pos,
    const float tan_fovx,
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const torch::Tensor& sh,
    const int degree,
    const bool prefiltered,
    const bool debug
) {
    if (!g_metal_rasterizer) {
        g_metal_rasterizer = std::make_unique<MetalRasterizer::MetalRasterizerImpl>();
        if (!g_metal_rasterizer->initialize()) {
            throw std::runtime_error("Failed to initialize Metal rasterizer");
        }
    }
    
    const int P = means3D.size(0);
    const int D = colors.size(1);
    const int M = 0; // Max SH degree
    const int R = 0; // Rendered features
    const int H = image_height;
    const int W = image_width;
    const float focal_x = (float)W / (2.0f * tan_fovx);
    const float focal_y = (float)H / (2.0f * tan_fovy);
    
    // Create output tensors
    auto out_color = torch::zeros({H, W, 3}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    auto out_depth = torch::zeros({H, W}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    auto out_alpha = torch::zeros({H, W}, torch::dtype(torch::kInt32).device(torch::kCPU));
    auto out_cum_alpha = torch::zeros({H, W}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    auto radii = torch::zeros({P}, torch::dtype(torch::kInt32).device(torch::kCPU));
    
    // Prepare arguments
    RasterizeArgs args;
    args.P = P;
    args.D = D;
    args.M = M;
    args.R = R;
    args.H = H;
    args.W = W;
    args.focal_x = focal_x;
    args.focal_y = focal_y;
    args.tan_fovx = tan_fovx;
    args.tan_fovy = tan_fovy;
    args.background = background.data_ptr<float>();
    args.means3D = means3D.data_ptr<float>();
    args.colors = colors.data_ptr<float>();
    args.opacity = opacity.data_ptr<float>();
    args.scales = scales.data_ptr<float>();
    args.rotations = rotations.data_ptr<float>();
    args.cov3D_precomp = cov3D_precomp.numel() > 0 ? cov3D_precomp.data_ptr<float>() : nullptr;
    args.viewmatrix = viewmatrix.data_ptr<float>();
    args.projmatrix = projmatrix.data_ptr<float>();
    args.cam_pos = cam_pos.data_ptr<float>();
    args.prefiltered = prefiltered;
    
    // Temporary buffers
    auto geomBuffer = torch::zeros({P * 256}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    auto binningBuffer = torch::zeros({P * 256}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    auto imgBuffer = torch::zeros({H * W * 256}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    
    // Call Metal rasterizer
    int result = g_metal_rasterizer->rasterize_triangles(
        args,
        out_color.data_ptr<float>(),
        out_depth.data_ptr<float>(),
        out_alpha.data_ptr<int>(),
        out_cum_alpha.data_ptr<float>(),
        radii.data_ptr<int>(),
        geomBuffer.data_ptr<float>(),
        binningBuffer.data_ptr<float>(),
        imgBuffer.data_ptr<float>()
    );
    
    if (result != 0) {
        throw std::runtime_error("Metal rasterization failed");
    }
    
    return std::make_tuple(out_color, out_depth, out_alpha, out_cum_alpha, radii);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizetrianglesBackwardCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const torch::Tensor& cam_pos,
    const float tan_fovx,
    const float tan_fovy,
    const torch::Tensor& radii,
    const torch::Tensor& sh,
    const int degree,
    const bool prefiltered,
    const bool debug,
    const torch::Tensor& grad_out_color,
    const torch::Tensor& grad_out_depth
) {
    if (!g_metal_rasterizer) {
        throw std::runtime_error("Metal rasterizer not initialized");
    }
    
    const int P = means3D.size(0);
    const int D = colors.size(1);
    const int M = 0;
    const int R = 0;
    const int H = grad_out_color.size(0);
    const int W = grad_out_color.size(1);
    const float focal_x = (float)W / (2.0f * tan_fovx);
    const float focal_y = (float)H / (2.0f * tan_fovy);
    
    // Create gradient tensors
    auto grad_means3D = torch::zeros({P, 3}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    auto grad_colors = torch::zeros({P, D}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    auto grad_opacity = torch::zeros({P}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    auto grad_scales = torch::zeros({P, 3}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    auto grad_rotations = torch::zeros({P, 4}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    auto grad_cov3D_precomp = torch::zeros({P, 6}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    
    // Prepare arguments
    RasterizeArgs args;
    args.P = P;
    args.D = D;
    args.M = M;
    args.R = R;
    args.H = H;
    args.W = W;
    args.focal_x = focal_x;
    args.focal_y = focal_y;
    args.tan_fovx = tan_fovx;
    args.tan_fovy = tan_fovy;
    args.background = background.data_ptr<float>();
    args.means3D = means3D.data_ptr<float>();
    args.colors = colors.data_ptr<float>();
    args.opacity = opacity.data_ptr<float>();
    args.scales = scales.data_ptr<float>();
    args.rotations = rotations.data_ptr<float>();
    args.cov3D_precomp = cov3D_precomp.numel() > 0 ? cov3D_precomp.data_ptr<float>() : nullptr;
    args.viewmatrix = viewmatrix.data_ptr<float>();
    args.projmatrix = projmatrix.data_ptr<float>();
    args.cam_pos = cam_pos.data_ptr<float>();
    args.prefiltered = prefiltered;
    
    // Temporary buffers
    auto geomBuffer = torch::zeros({P * 256}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    auto binningBuffer = torch::zeros({P * 256}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    auto imgBuffer = torch::zeros({H * W * 256}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    
    // Call Metal backward pass
    int result = g_metal_rasterizer->rasterize_triangles_backward(
        args,
        grad_out_color.data_ptr<float>(),
        grad_out_depth.data_ptr<float>(),
        radii.data_ptr<int>(),
        geomBuffer.data_ptr<float>(),
        binningBuffer.data_ptr<float>(),
        imgBuffer.data_ptr<float>(),
        grad_means3D.data_ptr<float>(),
        grad_colors.data_ptr<float>(),
        grad_opacity.data_ptr<float>(),
        grad_scales.data_ptr<float>(),
        grad_rotations.data_ptr<float>(),
        grad_cov3D_precomp.data_ptr<float>()
    );
    
    if (result != 0) {
        throw std::runtime_error("Metal backward pass failed");
    }
    
    return std::make_tuple(grad_means3D, grad_colors, grad_opacity, grad_scales, grad_rotations, grad_cov3D_precomp);
}

torch::Tensor markVisible(
    torch::Tensor& means3D,
    torch::Tensor& viewmatrix,
    torch::Tensor& projmatrix
) {
    // Simple visibility marking for Metal backend
    const int P = means3D.size(0);
    auto visible = torch::ones({P}, torch::dtype(torch::kBool).device(torch::kCPU));
    return visible;
}

torch::Tensor ComputeRelocationCUDA(
    torch::Tensor& opacity,
    torch::Tensor& scales,
    torch::Tensor& rotations,
    torch::Tensor& cov3D_precomp,
    torch::Tensor& means3D,
    torch::Tensor& viewmatrix,
    torch::Tensor& projmatrix,
    float focal_x,
    float focal_y,
    float tan_fovx,
    float tan_fovy,
    int image_height,
    int image_width,
    torch::Tensor& radii,
    int block_size
) {
    // Simple relocation computation for Metal backend
    const int P = means3D.size(0);
    auto relocation = torch::zeros({P}, torch::dtype(torch::kInt32).device(torch::kCPU));
    return relocation;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_triangles", &RasterizetrianglesCUDA);
    m.def("rasterize_triangles_backward", &RasterizetrianglesBackwardCUDA);
    m.def("mark_visible", &markVisible);
    m.def("compute_relocation", &ComputeRelocationCUDA);
}