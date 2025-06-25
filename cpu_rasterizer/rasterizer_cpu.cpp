/*
 * CPU fallback implementation for triangle rasterization
 * Copyright (C) 2024 - Mac M1 Port
 */

#include "rasterizer_cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

namespace CPURasterizer {

// Spherical harmonics constants (from original CUDA code)
constexpr float SH_C0 = 0.28209479177387814f;
constexpr float SH_C1 = 0.4886025119029199f;
constexpr float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f
};

CPURasterizer::CPURasterizer() {
    // Initialize temporary buffers
    temp_buffer.reserve(1024 * 1024);
    tile_ranges.reserve(1024);
    point_list.reserve(1024 * 1024);
}

CPURasterizer::~CPURasterizer() {
    // Cleanup handled by destructors
}

glm::vec3 CPURasterizer::computeColorFromSH(int idx, int deg, int max_coeffs, 
                                           const glm::vec3& means, const glm::vec3& campos, 
                                           const float* shs, bool* clamped) {
    glm::vec3 pos = means;
    glm::vec3 dir = glm::normalize(pos - campos);
    
    const glm::vec3* sh = reinterpret_cast<const glm::vec3*>(shs) + idx * max_coeffs;
    glm::vec3 result = SH_C0 * sh[0];
    
    if (deg > 0) {
        float x = dir.x;
        float y = dir.y;
        float z = dir.z;
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];
        
        if (deg > 1) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result = result +
                SH_C2[0] * xy * sh[4] +
                SH_C2[1] * yz * sh[5] +
                SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
                SH_C2[3] * xz * sh[7] +
                SH_C2[4] * (xx - yy) * sh[8];
        }
    }
    
    result += 0.5f;
    
    // Clamp colors
    if (clamped) {
        *clamped = (result.x < 0 || result.y < 0 || result.z < 0 ||
                   result.x > 1 || result.y > 1 || result.z > 1);
    }
    
    return glm::clamp(result, 0.0f, 1.0f);
}

glm::mat3 CPURasterizer::computeCov2D(const glm::vec3& mean, float focal_x, float focal_y, 
                                     float tan_fovx, float tan_fovy, const float* cov3D, 
                                     const float* viewmatrix) {
    // Transform point to camera space
    glm::mat4 W = glm::mat4(
        viewmatrix[0], viewmatrix[4], viewmatrix[8], viewmatrix[12],
        viewmatrix[1], viewmatrix[5], viewmatrix[9], viewmatrix[13],
        viewmatrix[2], viewmatrix[6], viewmatrix[10], viewmatrix[14],
        viewmatrix[3], viewmatrix[7], viewmatrix[11], viewmatrix[15]
    );
    
    glm::vec4 p_hom = W * glm::vec4(mean, 1.0f);
    glm::vec3 p_view = glm::vec3(p_hom) / p_hom.w;
    
    // Compute Jacobian of perspective projection
    float t = p_view.z;
    float limx = 1.3f * tan_fovx;
    float limy = 1.3f * tan_fovy;
    float txtz = t * t;
    float x = p_view.x;
    float y = p_view.y;
    
    glm::mat3 J = glm::mat3(
        focal_x / t, 0.0f, -(focal_x * x) / txtz,
        0.0f, focal_y / t, -(focal_y * y) / txtz,
        0.0f, 0.0f, 0.0f
    );
    
    // Transform covariance to camera space
    glm::mat3 W3 = glm::mat3(W);
    glm::mat3 Vrk = glm::mat3(
        cov3D[0], cov3D[1], cov3D[2],
        cov3D[3], cov3D[4], cov3D[5],
        cov3D[6], cov3D[7], cov3D[8]
    );
    
    glm::mat3 T = W3 * Vrk * glm::transpose(W3);
    glm::mat3 cov2D = J * T * glm::transpose(J);
    
    return cov2D;
}

void CPURasterizer::computeCov3D(const glm::vec3& scale, float mod, const glm::vec4& rot, 
                                float* cov3D) {
    // Build rotation matrix from quaternion
    float r = rot.x;
    float x = rot.y;
    float y = rot.z;
    float z = rot.w;
    
    glm::mat3 R = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );
    
    glm::mat3 S = glm::mat3(
        mod * scale.x, 0, 0,
        0, mod * scale.y, 0,
        0, 0, mod * scale.z
    );
    
    glm::mat3 M = S * R;
    glm::mat3 Sigma = glm::transpose(M) * M;
    
    cov3D[0] = Sigma[0][0];
    cov3D[1] = Sigma[0][1];
    cov3D[2] = Sigma[0][2];
    cov3D[3] = Sigma[1][0];
    cov3D[4] = Sigma[1][1];
    cov3D[5] = Sigma[1][2];
    cov3D[6] = Sigma[2][0];
    cov3D[7] = Sigma[2][1];
    cov3D[8] = Sigma[2][2];
}

int CPURasterizer::rasterize_triangles(
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
    const int P = args.P;
    const int H = args.H;
    const int W = args.W;
    
    // Initialize output
    std::fill_n(out_color, H * W * 3, 0.0f);
    if (out_depth) std::fill_n(out_depth, H * W, 0.0f);
    
    // Copy background
    if (args.background) {
        for (int i = 0; i < H * W; i++) {
            out_color[i * 3 + 0] = args.background[0];
            out_color[i * 3 + 1] = args.background[1];
            out_color[i * 3 + 2] = args.background[2];
        }
    }
    
    // Simple CPU rasterization - process each point
    glm::vec3 campos(args.cam_pos[0], args.cam_pos[1], args.cam_pos[2]);
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int idx = 0; idx < P; idx++) {
        // Get point data
        glm::vec3 p_world(args.means3D[idx * 3], args.means3D[idx * 3 + 1], args.means3D[idx * 3 + 2]);
        
        // Transform to screen space
        glm::vec4 p_hom = glm::vec4(p_world, 1.0f);
        
        // Simple orthographic projection for now
        float x_screen = (p_world.x + 1.0f) * 0.5f * W;
        float y_screen = (p_world.y + 1.0f) * 0.5f * H;
        
        int px = static_cast<int>(x_screen);
        int py = static_cast<int>(y_screen);
        
        if (px >= 0 && px < W && py >= 0 && py < H) {
            int pixel_idx = py * W + px;
            
            // Simple alpha blending
            float alpha = args.opacity ? args.opacity[idx] : 1.0f;
            
            if (args.colors) {
                out_color[pixel_idx * 3 + 0] = args.colors[idx * args.D + 0] * alpha;
                out_color[pixel_idx * 3 + 1] = args.colors[idx * args.D + 1] * alpha;
                out_color[pixel_idx * 3 + 2] = args.colors[idx * args.D + 2] * alpha;
            }
            
            if (radii) radii[idx] = 1;
        } else {
            if (radii) radii[idx] = 0;
        }
    }
    
    return 0;
}

int CPURasterizer::rasterize_triangles_backward(
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
    const int P = args.P;
    const int H = args.H;
    const int W = args.W;
    
    // Initialize gradients
    if (grad_means3D) std::fill_n(grad_means3D, P * 3, 0.0f);
    if (grad_colors) std::fill_n(grad_colors, P * args.D, 0.0f);
    if (grad_opacity) std::fill_n(grad_opacity, P, 0.0f);
    if (grad_scales) std::fill_n(grad_scales, P * 3, 0.0f);
    if (grad_rotations) std::fill_n(grad_rotations, P * 4, 0.0f);
    if (grad_cov3D_precomp) std::fill_n(grad_cov3D_precomp, P * 6, 0.0f);
    
    // Simple backward pass
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int idx = 0; idx < P; idx++) {
        if (!radii || radii[idx] == 0) continue;
        
        // Simple gradient computation
        glm::vec3 p_world(args.means3D[idx * 3], args.means3D[idx * 3 + 1], args.means3D[idx * 3 + 2]);
        
        float x_screen = (p_world.x + 1.0f) * 0.5f * W;
        float y_screen = (p_world.y + 1.0f) * 0.5f * H;
        
        int px = static_cast<int>(x_screen);
        int py = static_cast<int>(y_screen);
        
        if (px >= 0 && px < W && py >= 0 && py < H) {
            int pixel_idx = py * W + px;
            
            // Backpropagate color gradients
            if (grad_colors && args.colors) {
                float alpha = args.opacity ? args.opacity[idx] : 1.0f;
                grad_colors[idx * args.D + 0] = grad_out_color[pixel_idx * 3 + 0] * alpha;
                grad_colors[idx * args.D + 1] = grad_out_color[pixel_idx * 3 + 1] * alpha;
                grad_colors[idx * args.D + 2] = grad_out_color[pixel_idx * 3 + 2] * alpha;
            }
            
            // Backpropagate opacity gradients
            if (grad_opacity && args.colors) {
                grad_opacity[idx] = 
                    grad_out_color[pixel_idx * 3 + 0] * args.colors[idx * args.D + 0] +
                    grad_out_color[pixel_idx * 3 + 1] * args.colors[idx * args.D + 1] +
                    grad_out_color[pixel_idx * 3 + 2] * args.colors[idx * args.D + 2];
            }
        }
    }
    
    return 0;
}

} // namespace CPURasterizer