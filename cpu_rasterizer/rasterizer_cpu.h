/*
 * CPU fallback for triangle rasterization
 * Copyright (C) 2024 - Mac M1 Port
 */

#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace CPURasterizer {

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

class CPURasterizer {
private:
    std::vector<float> temp_buffer;
    std::vector<int> tile_ranges;
    std::vector<uint64_t> point_list;
    
    // Helper functions
    glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, 
                                const glm::vec3& means, const glm::vec3& campos, 
                                const float* shs, bool* clamped);
    
    glm::mat3 computeCov2D(const glm::vec3& mean, float focal_x, float focal_y, 
                          float tan_fovx, float tan_fovy, const float* cov3D, 
                          const float* viewmatrix);
    
    void computeCov3D(const glm::vec3& scale, float mod, const glm::vec4& rot, 
                     float* cov3D);
    
    void preprocessCPU(int P, int D, int M, const float* means3D, 
                      const glm::vec3& campos, const float* colors, 
                      const float* opacity, const float* scales, 
                      const glm::vec4* rot, const float* cov3D_precomp,
                      const float* viewmatrix, const float* projmatrix,
                      float tan_fovx, float tan_fovy, float focal_x, float focal_y,
                      int img_height, int img_width, int* radii, 
                      float2* means2D, float* depths, float* cov3Ds, 
                      float* rgb, float4* conic_opacity, const dim3& grid, 
                      uint32_t* tiles_touched, bool prefiltered);
    
    void renderCPU(const dim3& grid, const dim3& block, const RasterizeArgs& args,
                  uint2* ranges, uint32_t* point_list, int W, int H,
                  float2* means2D, float* colors, float4* conic_opacity,
                  float* final_T, uint32_t* n_contrib, const float* bg_color,
                  float* out_color);

public:
    CPURasterizer();
    ~CPURasterizer();
    
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

// CUDA-style types for compatibility
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int _x = 1, unsigned int _y = 1, unsigned int _z = 1) : x(_x), y(_y), z(_z) {}
};

struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };
struct uint2 { unsigned int x, y; };

} // namespace CPURasterizer