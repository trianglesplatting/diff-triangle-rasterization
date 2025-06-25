#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2024, University of Liege, KAUST and University of Oxford
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension
import os
import sys
import platform
import torch

def get_extension():
    """Determine which extension to build based on platform and CUDA availability"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    glm_include = os.path.join(base_dir, "third_party/glm/")
    
    # Check if we're on Apple Silicon
    is_apple_silicon = (platform.system() == "Darwin" and 
                       (platform.machine() == "arm64" or "arm" in platform.processor().lower()))
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available() and not is_apple_silicon
    
    if cuda_available:
        # CUDA backend (original)
        return CUDAExtension(
            name="diff_triangle_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu", 
                "cuda_rasterizer/backward.cu",
                "cuda_rasterizer/utils.cu",
                "rasterize_points.cu",
                "ext.cpp"
            ],
            extra_compile_args={
                "nvcc": ["-I" + glm_include, "--use_fast_math"],
                "cxx": ["-I" + glm_include]
            }
        )
    
    elif is_apple_silicon:
        # Metal backend for Apple Silicon
        return CppExtension(
            name="diff_triangle_rasterization._C",
            sources=[
                "metal_rasterizer/rasterizer_metal.mm",
                "ext_metal.mm"
            ],
            include_dirs=[glm_include, "metal_rasterizer/"],
            extra_compile_args={
                "cxx": ["-std=c++17", "-I" + glm_include, "-DUSE_METAL=1", "-ObjC++"]
            },
            extra_link_args=[
                "-framework", "Metal",
                "-framework", "MetalKit", 
                "-framework", "MetalPerformanceShaders",
                "-framework", "Foundation"
            ]
        )
    
    else:
        # CPU fallback
        extra_args = ["-std=c++17", "-I" + glm_include, "-DUSE_CPU=1"]
        extra_link_args = []
        
        # Try to enable OpenMP
        try:
            import subprocess
            result = subprocess.run(["gcc", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                extra_args.extend(["-fopenmp", "-DUSE_OPENMP=1"])
                extra_link_args.append("-fopenmp")
        except:
            pass
            
        return CppExtension(
            name="diff_triangle_rasterization._C",
            sources=[
                "cpu_rasterizer/rasterizer_cpu.cpp",
                "ext_cpu.cpp"
            ],
            include_dirs=[glm_include, "cpu_rasterizer/"],
            extra_compile_args={"cxx": extra_args},
            extra_link_args=extra_link_args
        )

setup(
    name="diff_triangle_rasterization",
    packages=['diff_triangle_rasterization'],
    ext_modules=[get_extension()],
    cmdclass={
        'build_ext': BuildExtension
    }
)
