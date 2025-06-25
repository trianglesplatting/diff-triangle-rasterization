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

cmake_minimum_required(VERSION 3.20)

# Detect platform and available languages
if(APPLE)
    project(DiffRast LANGUAGES CXX OBJCXX)
    set(USE_CUDA OFF)
    set(USE_METAL ON)
else()
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        project(DiffRast LANGUAGES CUDA CXX)
        set(USE_CUDA ON)
        set(USE_METAL OFF)
    else()
        project(DiffRast LANGUAGES CXX)
        set(USE_CUDA OFF)
        set(USE_METAL OFF)
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

if(USE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
endif()

# Find OpenMP for CPU fallback
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Build appropriate backend
if(USE_CUDA)
    # CUDA backend (original)
    add_library(CudaRasterizer
        cuda_rasterizer/backward.h
        cuda_rasterizer/backward.cu
        cuda_rasterizer/forward.h
        cuda_rasterizer/forward.cu
        cuda_rasterizer/auxiliary.h
        cuda_rasterizer/rasterizer_impl.cu
        cuda_rasterizer/rasterizer_impl.h
        cuda_rasterizer/rasterizer.h
    )
    
    set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES "70;75;86")
    target_include_directories(CudaRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer)
    target_include_directories(CudaRasterizer PRIVATE third_party/glm ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    
    set(RASTERIZER_LIB CudaRasterizer)
    
elseif(USE_METAL AND APPLE)
    # Metal backend for Apple Silicon
    add_library(MetalRasterizer
        metal_rasterizer/rasterizer_metal.h
        metal_rasterizer/rasterizer_metal.mm
    )
    
    find_library(METAL_FRAMEWORK Metal)
    find_library(METALKIT_FRAMEWORK MetalKit)
    find_library(MPS_FRAMEWORK MetalPerformanceShaders)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    
    target_link_libraries(MetalRasterizer 
        ${METAL_FRAMEWORK} 
        ${METALKIT_FRAMEWORK} 
        ${MPS_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )
    
    target_include_directories(MetalRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/metal_rasterizer)
    target_include_directories(MetalRasterizer PRIVATE third_party/glm)
    target_compile_definitions(MetalRasterizer PRIVATE USE_METAL=1)
    
    set(RASTERIZER_LIB MetalRasterizer)
    
else()
    # CPU fallback
    add_library(CPURasterizer
        cpu_rasterizer/rasterizer_cpu.h
        cpu_rasterizer/rasterizer_cpu.cpp
    )
    
    if(OpenMP_CXX_FOUND)
        target_link_libraries(CPURasterizer OpenMP::OpenMP_CXX)
        target_compile_definitions(CPURasterizer PRIVATE USE_OPENMP=1)
    endif()
    
    target_include_directories(CPURasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpu_rasterizer)
    target_include_directories(CPURasterizer PRIVATE third_party/glm)
    target_compile_definitions(CPURasterizer PRIVATE USE_CPU=1)
    
    set(RASTERIZER_LIB CPURasterizer)
    
endif()

# Export the selected backend
set(DIFF_RAST_BACKEND ${RASTERIZER_LIB} PARENT_SCOPE)
