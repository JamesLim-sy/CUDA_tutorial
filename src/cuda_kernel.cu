#include "cuda_test.h"

template<typename T>
__global__ void  primitive_add(T *x, T *y, T *z, size_t num)
{
    int    i      = threadIdx.x + blockIdx.x * blockDim.x;   
    size_t stride = blockDim.x * gridDim.x ; 
    if (i < num) {
       z[i] = __hadd(x[i], y[i]);
    }
}


template<typename T>
__global__ void  grid_stride_add_half(T *x, T *y, T *z, size_t num)
{
    int    i      = 0;
    int    idx    = threadIdx.x + blockIdx.x * blockDim.x;   
    size_t stride = blockDim.x * gridDim.x; 
    for (i = idx; i < num; i += stride){
        z[i] = __hadd(x[i], y[i]);
    }
}


template<typename T>
__global__ void  grid_stride_add_half2(T *x, T *y, T *z, size_t num)
{
    int    i      = 0;
    int    idx    = threadIdx.x + blockIdx.x * blockDim.x;   
    size_t stride = blockDim.x * gridDim.x; 
    size_t loop   = num >> 1;

    __half2 *p_x = (__half2 *)x;
    __half2 *p_y = (__half2 *)y;
    __half2 *p_z = (__half2 *)z;

    for (i = idx; i < loop; i += stride){
        p_z[i] = __hadd2(p_x[i], p_y[i]);
    }
}


template<typename T>
__global__ void  grid_stride_add_half2_vec2_ld(T *x, T *y, T *z, size_t num)
{
    int    i      = 0;
    int    idx    = threadIdx.x + blockIdx.x * blockDim.x;   
    size_t stride = blockDim.x  * gridDim.x; 
    size_t loop   = num >> 2;
    
    __half2 p_x1, p_y1, p_x2, p_y2;
    struct half2_float *dst_z = (half2_float *)z;
    struct half2_float *src_x = (half2_float *)x;
    struct half2_float *src_y = (half2_float *)y;
    
    for (i = idx; i < loop; i += stride){
        p_x1 = src_x[i].x;
        p_y1 = src_y[i].x;
        p_x2 = src_x[i].y;
        p_y2 = src_y[i].y;
        
        dst_z[i].x = p_x1 + p_y1;
        dst_z[i].y = p_x2 + p_y2;
    }
}


template<typename T>
__global__ void  grid_stride_add_half2_vec4_ld(T *x, T *y, T *z, size_t num)
{
    int    i      = 0;
    int    idx    = threadIdx.x + blockIdx.x * blockDim.x;   
    size_t stride = blockDim.x  * gridDim.x; e
    size_t loop   = num >> 3;
    
    __half2 p_x1, p_y1, p_x2, p_y2;
    __half2 p_x3, p_y3, p_x4, p_y4;
    struct half2_double *dst_z = (half2_double *)z;
    struct half2_double *src_x = (half2_double *)x;
    struct half2_double *src_y = (half2_double *)y;
    
    for (i = idx; i < loop; i += stride){
        p_x1 = src_x[i].x;
        p_y1 = src_y[i].x;
        p_x2 = src_x[i].y;
        p_y2 = src_y[i].y;
        
        p_x3 = src_x[i].w;
        p_y3 = src_y[i].w;
        p_x4 = src_x[i].z;
        p_y4 = src_y[i].z;
        
        dst_z[i].x = __hadd2(p_x1, p_y1);
        dst_z[i].y = __hadd2(p_x2, p_y2);
        dst_z[i].w = __hadd2(p_x3, p_y3);
        dst_z[i].z = __hadd2(p_x4, p_y4);
    }
}