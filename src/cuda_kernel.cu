
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "../inc/cuda_test.h"


template<typename T>
__global__ void  primitive_add(T *x, T *y, T *z, size_t num)
{
    int    i      = threadIdx.x + blockIdx.x * blockDim.x; 
    if (i < num) {
       z[i] = x[i] + y[i];
    }
}


template<typename T>
__global__ void  grid_stride_add_half(T *x, T *y, T *z, size_t num)
{
    int    i      = 0;
    int    idx    = threadIdx.x + blockIdx.x * blockDim.x;   
    size_t stride = blockDim.x * gridDim.x; 

    for (i = idx; i < num; i += stride){
        z[i] = x[i] + y[i];
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
        p_z[i] =  p_x[i] + p_y[i];
    }
}


struct half2_float {
    half2 x;
    half2 y;
};

struct half2_double {
    half2 x;
    half2 y;
    half2 w;
    half2 z;
};


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
        
        dst_z[i].x = __hadd2(p_x1, p_y1);
        dst_z[i].y = __hadd2(p_x2, p_y2);
    }
}



template<typename T>
__global__ void  grid_stride_add_half2_vec4_ld(T *x, T *y, T *z, size_t num)
{
    int    i      = 0;
    int    idx    = threadIdx.x + blockIdx.x * blockDim.x;   
    size_t stride = blockDim.x  * gridDim.x;
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



template<typename T>
__global__ void  grid_stride_add_half2_vec2_float_ld(T *x, T *y, T *z, size_t num)
{
    int    i      = 0;
    int    idx    = threadIdx.x + blockIdx.x * blockDim.x;   
    size_t stride = blockDim.x  * gridDim.x; 
    size_t loop   = num >> 2;
    
    struct half2_float *dst_z = (half2_float *)z;
    float2 *src_x = (float2 *)x;
    float2 *src_y = (float2 *)y;
    float2 tmp1, tmp2; 
    
    for (i = idx; i < loop; i += stride){
        tmp1 = src_x[i];
        tmp2 = src_y[i];
        dst_z[i].x = __hadd2(__float2half2_rn(tmp1.x), __float2half2_rn(tmp2.x));
        dst_z[i].y = __hadd2(__float2half2_rn(tmp1.y), __float2half2_rn(tmp2.y));
    }
}


template<typename T>
__global__ void  grid_stride_add_half2_vec4_float_ld(T *x, T *y, T *z, size_t num)
{
    int    i      = 0;
    int    idx    = threadIdx.x + blockIdx.x * blockDim.x;   
    size_t stride = blockDim.x  * gridDim.x; 
    size_t loop   = num >> 3;
    
    struct half2_double *dst_z = (half2_double *)z;
    float4 *src_x = (float4 *)x;
    float4 *src_y = (float4 *)y;
    float4 tmp1, tmp2; 
    
    for (i = idx; i < loop; i += stride){
        tmp1 = src_x[i];
        tmp2 = src_y[i];
        dst_z[i].x = __hadd2(__float2half2_rn(tmp1.x), __float2half2_rn(tmp2.x));
        dst_z[i].y = __hadd2(__float2half2_rn(tmp1.y), __float2half2_rn(tmp2.y));
        dst_z[i].w = __hadd2(__float2half2_rn(tmp1.w), __float2half2_rn(tmp2.w));
        dst_z[i].z = __hadd2(__float2half2_rn(tmp1.z), __float2half2_rn(tmp2.z));
    }
}




template<typename T>
__global__ void  grid_stride_add_vec2_float_version2(T *x, T *y, T *z, size_t num)
{
    int    i      = 0;
    int    idx    = threadIdx.x + blockIdx.x * blockDim.x;   
    size_t stride = blockDim.x  * gridDim.x; 
    size_t loop   = num >> 2;
    
    float2 *src_x = reinterpret_cast<float2 *>(x);
    float2 *src_y = reinterpret_cast<float2 *>(y);
    float2 *dst_z = reinterpret_cast<float2 *>(z);
    float2 x_h4, y_h4, z_h4;
    
    for (i = idx; i < loop; i += stride){
        x_h4 = src_x[i];
        y_h4 = src_y[i];

        half2 *x_h2 = reinterpret_cast<half2 *>(&x_h4);
        half2 *y_h2 = reinterpret_cast<half2 *>(&y_h4);
        half2 *z_h2 = reinterpret_cast<half2 *>(&z_h4);
        z_h2[0] = __hadd2(x_h2[0], y_h2[0]);
        z_h2[1] = __hadd2(x_h2[1], y_h2[1]);

        dst_z[i] = z_h4;
    }
}



template<typename T>
__global__ void  grid_stride_add_vec4_float_version2(T *x, T *y, T *z, size_t num)
{
    int    i      = 0;
    int    idx    = threadIdx.x + blockIdx.x * blockDim.x;   
    size_t stride = blockDim.x  * gridDim.x; 
    size_t loop   = num >> 3;
    
    float4 *src_x = reinterpret_cast<float4 *>(x);
    float4 *src_y = reinterpret_cast<float4 *>(y);
    float4 *dst_z = reinterpret_cast<float4 *>(z);
    float4 x_h8, y_h8, z_h8;
    
    for (i = idx; i < loop; i += stride){
        x_h8 = src_x[i];
        y_h8 = src_y[i];

        half2 *x_h2 = reinterpret_cast<half2 *>(&x_h8);
        half2 *y_h2 = reinterpret_cast<half2 *>(&y_h8);
        half2 *z_h2 = reinterpret_cast<half2 *>(&z_h8);
        z_h2[0] = __hadd2( x_h2[0], y_h2[0] );
        z_h2[1] = __hadd2( x_h2[1], y_h2[1] );
        z_h2[2] = __hadd2( x_h2[2], y_h2[2] );
        z_h2[3] = __hadd2( x_h2[3], y_h2[3] );

        dst_z[i] = z_h8;
    }
}


template __global__ void  primitive_add<DATA_TYPE>(DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *z, size_t num);
template __global__ void  grid_stride_add_half<DATA_TYPE>(DATA_TYPE *x, DATA_TYPE *y,  DATA_TYPE *z, size_t num);
template __global__ void  grid_stride_add_half2<DATA_TYPE>(DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *z, size_t num);
template __global__ void  grid_stride_add_half2_vec2_ld<DATA_TYPE>(DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *z, size_t num);
template __global__ void  grid_stride_add_half2_vec4_ld<DATA_TYPE>(DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *z, size_t num);
template __global__ void  grid_stride_add_half2_vec2_float_ld<DATA_TYPE>(DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *z, size_t num);
template __global__ void  grid_stride_add_half2_vec4_float_ld<DATA_TYPE>(DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *z, size_t num);
template __global__ void  grid_stride_add_vec2_float_version2<DATA_TYPE>(DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *z, size_t num);
template __global__ void  grid_stride_add_vec4_float_version2<DATA_TYPE>(DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *z, size_t num);