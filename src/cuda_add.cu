#include <stdint.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
using namespace std;

typedef struct _mem_pointer_ {
    float *p_cpu = NULL;
    float *p_gpu = NULL;
} mem_pointer;


#ifndef CHECK_ERR
#define CHECK_ERR(condi, val) do{  \
    if (condi)  \
    {           \
        cout << "[LINE  FUNC]: " << __LINE__ << "\t" << __func__ << "\terr_val = " << val << endl; \
        return  -5;  \
    }               \
}while(0)
#endif


__global__ void  primitive_add(float *x, float *y, float *z, size_t num)
{
    // blockIdx.x : indicates the horizontal index in grid.
    // blockDim.x : indicates the horizontal total thread numbers inside a block.
    // gridDim.x  : indicates the hotizontal total block  numbers inside  a grid.
    int    i      = threadIdx.x + blockIdx.x * blockDim.x;   
    size_t stride = blockDim.x * gridDim.x ; 
    if (i < num) {
       z[i] = x[i] + y[i];
    }
}


__global__ void  grid_stride_add(float *x, float *y, float *z, size_t num)
{
    int    i      = 0;
    int    idx    = threadIdx.x + blockIdx.x * blockDim.x;   
    size_t stride = blockDim.x * gridDim.x; 
    
    for (i = idx; i < num; i += stride){
       z[i] = x[i] + y[i];
    }
}


__global__ void grid_stride_add_float2(float *p_x, float *p_y, float *p_z, size_t num) 
{
    int i   = 0;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;   
    size_t half_num   = num >> 1;
    size_t stride = blockDim.x * gridDim.x; 
    float2 *src_x = NULL, *src_y = NULL, *dst_z = NULL;
    
    for (i = idx; i < half_num; i += stride){
        src_x = &(((float2 *)p_x)[i]);
        src_y = &(((float2 *)p_y)[i]);
        dst_z = &(((float2 *)p_z)[i]);
        dst_z->x = src_x->x + src_y->x;
        dst_z->y = src_x->y + src_y->y;
    }
    if (idx == half_num && num % 2 == 1) {
        p_z[num - 1] = p_x[num - 1] + p_y[num - 1];
    }
}


__global__ void grid_stride_add_float4(float *p_x, float *p_y, float *p_z, size_t num) 
{
    int i    = 0;
    int tail = 0; 
    int idx  = threadIdx.x + blockIdx.x * blockDim.x;

    size_t quater_num = num >> 2;
    size_t stride = blockDim.x * gridDim.x;
    float4 *src_x = NULL, *src_y = NULL, *dst_z = NULL;
    
    for (i = idx; i < quater_num; i += stride) {
        src_x = &(((float4 *)p_x)[i]);
        src_y = &(((float4 *)p_y)[i]);
        dst_z = &(((float4 *)p_z)[i]);
        dst_z->x = src_x->x + src_y->x;
        dst_z->y = src_x->y + src_y->y;
        dst_z->z = src_x->z + src_y->z;
        dst_z->w = src_x->w + src_y->w;
    }

    tail = num % 4;
    while (idx == quater_num && tail) {
        p_z[num - tail] = p_x[num - tail] + p_y[num - tail];
        tail--;
    }
}


__global__ void  share_mem_add(float *x, float *y, float *z, size_t num)
{
    // __shared__ float cache[2][256];
    __shared__ float cache[256];

    int s   = 0;
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    cache[tid] = z[idx];
    for (s = idx; s < num; s += 256){
        cache[tid] += x[idx] + y[idx];
    }
    __syncthreads();
    z[idx] = cache[tid];
}


static size_t  mem_alloc_offset(float  **addr, 
                                int      align,
                                size_t   nByte)
{
    int     ret        = (cudaError_t)0;
    float  *p_dst      = NULL;
    size_t  mem_length = nByte + align;

    ret   = cudaMalloc((void **)addr, mem_length);
    CHECK_ERR(ret != 0, -5);

    p_dst = (float *)((size_t)(*addr) + align);
    CHECK_ERR(p_dst == NULL, -5);
    
    *addr = (float *)p_dst;
    return ret;
}


static int  rslt_check_func(void   *p_dst, 
                            size_t  nByte,
                            size_t  num, 
                            char   *str,
                            cudaEvent_t start,
                            cudaEvent_t stop,
                            size_t  loop)
{
    int     ret = 0; 
    float   eps = 1e-5;
    float   max_err  = 0.0f; 
    float   timespan = 0.0f;
    mem_pointer *m_dst = (mem_pointer *)p_dst;

    // To sync the mission accomplishement of GPU
    cudaMemcpy((void *)(m_dst->p_cpu), (void *)(m_dst->p_gpu), nByte, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);  
    cudaEventElapsedTime(&timespan, start, stop);

    for (int i = 0; i< num; ++i){
        max_err = fmax(max_err, fabs((m_dst->p_cpu)[i] - 30.0));
        if (max_err > eps) {
            cout << "[ERR] value : " << (m_dst->p_cpu)[i];
            cout << "\t[index] : " << i  << "\t[Loop]: " << loop << endl;
            return -10; 
        }
    }
    // cout << "\t[Max_Err]: " << max_err << " [FuncName]: " << str;
    // cout << "\t[TimeSpan]: " << timespan << " ms";
    // cout << "\t[Bwidth(GB/s)]: " << (num * sizeof(float) * 3) / timespan / 1e6 << endl;
    ret = (fabs(max_err) > 0.1) ? -5 : 0;
    return ret;
}


// 多模式状态
static int analysis_grid_block(void  *p_x, 
                               void  *p_y, 
                               void  *p_z,
                               int    grid_num, 
                               int    block_num, 
                               size_t nByte,
                               int    type)
{
    int   ret = (cudaError_t)0;  // which means success
    int     i = 0;
    size_t  N = nByte / sizeof(float); 
    size_t  loop = 0;
    float   alpha = 1.0;

    dim3 blocksize(block_num);
    dim3 gridsize(grid_num);

    cudaEvent_t start, stop;
    cublasHandle_t hdl;

    mem_pointer  *x = (mem_pointer *)p_x;
    mem_pointer  *y = (mem_pointer *)p_y;
    mem_pointer  *z = (mem_pointer *)p_z;

    for (i = 0; i < N; ++i)
    {
        x->p_cpu[i] = 10.0f;
        y->p_cpu[i] = 20.0f;
    }
    cudaMemcpy((void *)(x->p_gpu), (void *)(x->p_cpu), nByte, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)(y->p_gpu), (void *)(y->p_cpu), nByte, cudaMemcpyHostToDevice); 
    primitive_add <<< gridsize, blocksize >>> (x->p_gpu, y->p_gpu, z->p_gpu, N>>4);  // warm up

    while (loop < 1000)
    {
        cudaEventCreate(&start);
        cudaEventCreate( &stop);
        cudaMemset((void *)(z->p_gpu), 0, nByte);
        switch (type) {
            case 0 : {
                cudaEventRecord(start);
                grid_stride_add <<< gridsize, blocksize >>> (x->p_gpu, y->p_gpu, z->p_gpu, N);
                cudaEventRecord(stop);
                break;
            }
            case 1 : {
                cudaEventRecord(start);
                grid_stride_add_float2 <<< grid_num, block_num >>> (x->p_gpu, y->p_gpu, z->p_gpu, N);
                cudaEventRecord(stop);
                break;
            }
            case 2 : {
                cudaEventRecord(start);
                grid_stride_add_float4 <<< grid_num, block_num >>> (x->p_gpu, y->p_gpu, z->p_gpu, N);
                cudaEventRecord(stop);
                break;
            }
            case 3 :{
                cublasCreate(&hdl);

                cudaEventRecord(start);
                ret  = cublasSaxpy(hdl, N, &alpha, x->p_gpu, 1, y->p_gpu, 1);
                cudaEventRecord(stop);
                CHECK_ERR(ret != 0, ret);
                
                cudaMemcpy((void *)(z->p_gpu), (void *)(y->p_gpu), nByte, cudaMemcpyDeviceToDevice);
                cudaMemcpy((void *)(y->p_gpu), (void *)(y->p_cpu), nByte, cudaMemcpyHostToDevice); 
                cublasDestroy(hdl);
                break;
            }
            default : {
                cout << "[ERR]: line :" << __LINE__ << "\t __func__ :" <<  __func__ << endl;
            }
        }
        ret = rslt_check_func(z, nByte, N, "test", start, stop, loop);
        CHECK_ERR(ret != 0, ret);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);    
        loop++;
    }
    return 0;
}




int main(int argc, char *argv[])
{
    int     ret = (cudaError_t)0;  // which means success
    size_t  N = 1<<24;
    size_t  nByte = N * sizeof(float);
    size_t  block_num = 0;
    size_t  grid_num  = 0;
    int     i = 0, j = 0, init = 0;
    mem_pointer  x,y,z;
    
    const size_t block_num_lo = 32;
    const size_t block_num_up = 2048;
    const size_t grid_num_lo  = 1024;
    const size_t grid_num_up  = 1024<<7; 
    
    if (argc > 1) {
        init = atoi(argv[1]);
        cout << "init : " << init << endl;
    }
    x.p_cpu = (float *)malloc(nByte);
    CHECK_ERR(x.p_cpu == NULL, -5);

    y.p_cpu = (float *)malloc(nByte);
    CHECK_ERR(y.p_cpu == NULL, -5);

    z.p_cpu = (float *)malloc(nByte);
    CHECK_ERR(z.p_cpu == NULL, -5);

    switch  (init) {
        case 0:
        {
            ret = cudaMalloc((void **)&(x.p_gpu), nByte);
            CHECK_ERR(ret != 0, ret);
        
            ret = cudaMalloc((void **)&(y.p_gpu), nByte);
            CHECK_ERR(ret != 0, ret);

            ret = cudaMalloc((void **)&(z.p_gpu), nByte);
            CHECK_ERR(ret != 0, ret);
            cout << "nByte :" << nByte << endl;
        
            for (j = grid_num_lo; j < grid_num_up ; j <<= 1) {
                cout << "[GRID NUM] : " << j << endl;
                for (i = block_num_lo; i < block_num_up; i <<= 1){  
                    cout << "[BLOCK NUM] : " << i; 
                    ret = analysis_grid_block(&x, &y, &z, j, i, nByte, 0);
                    CHECK_ERR(ret != 0, -5);
                }
                cout << endl;
            }
            j = 0;
            for (i = block_num_lo; i < block_num_up; i <<= 1){  
                j = N / i; 
                cout << "[GRID  NUM] : " << j;
                cout << "[BLOCK NUM] : " << i; 
                ret = analysis_grid_block(&x, &y, &z, j, i, nByte, 0);
                CHECK_ERR(ret != 0, -5);
                cout << endl;
            }
            break;
        }

        case 1: 
        {
            const int aligned = 33;
            int offset = 0;
            
            block_num  = 256;
            grid_num   = 65536;
            for (i = 0; i < aligned; i++) 
            {
                offset = i * sizeof(float);
                ret = mem_alloc_offset(&(z.p_gpu), offset, nByte);
                CHECK_ERR(ret != 0, ret);
        
                ret = cudaMalloc((void **)&(y.p_gpu), nByte);
                CHECK_ERR(ret != 0, ret);
        
                ret = cudaMalloc((void **)&(x.p_gpu), nByte);
                CHECK_ERR(ret != 0, ret);
                
                cout << "[x.p_gpu]:" << x.p_gpu << " ";
                cout << "[y.p_gpu]:" << y.p_gpu << " ";
                cout << "[z.p_gpu]:" << z.p_gpu << " ";
                cout << "[offset ]:" << offset  << " "; 

                ret = analysis_grid_block(&x, &y, &z, grid_num, block_num, nByte, 0);
                CHECK_ERR(ret != 0, -5);
            }
            break;
        }
        case 2 : {
            block_num  = 256;
            grid_num   = 65536;
            
            ret = cudaMalloc((void **)&(x.p_gpu), nByte);
            CHECK_ERR(ret != 0, ret);
        
            ret = cudaMalloc((void **)&(y.p_gpu), nByte);
            CHECK_ERR(ret != 0, ret);
        
            ret = cudaMalloc((void **)&(z.p_gpu), nByte);
            CHECK_ERR(ret != 0, ret);
            cout << "nByte :" << nByte << endl;

            ret = analysis_grid_block(&x, &y, &z, grid_num, block_num, nByte, 0);
            ret = analysis_grid_block(&x, &y, &z, grid_num, block_num, nByte, 3);
            ret = analysis_grid_block(&x, &y, &z, grid_num / 2, block_num, nByte, 1);
            ret = analysis_grid_block(&x, &y, &z, grid_num / 4, block_num, nByte, 2);
            CHECK_ERR(ret != 0, -5);
            break;
        }
        default :{
            cout << "Please select one operation test !" << endl;
            return -5;
        }
    }

    cudaFree(x.p_gpu);
    cudaFree(y.p_gpu);
    cudaFree(z.p_gpu);
    free(x.p_cpu);
    free(y.p_cpu);
    free(z.p_cpu);    
    return  0;
}