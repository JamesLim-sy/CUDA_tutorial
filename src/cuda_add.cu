#include <stdint.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
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
    __syncthreads();  
}


__global__ void  grid_stride_add(float *x, float *y, float *z, size_t num)
{
    int i   = 0;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;   
    size_t stride = blockDim.x * gridDim.x; 
    for (i = idx; i < num; i += stride){
       z[i] = x[i] + y[i];
    }
}


__global__ void  share_mem_add(float *x, float *y, float *z, size_t num)
{
    __shared__ float cache[2][256];

    int s   = 0;
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    cache[0][tid] = x[idx];
    cache[1][tid] = y[idx];
    for (s = idx; s < num; s += 256){
        cache[0][s] += cache[1][s];
    }
    // __syncthreads();
    z[0] = cache[0][0];
}


static int rslt_check_func(void   *p_z, 
                           size_t  nByte,
                           size_t  num, 
                           char   *str,
                           cudaEvent_t start,
                           cudaEvent_t stop)
{
    int ret = 0; 
    mem_pointer *z = (mem_pointer *)p_z;

    // To sync the mission accomplishement of GPU
    cudaMemcpy((void *)(z->p_cpu), (void *)(z->p_gpu), nByte, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);  

    float   max_err  = 0.0f; 
    float   timespan = 0.0f;
    cudaEventElapsedTime(&timespan, start, stop);

    for (int i = 0; i< num; ++i){
        max_err = fmax(max_err, fabs((z->p_cpu)[i] - 30.0));
        if (max_err > 0.1f){
            cout << "[ERR] value : " << (z->p_cpu)[i];
            cout << "\tindex : " << i << endl;
            return -10; 
        }
    }
    cout << "\tMax  Err  : " << max_err << "\tFunc Name : " << str;
    cout << "\t TimeSpan : " << timespan << " ms";
    cout << "\t Bandwidth(GB/s): " << (num * 4 * 3) / timespan / 1e6 << endl;
    ret = (fabs(max_err) > 0.1) ? -5 : 0;
    return ret;
}


static int analysis_grid_block(void  *p_x, 
                               void  *p_y, 
                               void  *p_z,
                               int    grid_num, 
                               int    block_num, 
                               size_t nByte)
{
    int   ret = (cudaError_t)0;  // which means success
    int     i = 0;
    size_t  N = nByte / sizeof(float); 

    dim3 blocksize(block_num);
    dim3 gridsize(grid_num);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate( &stop);

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

    {
        memset(z->p_cpu, nByte, 0);
        cudaMemset((void *)z->p_gpu, 0, nByte);
        cudaEventRecord(start);
        grid_stride_add<<< gridsize, blocksize >>> (x->p_gpu, y->p_gpu, z->p_gpu, N);
        cudaEventRecord(stop);
    }
    ret = rslt_check_func(z, nByte, N, "grid-stride loop", start, stop);
    CHECK_ERR(ret != 0, ret);
    return 0;
}


static int analysis_no_grid_block(void  *p_x, 
                                void  *p_y, 
                                void  *p_z,
                                int    grid_num, 
                                int    block_num, 
                                size_t nByte)
{
    int   ret = (cudaError_t)0;  // which means success
    int     i = 0;
    size_t  N = nByte / sizeof(float); 

    dim3 blocksize(block_num);
    dim3 gridsize(grid_num);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate( &stop);

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

    {
#if 1
        memset(z->p_cpu, nByte, 0);
        cudaMemset((void *)z->p_gpu, 0, nByte);
        cudaEventRecord(start);
        primitive_add <<< gridsize, blocksize >>> (x->p_gpu, y->p_gpu, z->p_gpu, N);
        cudaEventRecord(stop);
        ret = rslt_check_func(z, nByte, N, "primitive_add", start, stop);
#else
        memset(z->p_cpu, nByte, 0);
        cudaMemset((void *)z->p_gpu, 0, nByte);
        cudaEventRecord(start);
        share_mem_add<<< gridsize, blocksize >>> (x->p_gpu, y->p_gpu, z->p_gpu, N);
        cudaEventRecord(stop);
        ret = rslt_check_func(z, nByte, N, "share_mem_add", start, stop);
#endif
    }
    CHECK_ERR(ret != 0, ret);

    return 0;
}


int main()
{
    int     i = 0, j = 0;
    size_t  N = 1<<24;
    size_t  nByte = N * sizeof(float);
    int     ret = (cudaError_t)0;  // which means success

    const size_t block_num_lo = 32;
    const size_t block_num_up = 2048;
    const size_t grid_num_lo = 1024;
    const size_t grid_num_up = 1024<<7; 

    mem_pointer x,y,z;

    x.p_cpu = (float *)malloc(nByte);
    CHECK_ERR(x.p_cpu == NULL, -5);

    y.p_cpu = (float *)malloc(nByte);
    CHECK_ERR(y.p_cpu == NULL, -5);

    z.p_cpu = (float *)malloc(nByte);
    CHECK_ERR(z.p_cpu == NULL, -5);

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
            ret = analysis_grid_block(&x, &y, &z, j, i, nByte);
            CHECK_ERR(ret != 0, -5);
        }
        cout << endl;
    }

    j = 0;
    for (i = block_num_lo; i < block_num_up; i <<= 1){  
        j = N / i; 
        cout << "[GRID  NUM] : " << j;
        cout << "[BLOCK NUM] : " << i; 
        ret = analysis_no_grid_block(&x, &y, &z, j, i, nByte);
        CHECK_ERR(ret != 0, -5);
        cout << endl;
    }

    cudaFree(x.p_gpu);
    cudaFree(y.p_gpu);
    cudaFree(z.p_gpu);
    free(x.p_cpu);
    free(y.p_cpu);
    free(z.p_cpu);    

    return  0;
}
