#include <stdint.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#define CHECK_ERR  std::cout<<"HERE : "<< __LINE__ << std::endl;

// Primitive 
// Thread work
__global__ void  primitive_add(float *x, float *y, float *z, size_t num)
{
    // blockIdx.x : indicates the horizontal index in grid.
    // blockDim.x : indicates the horizontal total thread numbers inside a block.
    // gridDim.x  : indicates the hotizontal total block  numbers inside  a grid.
    int i   = 0;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;   
    size_t stride = blockDim.x * gridDim.x;
    for (i = idx; i < num; i += stride){
       z[i] = x[i] + y[i];
    }
}


int main()
{
    cudaError_t ret = (cudaError_t)0;  // which means success
    size_t  N = 1<<20;
    size_t  nByte = N * sizeof(float);
    int     i = 0;
    float   max_err = 0.0f; 

    float *x = NULL;
    float *y = NULL;
    float *z = NULL;
    float *d_x = NULL;
    float *d_y = NULL;
    float *d_z = NULL;
    x = (float *)malloc(nByte);
    y = (float *)malloc(nByte);
    z = (float *)malloc(nByte);

    dim3 blocksize(256);
    dim3 gridsize( (N + blocksize.x - 1)/ blocksize.x );

    std :: cout << "nByte :" << nByte << std :: endl;
    ret = cudaMalloc((void **)&d_x, nByte);
    std :: cout <<"[LINE]:" << __LINE__ << "\t ret :" << ret << std:: endl;
    ret = cudaMalloc((void **)&d_y, nByte);
    ret = cudaMalloc((void **)&d_z, nByte);

    for (i = 0; i < N; ++i)
    {
        x[i] = 10.0;
        y[i] = 20.0;
    }
    cudaMemcpy((void *)d_x, (void *)x, nByte, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_y, (void *)y, nByte, cudaMemcpyHostToDevice);    
    primitive_add <<< gridsize, blocksize >>> (d_x, d_y, d_z, N);
    
    // To sync the mission accomplishement of GPU
    cudaDeviceSynchronize();  
    cudaMemcpy((void *)z, (void *)d_z, nByte, cudaMemcpyDeviceToHost);

    for (i = 0; i< N; ++i){
        max_err = fmax(max_err, fabs(z[i] - 30.0));
    }
    std::cout << "Max Err:" << max_err << std::endl;

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    return 1;
}
