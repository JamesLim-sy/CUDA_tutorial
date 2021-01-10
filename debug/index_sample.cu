#include <iostream>
#include <memory>
#include <vector>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
using namespace std;

#define _DEBUG_ 1
#define T       double  
#define IndexT  int

#define CHECK_ERR(condi, val)  do{ \
    if(condi) {\
        fprintf(stderr, "[%s %d]: CUDA Runtime Error: %d\n", __func__, __LINE__, ret);\
        return ret; \
    }\
}while(0);

#define START_PRINT(data, width, height)  do {\
    for(int i = 0; i < height; i++) {  \
        std::cout<< ">>> "; \
        auto tmp = &(data[i * width]); \
        for(int j=0; j< width; ++j) { \
            std::cout << tmp[j] << " " ; \ 
        } \
        std::cout<< std::endl; \
    }\
    std::cout<< std::endl;\
}while(0);



template <typename T_, typename IndexT_=int >
__global__ void index_kernel_1(IndexT_ *p_index, T_ *p_input, T_ *p_output, 
                               size_t pitch_idx,
                               size_t pitch_src, 
                               size_t width_index, 
                               size_t height_index) { 
    int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    int   iy = blockDim.y * blockIdx.y + threadIdx.y;
    int   tid    = iy * pitch_idx + ix;
    int   tid_x  = iy * pitch_src + ix;
    int   tid_y  = iy * width_index + ix;

    if (ix < width_index & iy < height_index)  {
        IndexT_ idx     = p_index[tid];
        p_output[tid_y] = p_input[tid_x - ix + idx];
        #if _DEBUG_
        printf("tid : %d \t tid_y : %d \tp_index : %d \tp_input : %d \t p_output : %d\n",
                tid, tid_y, p_index[tid], p_input[tid_x - ix + idx], p_output[tid_y]);
        #endif
    }
}

template <typename T_, typename IndexT_=int >
__global__ void index_kernel_2(IndexT_ *p_index, T_ *p_input, T_ *p_output);


int main()
{
    int  height = 2;
    int  width_value = 100;
    int  width_index = 16;
    
    cudaStream_t stream_id;
    cudaError_t ret = (cudaError_t)0;
    std::vector<T>       vec_value(height * width_value);
    std::vector<T>      vec_output(height * width_index);
    std::vector<IndexT>  vec_index(height * width_index);
    std::vector<T>         vec_dst(height * width_index);
    
    size_t pitch_input  = 0;
    auto   byte_output = width_index * sizeof(T);
    auto   byte_value  = width_value * sizeof(T);
    auto   byte_index  = width_index * sizeof(IndexT);
    IndexT *g_index  = nullptr;
    T      *g_value  = nullptr;
    T      *g_output = nullptr;
    
    for(int i=0; i< height; ++i)  {
        auto tmp = &(vec_value[i * width_value]);
        for(int j=0; j<width_value; ++j) {tmp[j] = i + j * (i + 2); }
    }
    for(int i=0; i< height; ++i) {
        auto tmp = &(vec_index[i * width_index]);
        for(int j=0; j< width_index; ++j) { tmp[j] = (i + 1) * j + j; }
    }
    START_PRINT(vec_value, width_value, height);
    START_PRINT(vec_index, width_index, height);
    cudaStreamCreate(&stream_id);

    int  test_case = 1;
    switch (test_case) {
        case 1: {   // 2D  index sample, while 
            ret = cudaMallocPitch(
                (void **)&g_index, &pitch_input, 
                byte_index + byte_value, height);
            g_value = (T *)(size_t(g_index) + byte_index);
            
            ret = cudaMalloc((void **)&g_output, height * byte_output);
            ret = cudaMemcpy2DAsync(
                (void *)g_index, pitch_input,
                static_cast<void*>(vec_index.data()), byte_index, byte_index, height,
                cudaMemcpyHostToDevice, stream_id);
            ret = cudaMemcpy2DAsync(
                (void *)(size_t(g_index) + byte_index), pitch_input,
                static_cast<void*>(vec_value.data()), byte_value, byte_value, height,
                cudaMemcpyHostToDevice, stream_id);

            dim3 block_dim(width_index, width_index);
            dim3 gird_dim((width_index + block_dim.x - 1) / block_dim.x, 
                              (height  + block_dim.y - 1) / block_dim.y);
            // cout << (void *)(size_t(g_value) + width_output) << endl;
            // int thread_x = width_index < 128 ? width_index : 128;
            // int block_x  = floor(width_index / (thread_x));
            index_kernel_1 <<<gird_dim, block_dim, 0, stream_id>>>(
                                       g_index, g_value, g_output,
                                       pitch_input / sizeof(IndexT), 
                                       pitch_input / sizeof(T), 
                                       width_index,
                                       height);
            ret = cudaMemcpyAsync(
                static_cast<void*>(vec_dst.data()), (void *)g_output, height * byte_output, 
                cudaMemcpyDeviceToHost, stream_id);
            CHECK_ERR(ret != (cudaError_t)0, ret);
        }
        case 2 : {  
            std::vector<T>       vec_value(height * width_value);
            std::vector<T>      vec_output(height * width_index);
            std::vector<IndexT>  vec_index(height * width_index);
            std::vector<T>         vec_dst(height * width_index);    
            ret = cudaMalloc((void**)&g_index , height * byte_index);
            ret = cudaMalloc((void**)&g_value , height * byte_value);
            ret = cudaMalloc((void**)&g_output, height * byte_output);       
            cudaMemcpyAsync((void *)g_index, static_cast<void*>(vec_index.data()),
                            height * byte_index, cudaMemcpyHostToDevice, stream_id);
            cudaMemcpyAsync((void *)g_value, static_cast<void*>(vec_value.data()),
                            height * byte_value, cudaMemcpyHostToDevice, stream_id);

            int  block_dim = ;
            int  grid_dim  = ;
            index_kernel_2<<< >>>(g_index, g_value, g_output
                            );                   
            cudaMemcpyAsync((void *)g_output, static_cast<void*>(vec_output.data()),
                            height * byte_output, cudaMemcpyDeviceToHost, stream_id);
        }
    }
    ret = cudaStreamSynchronize(stream_id);
    START_PRINT(vec_dst, width_index, height);
    cudaFree(g_index);
    cudaFree(g_output);
}