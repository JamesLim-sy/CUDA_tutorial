#include <iostream>
#include <memory>
#include <vector>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
using namespace std;


#define T     int
#define Tndex int 
#define ALIGNED(val, align_)  ((val + align_ - 1) & (~(align_ - 1)))
#define CHECK_ERR(condi, val)  do{ \
    if(condi) {\
        fprintf(stderr, "[%s %d]: CUDA Runtime Error: %d\n", __func__, __LINE__, ret);\
        return ret; \
    }\
}while(0);


template <typename T, typename IndexT=int >
__global__ void index_kernel(IndexT *p_index, T *p_input, T *p_output, 
                             size_t pitch_idx,
                             size_t pitch_src, 
                             size_t pitch_dst
                             size_t width_index, 
                             size_t height_index) 
{ 
    int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    int   iy = blockDim.y * blockIdx.y + threadIdx.y;
    int   tid    = iy * pitch_idx + ix;
    int   tid_x  = ty * pitch_src + ix;
    int   tid_y  = iy * pitch_dst + ix;
    
    if (ix < width_index & iy < height_index)
    {
        T *idx          = p_index[tid];
        p_output[tid_y] = p_input[tid_x - ix + idx];
    }
}



int main()
{
    int  batch_size = 2;
    int  width_value = 64;
    int  width_index = 16;
    
    std::vector<T> vec_value(batch_size * width_value);
    std::vector<T> vec_output(batch_size * width_index);
    std::vector<T> vec_index(batch_size * width_index);
    
    for(int i=0; i< batch_size; ++i)
    {
        auto tmp = &(vec_value[i * width_value]);
        for(int j=0; j<width_value; ++j) {
            tmp[j] = i + j * (i + 1);
        }
    }
    for(int i=0; i< batch_size; ++i)
    {
        auto tmp = &(vec_index[i * width_index]);
        for(int j=0; j< width_index; ++j) {
            tmp[j] = j * i + j * j + 1;
        }
    }
    for(int i=0; i< batch_size; ++i)
    {
        std::cout<< ">>> ";
        auto tmp = &(vec_value[i * width_value]);
        for(int j=0; j<width_value; ++j) {
            std::cout << tmp[j] << " " ;
        }
        std::cout<< std::endl;
    }
    std::cout<< std::endl;

    for(int i=0; i< batch_size; ++i)
    {
        auto tmp = &(vec_index[i * width_index]);
        for(int j=0; j< width_index; ++j) {
            std::cout << tmp[j] << " " ;
        }
        std::cout<< std::endl;
    }
    std::cout<< std::endl;

    cudaStream_t stream_id;
    T *g_input  = nullptr;
    T *g_output = nullptr;

    size_t pitch_input  = 0;
    size_t pitch_output = 0;
    auto width_output = width_index * sizeof(T);
    auto width_input  = (width_value + width_index) * sizeof(T);

    cudaError_t ret = (cudaError_t)0;
    ret = cudaStreamCreate(&stream_id);
    CHECK_ERR(ret != (cudaError_t)0, ret);

    ret = cudaMallocPitch((void **)&(g_input),
                &pitch_input, width_input, batch_size);
    CHECK_ERR(ret != (cudaError_t)0, ret);

    ret = cudaMalloc((void **)&(g_output), batch_size * width_index * sizeof(T));
    CHECK_ERR(ret != (cudaError_t)0, ret);


    pitch_input = width_input > pitch_input ? width_input : pitch_input;
    std::vector<T> test(batch_size * (width_value + width_index) );
    std::vector<T> dst(batch_size * (width_index));
    
    ret = cudaMemcpy2D(g_input, 
                       pitch_input,
                       static_cast<void*>(vec_index.data()),
                       width_output, 
                       width_output, 
                       batch_size,
                       cudaMemcpyHostToDevice);
    CHECK_ERR(ret != (cudaError_t)0, ret);

    // cout << (void *)(size_t(g_input) + width_output) << endl;
    ret = cudaMemcpy2D((void *)(size_t(g_input) + width_output),
                        pitch_input,
                        static_cast<void*>(vec_value.data()),
                        width_value * sizeof(T), // Pitch of source memory
                        width_value * sizeof(T), 
                        batch_size,
                        cudaMemcpyHostToDevice);
    CHECK_ERR(ret != (cudaError_t)0, ret);
    
    ret = cudaMemcpy2D(static_cast<void*>(test.data()), 
                       width_input,
                       g_input, 
                       pitch_input, // Pitch of source memory
                       width_input, 
                       batch_size,
                       cudaMemcpyDeviceToHost);
    CHECK_ERR(ret != (cudaError_t)0, ret);

    int thread_x = width_index < 128 ? width_index : 128;
    int block_x  = floor(width_index / (thread_x));

    dim3 block_dim(width_index, width_index);
    dim3 gird_dim( (width_index + block_dim.x - 1) / block_dim.x, 
                   (batch_size  + block_dim.y - 1) / block_dim.y);
    cout << block_x << "\t" << thread_x << "\t" << batch_size << endl;

    index_kernel <<<gird_dim, block_dim, 0, stream_id >>>(g_input,
                                                          ((T *)(size_t(g_input) + width_output)),
                                                          g_output,
                                                          pitch_input,
                                                          width_index,  
                                                          batch_size);

    ret = cudaMemcpy2D(static_cast<void*>(dst.data()), 
                       width_output,
                       g_output, 
                       width_output, // Pitch of source memory
                       width_output, 
                       batch_size,
                       cudaMemcpyDeviceToHost);
    CHECK_ERR(ret != (cudaError_t)0, ret);
    
    for(int i = 0; i < batch_size * width_index; ++i)
    {   
        cout << dst[i] << " ";
        if (i && (i + 1) % width_index == 0) {
            cout << endl;
        }
    }
    std::cout<< std::endl;

    cudaFree(g_input);
    cudaFree(g_output);
}