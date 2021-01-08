#include <iostream>
#include <memory>
#include <vector>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
using namespace std;


// template <typename T>
// __global__ void index_kernel(T *p_index, T *p_input, T *p_output, 
//                              size_t pitch, size_t width, size_t height) 
// { 
//     int   ix = blockDim.x * blockIdx.x + threadIdx.x;
//     int   iy = blockDim.y * blockIdx.y + threadIdx.y;
//     int   tid    = iy * 128 + ix;
//     int   tid_x  = tid;
//     int   tid_y  = iy * 128 + ix;
    
//     // int  *p_src = reinterpret_cast<int *>(p_input);
//     // int  *p_idx = reinterpret_cast<int *>(p_index);
//     // int  *p_dst = reinterpret_cast<int *>(p_output);

//     // for (int i = tid; i < (width>>2) * height; i += (width>>2)) 
//     // if (ix < (width))
//     if (ix < width & iy < height)
//     {
//         // int4 idx_i4 = p_idx[tid];
//         // int4 out_i4;
//         // int  *idx_i = reinterpret_cast<int *>(&idx_i4);
//         // int  *out_i = reinterpret_cast<int *>(&out_i4);

//         // out_i[0] = p_src[0];
//         // out_i[1] = p_src[1];
//         // out_i[2] = p_src[2];
//         // out_i[3] = p_src[3];     
//         // p_dst[tid] = out_i4;
//         // int *p_src = &(p_input[tid]);
//         int idx    = p_index[tid];
//         int *p_src = &(p_input[tid]);
//         p_output[tid] = p_src[idx];//p_input[idx];
//         printf("tid : %d  \tidx : %d  \t p_input: %d  \tp_src : %d\n",
//                tid, idx, p_input[tid_y], p_src[idx]);
//     }
// }
template <typename T>
__global__ void index_kernel(T *p_index, T *p_input, T *p_output, 
                             size_t pitch, size_t width, size_t height) 
{ 
    int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    int   iy = blockDim.y * blockIdx.y + threadIdx.y;
    int   tid    = iy * 128 + ix;
    int   tid_x  = tid;
    int   tid_y  = iy * width + ix;
    
    if (ix < width & iy < height)
    {
        int idx         = p_index[tid];
        p_output[tid_y] = p_input[tid - ix + idx];
    }
}


#define T int
#ifndef ALIGNED
#define ALIGNED(val, align_)  ((val + align_ - 1) & (~(align_ - 1)))
#endif
#define CHECK_ERR(condi, val)  do{ \
    if(condi) {\
        fprintf(stderr, "[%s %d]: CUDA Runtime Error: %d\n", __func__, __LINE__, ret);\
        return ret; \
    }\
}while(0);

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
    
    // cout <<  "width_input  : " << width_input << endl <<  "width_output : " << width_output << endl;
    // cout <<  "pitch_input  : " << pitch_input << endl <<  "pitch_output : " << pitch_output << endl;
    // cout << g_input << endl;

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

    // index_kernel <<<gird_dim, block_dim, 0, stream_id >>>(g_input,
    //                                                       ((T *)(size_t(g_input) + width_output)),
    //                                                       g_output,
    //                                                       pitch_input,
    //                                                       width_index,  
    //                                                       batch_size);
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



