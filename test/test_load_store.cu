#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>


using namespace std;

template<typename scalar_t>
struct mem_pointer {
    scalar_t *p_cpu;
    scalar_t *p_gpu;
};

template<typename scalar_t>
void mem_alloc(void *ptr, size_t byte_num) 
{
    struct mem_pointer<scalar_t> *p_dst = (mem_pointer<scalar_t> *)ptr;
    p_dst->p_cpu = (scalar_t *)malloc(byte_num);
    cudaMalloc((void **)&(p_dst->p_gpu), byte_num); 
}

template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};


template<typename scalar_t, int vec_size>
__global__ void  grid_stride_test_vec4_load(scalar_t * x, scalar_t * y, 
                                            scalar_t * z, size_t num)
{
    int    idx    = threadIdx.x + blockIdx.x * blockDim.x;   
    size_t stride = blockDim.x  * gridDim.x; 
    using vec_t   = aligned_vector<scalar_t, vec_size>;
    int  loop = num / vec_size;

    vec_t *src_x = reinterpret_cast<vec_t *>(x);
    vec_t *src_y = reinterpret_cast<vec_t *>(y);
    vec_t *dst_z = reinterpret_cast<vec_t *>(z);
    vec_t vec_x, vec_y, vec_z;
    
    for (int tid = idx; tid < loop; tid += stride){
        vec_x = src_x[tid];
        vec_y = src_y[tid];
        scalar_t *scalar_x = reinterpret_cast<scalar_t *>(&vec_x);
        scalar_t *scalar_y = reinterpret_cast<scalar_t *>(&vec_y);
        scalar_t *scalar_z = reinterpret_cast<scalar_t *>(&vec_z);
    
        #pragma unroll
        for (int i = 0; i < vec_size; i++) {
            scalar_z[i] = scalar_x[i] + scalar_y[i];
            // float tmp = __half2float(scalar_z[i]);
            // printf("z=%f\n", tmp);
        }
        dst_z[tid] = vec_z;
    }
}


template<typename scalar_t, int vec_size>
__global__ void  grid_stride_test_hadd2(scalar_t * x, scalar_t * y, 
                                        scalar_t * z, size_t num)
{
    int    tid    = threadIdx.x + blockIdx.x * blockDim.x;   
    size_t stride = blockDim.x  * gridDim.x;
    size_t loop = num >> 1;

    for (; tid < loop; tid += stride){
        half2 *scalar_x = reinterpret_cast<half2 *>(x);
        half2 *scalar_y = reinterpret_cast<half2 *>(y);
        half2 *scalar_z = reinterpret_cast<half2 *>(z);
        scalar_z[tid] = (scalar_x[tid] + scalar_y[tid]); // instead of __hadd2()
    
        // float2 tmp = __half22float2(scalar_z[tid]);
        // printf("%f\n", tmp.x);
    }
}


#define T float
int main(int argc, char* argv[]) {
    int     ret = (cudaError_t)0;  // which means success
    size_t  N = 1<<22 + 1;
    int     i = 0, j = 0;
    size_t  byte_num = N * sizeof(T);
    int threads = 256;
    int blocks  = (N + 256 - 1) / 256;
    
    int test_case = 0;
    if (argc > 1) {
        test_case = atoi(argv[1]);
    }

    mem_pointer<T>x;
    mem_pointer<T>y;
    mem_pointer<T>z;    
    mem_alloc<T>(&x, byte_num);
    mem_alloc<T>(&y, byte_num);
    mem_alloc<T>(&z, byte_num);
    
    for (i = 0; i < N; ++i){
        x.p_cpu[i] = (T)10.f;
        y.p_cpu[i] = (T)20.f;
    }
    cudaMemcpy((void *)(x.p_gpu), (void *)(x.p_cpu), byte_num, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)(y.p_gpu), (void *)(y.p_cpu), byte_num, cudaMemcpyHostToDevice);
    
    for (int i = 0; i < 5000; i++) {
        switch (test_case) {
            case 0 : {
                grid_stride_test_hadd2<T, 4><<<blocks>>3, threads>>>(x.p_gpu, y.p_gpu, z.p_gpu, N);
                break;
            }
            case 1 : {
                grid_stride_test_vec4_load<T, 4><<<blocks>>3, threads>>>(x.p_gpu, y.p_gpu, z.p_gpu, N);
                break;
            }
        }
    }
    cudaMemcpy((void *)(z.p_cpu), (void *)(z.p_gpu), byte_num, cudaMemcpyDeviceToHost);
}