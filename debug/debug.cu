#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;
#define CHECK_ERR(condi, val)  do{if(condi){return ret;}}while(0);

template<typename T>
int  rslt_check_func(void   *ptr, 
                     size_t  byte_num,
                     size_t  num, 
                     char   *str,
                     cudaEvent_t start,
                     cudaEvent_t stop,
                     size_t  loop);

template<typename T>
struct mem_pointer {
    T *p_cpu;
    T *p_gpu;
    T *p_gpu_real;
    double total_time = 0.0;
    double min_time   = 1000.0;
    double max_time   = 0.0;
};

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

struct block_thread {
    size_t block_num;
    size_t grid_num;
    size_t block_lower;
    size_t block_upper;
    size_t grid_lower ;
    size_t grid_upper ; 
};


/*************************************
* CUDA memory offset acquire.
*************************************/
template<typename T>
void  mem_alloc_offset(void *ptr, int  align, size_t byte_num);

template<typename T>
void mem_alloc(void *ptr, size_t byte_num);

template<typename T>
void mem_free(void *ptr);


template<typename T>
__global__ void  primitive_add(T *x, T *y, T *z, size_t num)
{
    int    i      = threadIdx.x + blockIdx.x * blockDim.x;   
    size_t stride = blockDim.x * gridDim.x ; 
    if (i < num) {
       z[i] = (x[i] + y[i]);
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
    // struct half2_float *dst_z = (half2_float *)z;
    // struct half2_float *src_x = (half2_float *)x;
    // struct half2_float *src_y = (half2_float *)y;
    float2 *dst_z = (float2 *)x;
    float2 *src_x = (float2 *)y;
    float2 *src_y = (float2 *)z;
    
    for (i = idx; i < loop; i += stride){
        p_x1 = src_x[i].x;
        p_y1 = src_y[i].x;
        p_x2 = src_x[i].y;
        p_y2 = src_y[i].y;
        
        dst_z[i].x = __hadd2(p_x1, p_y1);
        dst_z[i].y = __hadd2(p_x2, p_y2);
        // dst_z[i] = src_x[i];
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
int analysis_grid_block(void  *p_x, 
                        void  *p_y, 
                        void  *p_z,
                        void  *p_grid_block,
                        size_t byte_num,
                        int    loop,
                        int    type)
{
    int   ret = (cudaError_t)0;  // which means success
    int     i = 0;
    int     j = 0;
    size_t  N = byte_num / sizeof(T);
    float   alpha = 1.0;
    
    cublasHandle_t hdl;
    cudaEvent_t start, stop;
    struct mem_pointer<T> *x  = (mem_pointer<T> *)p_x;
    struct mem_pointer<T> *y  = (mem_pointer<T> *)p_y;
    struct mem_pointer<T> *z  = (mem_pointer<T> *)p_z;
    struct block_thread *p_gb = (block_thread *)p_grid_block;
    
    dim3 blocksize(p_gb->block_num);
    dim3 gridsize( p_gb->grid_num);
    primitive_add <<< p_gb->grid_num, p_gb->block_num >>> (x->p_gpu, y->p_gpu, z->p_gpu, N>>4);  // warm up
   
    // double tol1 = 0, tol2 = 0;
    // for (int j = 0; j < N; j++){
    //     tol1 += __half2float(x->p_cpu[j]);
    //     tol2 += __half2float(y->p_cpu[j]); 
    // }
    // printf("[val] : %f  %f\n", tol1 / N, tol2 / N);
    // cout <<  p_gb->grid_num << "\t" << p_gb->block_num << endl; 

    while (j < loop) {
        cudaEventCreate(&start);
        cudaEventCreate( &stop);
        cudaMemset((void *)(z->p_gpu), 0, byte_num);

        switch (type) {
            case 0 : {
                cudaEventRecord(start);
                grid_stride_add_half <<< p_gb->grid_num, p_gb->block_num >>> (x->p_gpu, y->p_gpu, z->p_gpu, N);
                cudaEventRecord(stop);
                break;
            }
            case 1 : {
                cudaEventRecord(start);
                grid_stride_add_half2 <<< (p_gb->grid_num >> 1), p_gb->block_num >>> (x->p_gpu, y->p_gpu, z->p_gpu, N);
                cudaEventRecord(stop);
                break;
            }
            case 2 : {
                cudaEventRecord(start);
                grid_stride_add_half2_vec2_ld <<< (p_gb->grid_num >> 2), p_gb->block_num >>> (x->p_gpu, y->p_gpu, z->p_gpu, N);
                cudaEventRecord(stop);
                break;
            }
            case 3 : {
                cudaEventRecord(start);
                grid_stride_add_half2_vec4_ld <<< (p_gb->grid_num >> 3), p_gb->block_num >>> (x->p_gpu, y->p_gpu, z->p_gpu, N);
                cudaEventRecord(stop);
                break;
            }
            case 4 : {
                cudaEventRecord(start);
                grid_stride_add_half2_vec2_float_ld <<< (p_gb->grid_num >> 2), p_gb->block_num >>> (x->p_gpu, y->p_gpu, z->p_gpu, N);
                cudaEventRecord(stop);
                break;
            }
            case 5 : {
                cudaEventRecord(start);
                grid_stride_add_half2_vec4_float_ld <<< (p_gb->grid_num >> 3), p_gb->block_num >>> (x->p_gpu, y->p_gpu, z->p_gpu, N);
                cudaEventRecord(stop);
                break;
            }
            default : {
                return -10;
                break;
            }
        }
        ret = rslt_check_func<T>(z, byte_num, N, "test", start, stop, loop);
        CHECK_ERR(ret != 0, ret);

        cudaEventDestroy(start);
        cudaEventDestroy(stop); 
        j++;
    }
    return 0;
}



/***********************************************************************
* To roughly evalute the time cost of each kernel in each block-thread
* combination conditions
************************************************************************/
template<typename T>
int perf_test_with_combination_block_thread(size_t data_num, int loop)
{
    int     ret = (cudaError_t)0;  // which means success
    size_t  N = data_num;
    int     i = 0, j = 0;
    size_t  byte_num = N * sizeof(T);
    
    cout << "[data_num]: " << N << "\t[byte_num]: " << byte_num << endl;

    mem_pointer<T>x;
    mem_pointer<T>y;
    mem_pointer<T>z;
    block_thread grid_block;

    grid_block.block_num = 256;
    grid_block.grid_num  = (N + 256 - 1) / 256;
    
    mem_alloc<T>(&x, byte_num);
    mem_alloc<T>(&y, byte_num);
    mem_alloc<T>(&z, byte_num);
    
    for (i = 0; i < N; ++i){
        x.p_cpu[i] = (T)10;
        y.p_cpu[i] = (T)20;
    }
    cudaMemcpy((void *)(x.p_gpu), (void *)(x.p_cpu), byte_num, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)(y.p_gpu), (void *)(y.p_cpu), byte_num, cudaMemcpyHostToDevice);

    ret = analysis_grid_block<T>(&x, &y, &z, &grid_block, N, loop, 0);
    ret = analysis_grid_block<T>(&x, &y, &z, &grid_block, N, loop, 1);
    ret = analysis_grid_block<T>(&x, &y, &z, &grid_block, N, loop, 2);
    ret = analysis_grid_block<T>(&x, &y, &z, &grid_block, N, loop, 3);
    ret = analysis_grid_block<T>(&x, &y, &z, &grid_block, N, loop, 4);
    ret = analysis_grid_block<T>(&x, &y, &z, &grid_block, N, loop, 5);
    mem_free<T>(&x);
    mem_free<T>(&y);
    mem_free<T>(&z);
    return  0;
}



// 多模式状态
/************************************************************************
* To evalute the effect of memory-addr offset on performance.
*************************************************************************/    
template<typename T>
int perf_test_with_mem_addr(size_t data_num, int loop)
{
    int       ret      = (cudaError_t)0;  // which means success
    int       i        = 0;
    int       j        = 0;
    int       offset   = 0;
    size_t    N        = data_num;
    size_t    byte_num = N * sizeof(T);  
    
    const int aligned = 128 * sizeof(T);
    const int up_limit = 128 / sizeof(T) + 1;
    
    struct mem_pointer<T>x;
    struct mem_pointer<T>y;
    struct mem_pointer<T>z;
    mem_alloc_offset<T>(&x, aligned, byte_num);
    mem_alloc_offset<T>(&z, aligned, byte_num);
    mem_alloc_offset<T>(&y, aligned, byte_num);
    
    block_thread grid_block;
    grid_block.block_num  = 256;
    grid_block.grid_num   = (N + 256 - 1) / 256;
    cout << "[data_num]: " << N << "\t[byte_num]: " << byte_num << endl;

    for (i = 0; i < up_limit; ++i) 
    {
        offset = i * sizeof(T);
        z.p_gpu = (T *)((size_t)(z.p_gpu_real) + offset);
        z.total_time = 0.0; 
        z.max_time   = 0.0; 
        z.min_time   = 1000.0;

        for (j = 0; j < N; ++j){
            x.p_cpu[j] = (T)10;
            y.p_cpu[j] = (T)20;
        }
        cudaMemcpy((void *)(x.p_gpu), (void *)(x.p_cpu), byte_num, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)(y.p_gpu), (void *)(y.p_cpu), byte_num, cudaMemcpyHostToDevice);

        cout << "[x.p_gpu]:" << x.p_gpu << " ";
        cout << "[y.p_gpu]:" << y.p_gpu << " ";
        cout << "[z.p_gpu]:" << z.p_gpu << " ";
        cout << "[offset ]:" << offset  << "\t"; 
        ret = analysis_grid_block<T>(&x, &y, &z, &grid_block, byte_num, loop, 0);
        cout << "\t[AVG]:  " << z.total_time / loop ;
        cout << "\t[MAX]: "  << z.max_time << "\t[MIN]: " << z.min_time << endl;
        CHECK_ERR(ret != 0, -5);
    }
    
    for (i = 0; i < up_limit; i++) 
    {
        offset = i * sizeof(T);
        x.p_gpu = (T *)((size_t)(x.p_gpu_real) + offset);
        z.total_time = 0.0; 
        z.max_time   = 0.0; 
        z.min_time   = 1000.0;
 
        cudaMemcpy((void *)(x.p_gpu), (void *)(x.p_cpu), byte_num, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)(y.p_gpu), (void *)(y.p_cpu), byte_num, cudaMemcpyHostToDevice);

        cout << "[x.p_gpu]:" << x.p_gpu << "   ";
        cout << "[y.p_gpu]:" << y.p_gpu << "   ";
        cout << "[z.p_gpu]:" << z.p_gpu << "   ";
        cout << "[offset ]:" << offset  << "\t"; 
        ret = analysis_grid_block<T>(&x, &y, &z, &grid_block, byte_num, loop, 0);
        cout << "\t[AVG]:  " << z.total_time / loop ;
        cout << "\t[MAX]: "  << z.max_time << "\t[MIN]: " << z.min_time << endl;

        CHECK_ERR(ret != 0, -5);
    }
    mem_free<T>(&x);
    mem_free<T>(&y);
    mem_free<T>(&z);
    return 0;
}



#define DATA_TYPE  char
int main(int argc, char *argv[])
{
    int offset = 0;
    int  ret = (cudaError_t)0;
    size_t N = 1<<24;

    if (argc > 1) { 
        offset = atoi(argv[1]);
    }
    N = 1<<offset;

    // case1 : Perf_test with combination block thread
    ret = perf_test_with_combination_block_thread<DATA_TYPE>(N, 5);

    // case2 : Perf_test with memory offset
    // ret = perf_test_with_mem_addr<DATA_TYPE>(N, 100);

    return 0;
}




template<typename T>
int  rslt_check_func(void   *ptr, 
                     size_t  byte_num,
                     size_t  num, 
                     char   *str,
                     cudaEvent_t start,
                     cudaEvent_t stop,
                     size_t  loop)
{
    // To sync the mission accomplishement of GPU
    float   timespan = 0.0;
    int     i   = 0 ;
    int     ret = 0; 
    float   eps = 1e-5;
    float   max_err  = 0.0; 
    float   target   = 30.0;
    struct mem_pointer<T> *p_dst = (mem_pointer<T> *)ptr;
    double  val_sum = 0;

    cudaEventSynchronize(stop);  
    cudaEventElapsedTime(&timespan, start, stop);
    cudaMemcpy((void *)(p_dst->p_cpu), (void *)(p_dst->p_gpu), byte_num, cudaMemcpyDeviceToHost);

    for (i = 0; i < num; ++i){
        max_err =  __half2float(p_dst->p_cpu[i]) -  __half2float(target);
        if (fabs(max_err) > eps ) {
            cout << "[ERR] value: " << __half2float(p_dst->p_cpu[i]);
            cout << "\t[index] : " << i  << "\t[Loop]: " << loop << endl;
            return -10; 
        }
    }
    p_dst->total_time += timespan;
    p_dst->min_time    = fmin(p_dst->min_time, timespan);
    p_dst->max_time    = fmax(p_dst->max_time, timespan);
    ret = (max_err > eps) ? -5 : 0;
    return ret;
}


/*************************************
* CUDA memory offset acquire.
*************************************/
template<typename T>
void  mem_alloc_offset(void    *ptr, 
                       int      align,
                       size_t   byte_num)
{
    struct mem_pointer<T> *p_src = (mem_pointer<T> *)ptr;
    size_t gpu_length = byte_num + align;

    p_src->p_cpu = (T *)malloc(byte_num);
    cudaMalloc((void **)&(p_src->p_gpu_real), gpu_length); 
    p_src->p_gpu = p_src->p_gpu_real;
}


template<typename T>
void mem_alloc(void *ptr, size_t byte_num) 
{
    struct mem_pointer<T> *p_dst = (mem_pointer<T> *)ptr;
    p_dst->p_cpu = (T *)malloc(byte_num);
    cudaMalloc((void **)&(p_dst->p_gpu), byte_num); 
    p_dst->p_gpu_real = nullptr;
}


template<typename T>
void mem_free(void *ptr) 
{
    struct mem_pointer<T> *p  = (mem_pointer<T> *)ptr;
    free(p->p_cpu);

    if (p->p_gpu_real != nullptr) {
        cudaFree(p->p_gpu_real);
    }
    else {
        cudaFree(p->p_gpu);
    }
}