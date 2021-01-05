#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>
using namespace std;

#define DATA_TYPE  __half


#define CHECK_ERR(condi, err)   do{ \
    if (condi) {   \
        cout << "[LINE, FUNC]: " << __LINE__ << " , " << __func__  << "\t" << err << endl ;\
    } \
}while(0);

// #define PRINT_DUMP(val, fmt)   do{printf("[%s %d]: "#val" = "#fmt"\n", __func__, __LINE__, val);}while(0)

template<typename T>
struct mem_pointer {
    T *p_cpu;
    T *p_gpu;
    T *p_gpu_real;
    double total_time = 0.0;
    double min_time   = 1000.0;
    double max_time   = 0.0;
};

struct block_thread {
    size_t block_num;
    size_t grid_num;
    size_t block_lower;
    size_t block_upper;
    size_t grid_lower ;
    size_t grid_upper ; 
};

/* Case specialization */
int device_test();

template<typename T>
int perf_test_with_stream(size_t data_num, size_t loop);

template<typename T>
int perf_test_with_mem_addr(size_t data_num, size_t loop);

template<typename T>
int perf_test_for_single_validation(size_t data_num, size_t loop);

template<typename T>
int perf_test_with_combination_block_thread(size_t data_num, size_t loop);

/* kernel specialization */
template<typename T>
__global__ void  primitive_add(T *x, T *y, T *z, size_t num);


template<typename T>
__global__ void  grid_stride_add_half(T *x, T *y, T *z, size_t num);


template<typename T>
__global__ void  grid_stride_add_half2(T *x, T *y, T *z, size_t num);


template<typename T>
__global__ void  grid_stride_add_half2_vec2_ld(T *x, T *y, T *z, size_t num);


template<typename T>
__global__ void  grid_stride_add_half2_vec4_ld(T *x, T *y, T *z, size_t num);


template<typename T>
__global__ void  grid_stride_add_half2_vec2_float_ld(T *x, T *y, T *z, size_t num);


template<typename T>
__global__ void  grid_stride_add_half2_vec4_float_ld(T *x, T *y, T *z, size_t num);


template<typename T>
__global__ void  grid_stride_add_vec2_float_version2(T *x, T *y, T *z, size_t num);

template<typename T>
__global__ void  grid_stride_add_vec4_float_version2(T *x, T *y, T *z, size_t num);



/* Accassary specialization */
template<typename T>
int analysis_grid_block(void  *p_x, 
                        void  *p_y, 
                        void  *p_z,
                        void  *p_grid_block,
                        size_t byte_num,
                        int    loop,
                        int    type);

template<typename T>
int  grid_block_combination(void   *x, 
                            void   *y, 
                            void   *z, 
                            void   *p_grid_block, 
                            size_t  byte_num,
                            int     loop, 
                            int     func_type);

template<typename T>
int  rslt_check_func(void        *ptr, 
                     size_t       byte_num,
                     size_t       num, 
                     char        *str,
                     cudaEvent_t  start,
                     cudaEvent_t  stop,
                     size_t       loop);

template<typename T>
void  mem_alloc_offset(void *p_src, int align, size_t byte_num);

template<typename T>
void mem_alloc(void *ptr, size_t byte_num); 

template<typename T>
void mem_free(void  *ptr);
