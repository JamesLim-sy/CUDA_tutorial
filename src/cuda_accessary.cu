#include "cuda_test.h"

/*************************************
* CUDA result check.
*************************************/
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
    cudaEventSynchronize(stop);  
    cudaEventElapsedTime(&timespan, start, stop);
    int     i   = 0 ;
    int     ret = 0; 
    float   eps = 1e-5;
    float   max_err  = 0.0; 
    float   target   = 30.0;
    struct mem_pointer<T> *p_dst = (mem_pointer<T> *)ptr;
    double  val_sum = 0;

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



template<typename T>
int  grid_block_combination(void   *x, 
                            void   *y, 
                            void   *z, 
                            void   *p_grid_block, 
                            size_t  byte_num,
                            int     loop, 
                            int     func_type) 
{
    int  i = 0, j = 0;
    int  ret  = (cudaError_t)0;
    block_thread *p_gb = (block_thread *)p_grid_block;
    struct mem_pointer<T> *p_x = (mem_pointer<T> *)x;
    struct mem_pointer<T> *p_y = (mem_pointer<T> *)y;
    struct mem_pointer<T> *p_z = (mem_pointer<T> *)z;

    cout << "half :" << endl; 
    for (i = p_gb->block_lower; i < p_gb->block_upper; i <<= 1)
    {  
        for (j = p_gb->grid_lower; j < p_gb->grid_upper ; j <<= 1) 
        {
            p_z->total_time = 0.0; 
            p_z->max_time   = 0.0; 
            p_z->min_time   = 1000.0;
            cout << "[BLOCK NUM] : " << i << "  [GRID NUM] : " << j; 
            ret = analysis_grid_block<DATA_TYPE>(&x, &y, &z, j, i, byte_num, loop, func_type);
            CHECK_ERR(ret != 0, -5);
            cout << "\t[AVG]:  " << p_z->total_time / loop ;
            cout << "\t[MAX]: " << p_z->max_time << "\t[MIN]: " << p_z->min_time << endl;
        }
        cout << endl;
    }
    return ret;
}


/*
    cout << "half2_float4 :" << endl;
    for (i = block_num_lo; i < block_num_up; i <<= 1){  
        for (j = grid_num_lo; j < grid_num_up ; j <<= 1) { 
            cout << "[BLOCK NUM] : " << i << " [GRID NUM] : " << j;
            z.total_time = 0.0;  z.max_time   = 0.0; z.min_time   = 1000.0;
            ret = analysis_grid_block<DATA_TYPE>(&x, &y, &z, j, i, byte_num, loop, 3);
            CHECK_ERR(ret != 0, -5);
            cout << "\t[AVG]:  " << z.total_time / loop  << "\t[MAX]: " << z.max_time << "\t[MIN]: " << z.min_time << endl;
        }
        cout << endl;
    }
*/


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
    dim3 gridsize(p_gb->grid_num);
    primitive_add <<< p_gb->grid_num, p_gb->block_num >>> (x->p_gpu, y->p_gpu, z->p_gpu, N>>4);  // warm up
   
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
                grid_stride_add_half2 <<< p_gb->grid_num, p_gb->block_num >>> (x->p_gpu, y->p_gpu, z->p_gpu, N);
                cudaEventRecord(stop);
                break;
            }
            case 2 : {
                cudaEventRecord(start);
                grid_stride_add_half2_vec2_ld <<< p_gb->grid_num, p_gb->block_num >>> (x->p_gpu, y->p_gpu, z->p_gpu, N);
                cudaEventRecord(stop);
                break;
            }
            case 3 : {
                cudaEventRecord(start);
                grid_stride_add_half2_vec4_ld <<< p_gb->grid_num, p_gb->block_num >>> (x->p_gpu, y->p_gpu, z->p_gpu, N);
                cudaEventRecord(stop);
                break;
            }
            case 4 : {
                cublasCreate(&hdl); 
                cudaEventRecord(start);
                ret  = cublasSaxpy(hdl, N, &alpha, (float *)x->p_gpu, 1, (float *)y->p_gpu, 1);
                cudaEventRecord(stop);
                CHECK_ERR(ret != 0, ret);
                
                cudaMemcpy((void *)(z->p_gpu), (void *)(y->p_gpu), byte_num, cudaMemcpyDeviceToDevice);
                cudaMemcpy((void *)(y->p_gpu), (void *)(y->p_cpu), byte_num, cudaMemcpyHostToDevice); 
                cublasDestroy(hdl);
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


/*************************************
* CUDA memory offset acquire.
*************************************/
template<typename T>
void  mem_alloc_offset(void    *p_src, 
                       int      align,
                       size_t   byte_num)
    struct mem_pointer<T> *p_src = (mem_pointer<T> *)ptr;
    size_t gpu_length = byte_num + align;

    p_src->p_cpu = (DATA_TYPE *)malloc(byte_num);
    cudaMalloc((void **)&(p_src->p_gpu_real), gpu_length); 
    p_src->p_gpu = p_src->p_gpu_real;
}


template<typename T>
void mem_alloc(void *ptr, size_t byte_num) 
{
    struct mem_pointer<T> *p_dst = (mem_pointer<T> *)ptr;
    p_dst->p_cpu = (DATA_TYPE *)malloc(byte_num);
    cudaMalloc((void **)&(p_dst->p_gpu), byte_num); 
    p_dst->p_gpu_real = nullptr;
}


template<typename T>
void mem_free(void *ptr) 
{
    struct mem_pointer<T> *p  = (mem_pointer<T> *)ptr;
    free(p->p_cpu);

    if (p->p_gpu_real) {
        cudaFree(p->p_gpu_real);
    }
    else {
        cudaFree(p->p_gpu);
    }
}

