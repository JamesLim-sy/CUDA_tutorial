#include "../inc/cuda_test.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/*************************************
* Device properties info. query
*************************************/
int device_test()
{
    int nDevice = 0;
    cudaGetDeviceCount(&nDevice);  // Returns the number of CUDA-capable devices attached 
    cout << "Device info. acquire: " << endl;
    
    for (int i = 0; i < nDevice; ++i) { 
        cudaDeviceProp  prop;      // Core struct filed for querying the device info. 
        cudaGetDeviceProperties(&prop, i);
        cout << "[                    NO.]: " << i << endl;
        cout << "[                   Name]: " << prop.name << endl;
        cout << "[             Clock Rate]: " << prop.clockRate << endl;
        cout << "[        Mem  Clock Rate]: " << prop.memoryClockRate << endl;
        cout << "[        Bus Width(bits)]: " << prop.memoryBusWidth  << endl;
        cout << "[              Warp Size]: " << prop.warpSize << endl;
        cout << "[Total Global Memory(GB)]: " << prop.totalGlobalMem / (1<<30) << endl;

        /* All devices of the same compute capability have the same limites below 
        *                    |   Max Threads per Thread Block
        * compute capability {   Max Threads per SM
        *                    |   Max Thread  Blocks per SM   */
        cout << "[     Compute Capability]: " << prop.major << "." << prop.minor  << endl;
        cout << "[  Max Threads per   SM ]: " << prop.maxThreadsPerMultiProcessor << endl;
        cout << "[  Max Threads per block]: " << prop.maxThreadsPerBlock         << endl;
        // cout << "[  Max Blocks  per   SM ]: " << prop.maxBlocksPerMultiProcessor << endl;

        cout << "[      Concurrent Kenels]: " << prop.concurrentKernels   << endl;
        cout << "[  Multiprocessor Counts]: " << prop.multiProcessorCount << endl;
        cout << "[ Memory BandWidht(GB/s)]: " << 2 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << endl; // Caculating theorical memory bandwidth of each GPU device. 
        cout << endl;
    }
    return 0;
}

template<typename T>
int perf_test_for_single_validation(size_t data_num, size_t loop)
{
    int     ret = (cudaError_t)0;  // which means success
    size_t  N = data_num;
    int     i = 0;
    size_t  byte_num   = N * sizeof(T);
 
    cout << "Quantity: " << data_num << "\tbyte_num :" << byte_num << endl;
    mem_pointer<T>x;
    mem_pointer<T>y;
    mem_pointer<T>z;
    block_thread grid_block;
    grid_block.block_num  = 128;
    grid_block.grid_num   = (N + 128 - 1) / 128;
    
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
    if (typeid(T) == typeid(__half))  {
        ret = analysis_grid_block<T>(&x, &y, &z, &grid_block, N, loop, 1);
        ret = analysis_grid_block<T>(&x, &y, &z, &grid_block, N, loop, 2);
        ret = analysis_grid_block<T>(&x, &y, &z, &grid_block, N, loop, 3);
        ret = analysis_grid_block<T>(&x, &y, &z, &grid_block, N, loop, 4);
        ret = analysis_grid_block<T>(&x, &y, &z, &grid_block, N, loop, 5);
        ret = analysis_grid_block<T>(&x, &y, &z, &grid_block, N, loop, 6);
        ret = analysis_grid_block<T>(&x, &y, &z, &grid_block, N, loop, 7);
    }
    else {
        ret = analysis_grid_block<T>(&x, &y, &z, &grid_block, N, loop, 10);
    }

    mem_free<T>(&x);
    mem_free<T>(&y);
    mem_free<T>(&z);
    return ret;
}


/***********************************************************************
* To roughly evalute the time cost of each kernel in each block-thread
* combination conditions
************************************************************************/
template<typename T>
int perf_test_with_combination_block_thread(size_t data_num, size_t loop)
{
    int     ret = (cudaError_t)0;  // which means success
    size_t  N = data_num;
    int     i = 0;
    size_t  byte_num   = N * sizeof(T);
    
    mem_pointer<T>x;
    mem_pointer<T>y;
    mem_pointer<T>z;
    block_thread grid_block;

    grid_block.block_lower = 32;
    grid_block.block_upper = 2048;    
    grid_block.grid_lower = 512;
    grid_block.grid_upper = 1024<<7;
    
    mem_alloc<T>(&x, byte_num);
    mem_alloc<T>(&y, byte_num);
    mem_alloc<T>(&z, byte_num);
    
    cout << "Quantity: " << data_num << "\tbyte_num :" << byte_num << endl;
    
    for (i = 0; i < N; ++i){
        x.p_cpu[i] = (T)10;
        y.p_cpu[i] = (T)20;
    }
    cudaMemcpy((void *)(x.p_gpu), (void *)(x.p_cpu), byte_num, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)(y.p_gpu), (void *)(y.p_cpu), byte_num, cudaMemcpyHostToDevice);
    
    ret = grid_block_combination<T>(&x, &y, &z, &grid_block, N, loop, 0);
    if (typeid(T) == typeid(__half))  {
        ret = grid_block_combination<T>(&x, &y, &z, &grid_block, N, loop, 1);
        ret = grid_block_combination<T>(&x, &y, &z, &grid_block, N, loop, 2);
        ret = grid_block_combination<T>(&x, &y, &z, &grid_block, N, loop, 3);
        ret = grid_block_combination<T>(&x, &y, &z, &grid_block, N, loop, 4);
        ret = grid_block_combination<T>(&x, &y, &z, &grid_block, N, loop, 5);
        ret = grid_block_combination<T>(&x, &y, &z, &grid_block, N, loop, 6);
        ret = grid_block_combination<T>(&x, &y, &z, &grid_block, N, loop, 7);
    }
    else {
        ret = grid_block_combination<T>(&x, &y, &z, &grid_block, N, loop, 10);
    }
    cout << "Perf_test_with_combination_block_thread done!" << endl;
    mem_free<T>(&x);
    mem_free<T>(&y);
    mem_free<T>(&z);
    return ret;
}



/************************************************************************
* To evalute the effect of memory-addr offset on performance.
*************************************************************************/
template<typename T>
int perf_test_with_mem_addr(size_t data_num, size_t loop)
{
    int       ret      = (cudaError_t)0;  // which means success
    int       i        = 0;
    int       j        = 0;
    int       offset   = 0;
    size_t    N        = data_num;
    size_t    byte_num = N * sizeof(T);  
    
    const int aligned  = 256;
    const int up_limit = (N + 256 - 1) / 256;
    
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

    for (j = 0; j < N; ++j){
        x.p_cpu[j] = __float2half(10);
        y.p_cpu[j] = __float2half(20);
    }

    for (i = 0; i < up_limit; ++i) 
    {
        offset = i * sizeof(T);
        z.p_gpu = (T *)((size_t)(z.p_gpu_real) + offset);
        z.total_time = 0.0; 
        z.max_time   = 0.0; 
        z.min_time   = 1000.0;
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
    cout << "perf_test_with_mem_addr done!" << endl;
    mem_free<T>(&x);
    mem_free<T>(&y);
    mem_free<T>(&z);
    return 0;
}



/************************************************************************
* To roughly evalute the performance of default-stream with mulit-streams
*************************************************************************/    
template<typename T>
int perf_test_with_stream(size_t data_num, size_t loop)
{
    int       i          = 0;
    int       j          = 0;
    size_t    N          = data_num; 
    int       ret        = (cudaError_t)0;  // which means success
    size_t    byte_num   = N * sizeof(T);  
    const int stream_num = 4;
    int       offset     = 0 ;
    int       grid_size  = 0; 
    
    cudaEvent_t  start, stop;
    cudaStream_t stream_id[stream_num];
    size_t stream_size = N / stream_num;
    size_t stream_byte = stream_size * sizeof(T);
    
    mem_pointer<T>x;
    mem_pointer<T>y;
    mem_pointer<T>z;
    block_thread grid_block;

    grid_block.block_num = 256;
    grid_block.grid_num  = 65536;
    dim3 blocksize(grid_block.block_num);
    dim3 gridsize(grid_block.grid_num);

    mem_alloc<T>(&x, byte_num);
    mem_alloc<T>(&y, byte_num);
    mem_alloc<T>(&z, byte_num);

    for (i = 0; i < N; ++i){
        x.p_cpu[i] = (T)10;
        y.p_cpu[i] = (T)20;
    }
    for (i = 0; i < stream_num; ++i) {
        cudaStreamCreate(&stream_id[i]);
    }
    primitive_add <<< grid_block.grid_num, grid_block.block_num >>> (x.p_gpu, y.p_gpu, z.p_gpu, N>>4);  // warm up

    cudaEventCreate(&start);
    cudaEventCreate( &stop);
    while (j < loop){
        cudaMemset((void *)(z.p_gpu), 0, byte_num);

        cudaEventRecord(start);
        cudaMemcpy((void *)(x.p_gpu), (void *)(x.p_cpu), byte_num, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)(y.p_gpu), (void *)(y.p_cpu), byte_num, cudaMemcpyHostToDevice); 
        grid_stride_add_half <<< grid_block.grid_num, grid_block.block_num >>> (x.p_gpu, y.p_gpu, z.p_gpu, N);
        cudaMemcpy((void *)(z.p_cpu), (void *)(z.p_gpu), byte_num, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
 
        ret = rslt_check_func<T>(&z, byte_num, N, "default-stream", start, stop, loop);
        CHECK_ERR(ret != 0, ret);
        j++;
    }

    grid_size  = grid_block.grid_num / stream_num; 
    while (j < loop){
        cudaMemset((void *)(z.p_gpu), 0, byte_num);

        cudaEventRecord(start);
        for (i = 0; i < stream_num; ++i) {
            offset = i * stream_size;
            cudaMemcpyAsync(&(x.p_gpu[offset]), &(x.p_cpu[offset]), stream_byte, cudaMemcpyHostToDevice, stream_id[i]);
            cudaMemcpyAsync(&(y.p_gpu[offset]), &(y.p_cpu[offset]), stream_byte, cudaMemcpyHostToDevice, stream_id[i]);
            grid_stride_add_half <<< grid_size, grid_block.block_num, 0, stream_id[i] >>> (&(x.p_gpu[offset]), 
                                                                                      &(y.p_gpu[offset]),
                                                                                      &(z.p_gpu[offset]), 
                                                                                      stream_size);
            cudaMemcpyAsync(&(z.p_cpu[offset]), &(z.p_gpu[offset]), stream_byte, cudaMemcpyHostToDevice, stream_id[i]);
        }
        cudaEventRecord(stop);
        ret = rslt_check_func<T>(&z, byte_num, N, "4-multi-stream", start, stop, loop);
        CHECK_ERR(ret != 0, ret);  
        j++;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);  

    cout << "perf_test_with_stream done!" << endl;
    mem_free<T>(&x);
    mem_free<T>(&y);
    mem_free<T>(&z);
    return 0;
}


template int perf_test_with_stream<DATA_TYPE>(size_t data_num, size_t loop);
template int perf_test_with_mem_addr<DATA_TYPE>(size_t data_num, size_t loop);
template int perf_test_for_single_validation<DATA_TYPE>(size_t data_num, size_t loop);
template int perf_test_with_combination_block_thread<DATA_TYPE>(size_t data_num, size_t loop);