#include "cuda_test.h"

/*************************************
* Device properties info. query
*************************************/
int device_test()
{
    int nDevice = 0;
    cudaError_t err = cudaSuccess;
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




/***********************************************************************
* To roughly evalute the time cost of each kernel in each block-thread
* combination conditions
************************************************************************/
template<typename T>
int perf_test_with_combination_block_thread(size_t data_num)
{
    int     ret = (cudaError_t)0;  // which means success
    size_t  N = data_num;
    int     i = 0, j = 0;
    int     loop = 1000;
    size_t  byte_num   = N * sizeof(DATA_TYPE);
    
    mem_pointer<DATA_TYPE>x;
    mem_pointer<DATA_TYPE>y;
    mem_pointer<DATA_TYPE>z;
    block_thread grid_block;

    grid_block.block_lower = 32;
    grid_block.block_upper = 2048;    
    grid_block.grid_lower = 512;
    grid_block.grid_upper = 1024<<7;
    
    mem_alloc(&x, byte_num);
    mem_alloc(&y, byte_num);
    mem_alloc(&z, byte_num);
    
    cout << "Perf_test_with_combination_block_thread :" << endl;
    cout << "Quantity: " << data_num << "\tbyte_num :" << byte_num << endl;

    for (i = 0; i < N; ++i){
        x.p_cpu[i] = 10.0;
        y.p_cpu[i] = 20.0;
    }
    cudaMemcpy((void *)(x.p_gpu), (void *)(x.p_cpu), byte_num, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)(y.p_gpu), (void *)(y.p_cpu), byte_num, cudaMemcpyHostToDevice);

    ret = grid_block_combination(&x, &y, &z, &grid_block, byte_num, loop, 0);
    CHECK_ERR(ret != 0, ret);
    ret = grid_block_combination(&x, &y, &z, &grid_block, byte_num, loop, 1);
    CHECK_ERR(ret != 0, ret);
    ret = grid_block_combination(&x, &y, &z, &grid_block, byte_num, loop, 2);
    CHECK_ERR(ret != 0, ret);
    ret = grid_block_combination(&x, &y, &z, &grid_block, byte_num, loop, 3);
    CHECK_ERR(ret != 0, ret);
    ret = grid_block_combination(&x, &y, &z, &grid_block, byte_num, loop, 4);
    CHECK_ERR(ret != 0, ret);

    mem_free(&x);
    mem_free(&y);
    mem_free(&z);
    return  0;
}



// 多模式状态
/************************************************************************
* To evalute the effect of memory-addr offset on performance.
*************************************************************************/    
template<typename T>
int perf_test_with_mem_addr(size_t data_num)
{
    int       i       = 0;
    size_t    N          = data_num; 
    int       ret        = (cudaError_t)0;  // which means success
    size_t    loop       =  1000;
    size_t    byte_num   = N * sizeof(T);  
    
    const int aligned = 128;
    const int up_limit = 33;
    
    
    mem_pointer<DATA_TYPE>x;
    mem_pointer<DATA_TYPE>y;
    mem_pointer<DATA_TYPE>z;
    mem_alloc_offset(&x, aligned, byte_num);
    mem_alloc_offset(&z, aligned, byte_num);
    mem_alloc_offset(&y, aligned, byte_num);
    
    block_thread grid_block;
    grid_block.block_num  = 256;
    grid_block.grid_num   = 65536;
        
    for (i = 0; i < up_limit; i++) 
    {
        offset = i * sizeof(T);
        z.p_gpu = (T *)((size_t)(z.p_gpu) + offset);

        cout << "[x.p_gpu]:" << x.p_gpu << " ";
        cout << "[y.p_gpu]:" << y.p_gpu << " ";
        cout << "[z.p_gpu]:" << z.p_gpu << " ";
        cout << "[offset ]:" << offset  << " "; 
        ret = analysis_grid_block<T>(&x, &y, &z, &grid_block, byte_num, loop, 0);
        CHECK_ERR(ret != 0, -5);
    }
    for (i = 0; i < up_limit; i++) 
    {
        offset = i * sizeof(T);
        x.p_gpu = (T *)((size_t)(x.p_gpu) + offset);

        cout << "[x.p_gpu]:" << x.p_gpu << " ";
        cout << "[y.p_gpu]:" << y.p_gpu << " ";
        cout << "[z.p_gpu]:" << z.p_gpu << " ";
        cout << "[offset ]:" << offset  << " "; 
        ret = analysis_grid_block<T>(&x, &y, &z, &grid_block, byte_num, loop, 0);
        CHECK_ERR(ret != 0, -5);
    }
    mem_free(&x);
    mem_free(&y);
    mem_free(&z);
    return 0;
}




// 多模式状态
/************************************************************************
* To roughly evalute the performance of default-stream with mulit-streams
*************************************************************************/    
template<typename T>
int perf_test_with_stream(size_t data_num)
{
    int       i          = 0;
    int       j          = 0;
    size_t    N          = data_num; 
    int       ret        = (cudaError_t)0;  // which means success
    size_t    loop       =  1000;
    size_t    byte_num   = N * sizeof(T);  
    const int stream_num = 4;
    int       offset     = 0 ;
    int       grid_size  = 0; 
    
    cudaEvent_t  start, stop;
    cudaStream_t stream_id[stream_num];
    size_t stream_size = N / stream_num;
    size_t stream_byte = stream_size * sizeof(T);
    
    mem_pointer<DATA_TYPE>x;
    mem_pointer<DATA_TYPE>y;
    mem_pointer<DATA_TYPE>z;
    block_thread grid_block;

    grid_block.block_num = 256;
    grid_block.grid_num  = 65536;
    dim3 blocksize(grid_block.block_num);
    dim3 gridsize(grid_block.grid_num);

    mem_alloc(&x, byte_num);
    mem_alloc(&y, byte_num);
    mem_alloc(&z, byte_num);

    for (i = 0; i < N; ++i){
        x.p_cpu[i] = 10.0f;
        y.p_cpu[i] = 20.0f;
    }
    for (i = 0; i < stream_num; ++i) {
        cudaStreamCreate(&stream_id[i]);
    }
    primitive_add <<< grid_block.grid_num, grid_block.block_num >>> (x->p_gpu, y->p_gpu, z->p_gpu, N>>4);  // warm up

    cudaEventCreate(&start);
    cudaEventCreate( &stop);
    while (j < loop){
        cudaMemset((void *)(z.p_gpu), 0, byte_num);

        cudaEventRecord(start);
        cudaMemcpy((void *)(x.p_gpu), (void *)(x.p_cpu), byte_num, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)(y.p_gpu), (void *)(y.p_cpu), byte_num, cudaMemcpyHostToDevice); 
        grid_stride_add <<< grid_num, block_num >>> (x.p_gpu, y.p_gpu, z.p_gpu, N);
        cudaMemcpy((void *)(z.p_cpu), (void *)(z.p_gpu), byte_num, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
 
        ret = rslt_check_func<T>(z, byte_num, N, "default-stream", start, stop, loop);
        CHECK_ERR(ret != 0, ret);
        j++;
    }

    grid_size  = grid_block.grid_num / stream_num; 
    while (j < loop){
        cudaMemset((void *)(z.p_gpu), 0, byte_num);

        cudaEventRecord(start);
        for (i = 0; i < stream_num; ++i) {
            osffset = i * stream_size;
            cudaMemcpyAsync(&(x.p_gpu[offset]), &(x.p_cpu[offset]), stream_byte, cudaMemcpyHostToDevice, stream_id[i]);
            cudaMemcpyAsync(&(y.p_gpu[offset]), &(y.p_cpu[offset]), stream_byte, cudaMemcpyHostToDevice, stream_id[i]);
            grid_stride_add <<< grid_size, grid_block.block_num, 0, stream_id[i] >>> (&(x.p_gpu[offset]), 
                                                                                      &(y.p_gpu[offset]),
                                                                                      &(z.p_gpu[offset]), 
                                                                                      stream_size);
            cudaMemcpyAsync(&(z.p_cpu[offset]), &(z.p_gpu[offset]), stream_byte, cudaMemcpyHostToDevice, stream_id[i]);
        }
        cudaEventRecord(stop);
        ret = rslt_check_func<T>(z, byte_num, N, "4-multi-stream", start, stop, loop);
        CHECK_ERR(ret != 0, ret);  
        j++;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);  

    mem_free(&x);
    mem_free(&y);
    mem_free(&z);
    return 0;
}