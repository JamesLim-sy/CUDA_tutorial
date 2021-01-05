/*************************************
* Device properties info. query
*************************************/
#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include "../inc/cuda_test.h"

int main(int argc, char *argv[])
{
    int offset = 0;
    int  ret = (cudaError_t)0;
    size_t N = 1<<24;
    size_t loop = 500;

    if (argc > 1) { 
        offset = atoi(argv[1]);
    }
    N = 1<<offset;
    
    // case1 : Query device info.
    ret = device_test();

    // case4 : Perf_test for single_validation 
    ret = perf_test_for_single_validation<DATA_TYPE>(N, loop);

    // case3 : Perf_test with combination block thread
    ret = perf_test_with_combination_block_thread<DATA_TYPE>(N, loop);

    // case4 : Perf_test with memory address offset.
    ret = perf_test_with_mem_addr<DATA_TYPE>(N, loop);
    
    // case5 : Perf_test with multi-stream
    ret = perf_test_with_stream<DATA_TYPE>(N, loop);

    return 0;
}


