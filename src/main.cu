/*************************************
* Device properties info. query
*************************************/
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cuda_test.h"


#define DATA_TYPE  __half
int main(int argc, char *argv[])
{
    size_t N = 1<<24;

    // case1 : Query device info.
    ret = device_test();

    // case2 : Perf_test with combination block thread
    ret = perf_test_with_combination_block_thread<DATA_TYPE>(N);

    // case3 : Perf_test with memory address offset.
    ret = perf_test_with_mem_addr<DATA_TYPE>(N);
    
    // case4 : 
    ret = perf_test_with_stream<DATA_TYPE>(N);

    return 0;
}


