/*************************************
* Device properties info. query
*************************************/
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>

#define CHECK_ERR(val)   do{ \
    if ( err != cudaSuccess) {   \
        cout << "[LINE, FUNC]: " << (__line__, __func__)   << "\t" <<cudaGetErrorString(val);\
    } \
}while(0);


using namespace std;
int main(int argc, char *argv[])
{
    int nDevice = 0;
    cudaError_t err = cudaSuccess;
    cudaGetDeviceCount(&nDevice);  // Returns the number of CUDA-capable devices attached 

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


