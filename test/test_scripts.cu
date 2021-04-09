#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <string>
#include <functional>

#define INT_BITS  32
#define DEBUG_LOG  do {    \
    std::cout << out_size << std::endl;\
    std::cout << "a[0]:\t"; for (int x=0; x < out_size; ++x) {std::cout << a[0][x] << "\t"; } std::cout << std::endl;\
    std::cout << "a[1]:\t"; for (int x=0; x < out_size; ++x) {std::cout << a[1][x] << "\t"; } std::cout << std::endl;\
    std::cout << "out :\t"; for (int x=0; x < out_size; ++x) {std::cout << (*out)[x] << "\t"; } std::cout << std::endl;\
}while(0);



template <typename index_t>
struct DivMod {
  index_t div, mod;

  __forceinline__ __device__  DivMod(index_t div, index_t mod) : div(div), mod(mod) { }
};


template <typename index_t>
struct FastDivMod  {
  FastDivMod() { }
  
  explicit FastDivMod(index_t d) : divisor(d) {
    for (shift_val = 0; shift_val < INT_BITS; ++shift_val) {
        if ((1 << shift_val) >= divisor) {
            break;
        }
    }
    uint64_t one_uint64 = 1;
    uint64_t temp_div = ((one_uint64 << INT_BITS) * ((one_uint64 << shift_val) - divisor)) / divisor + 1;
    multiplier = temp_div;
  }

  __forceinline__ __device__ index_t div(index_t n) const {   
    index_t t = __umulhi(n, multiplier);
    return (t + n) >> shift_val;
  }
 
  __forceinline__ __device__ DivMod<index_t> divmod(index_t n) const {
    index_t q = div(n);
    return DivMod<index_t>(q, n - q * divisor);
  }

  index_t divisor;  
  index_t multiplier; 
  index_t shift_val;
};


template <typename T, typename vec_t>
struct func1 {
    T operator()(vec_t arr) {
        return std::accumulate(arr.begin(), arr.end(), 1, std::multiplies<T>());
    }
};

template <typename T, typename vec_t>
struct func2 {
    T operator()(vec_t arr) {
        return std::accumulate(arr.begin(), arr.end(), 0, std::plus<T>());
    }
};


template <typename T, typename vec_t> 
struct TensorLoader {
    TensorLoader() {}

    TensorLoader(vec_t &in_tensor, vec_t &out_tensor) {
        vec_t temp_vec(in_tensor);
        std::transform(temp_vec.begin(), temp_vec.end(),
                        out_tensor.begin(), temp_vec.begin(), std::minus<T>()); 
        bool is_broadcast = std::accumulate(temp_vec.begin(), temp_vec.end(),
                            0, std::plus<T>()) == 0 ? false : true;

        if (is_broadcast) {
            Funcp = func1<T, vec_t>(); 
        } else {
            Funcp = func2<T, vec_t>();
        }
    }
    std::function<T(vec_t)> Funcp;
};


// template <typename T, typename vec_t, int N>
// void TensorsReorganise(std::vector<vec_t> *in_tenosr_data, vec_t * out_tensor_data, 
//                        int &out_size, int &max_idx, int low_idx,  int cnt) {

//     auto VectorReorganise = [] (vec_t *vec, int l_idx, int m_idx) { 
//                              (*vec)[m_idx-1] = std::accumulate(vec->begin() + l_idx, \
//                                         vec->begin() + m_idx,1, std::multiplies<T>());
//                              vec->erase(vec->begin() + l_idx, vec->begin() + m_idx - 1); };

//     if (cnt > 1) {
//         #pragma unroll
//         for (int j=0; j < N; ++j) {
//             VectorReorganise(&((*in_tenosr_data)[j]), low_idx, max_idx);
//         } 
//         VectorReorganise(out_tensor_data, low_idx, max_idx);
//         out_size -= --cnt;
//         max_idx -= cnt;
//     } else if (cnt < 1) {
//         max_idx++;
//     }
// }

template <typename T, typename vec_t, int N>
void DimsReorganise(std::vector<vec_t> *in_tenosr_data, vec_t * out_tensor_data, 
                    int &out_size,
                    void (*merge_func) (bool *, std::vector<vec_t> *, vec_t *, int, int)) {

    auto VectorReorganise = [] (vec_t *vec, int l_idx, int m_idx) { 
                             (*vec)[m_idx-1] = std::accumulate(vec->begin() + l_idx, \
                                        vec->begin() + m_idx,1, std::multiplies<T>());
                             vec->erase(vec->begin() + l_idx, vec->begin() + m_idx - 1); };

    int i = 0;
    while (i < out_size) {
        int cnt = 0;
        int low_idx = i;
        bool equal_flag = true;
        do{
            merge_func(&equal_flag, in_tenosr_data, out_tensor_data, N, i);
            if (equal_flag) {
                i++; cnt++; 
            } else {
                break;
            }
        } while(i < out_size);

        if (cnt > 1) {
            #pragma unroll
            for (int j=0; j < N; ++j) {
                VectorReorganise(&((*in_tenosr_data)[j]), low_idx, i);
            } 
            VectorReorganise(out_tensor_data, low_idx, i);
            out_size -= --cnt;
            i -= cnt;
        } else if (cnt < 1) {
            i++;
        }
    }
}


/*
* Dimension Merging took account of below conditions:
*
* 1. [32, 2, 16, 12] + [32, 1,  1, 12] =>   [32, 2, 16, 12] + [32, 1,  1, 12]  ==>  [32, 32,  12] + [32, 1,  12]
* 2. [32, 2, 16, 12] + [32, 1, 16, 12] =>   [32, 2, 16, 12] + [32, 1, 16, 12]  ==>  [32,  2, 192] + [32, 1, 192]
* 3. [32, 2, 16, 12] + [ 1, 2, 16, 12] =>   [32, 2, 16, 12] + [ 1, 2, 16, 12]  ==>  [32, 384] + [ 1, 384]
* 4. [32, 2, 16, 12] + [32, 1,  1,  1] =>   [32, 2, 16, 12] + [32, 1,  1,  1]  ==>  [32, 192] + [32,   1]
* 5. [32, 2, 16, 12] + [32, 2 ]        =>   [32, 2, 16, 12] + [32, 2,  1,  1]  ==>  [64, 192] + [64, 1]
* 6. [32, 2, 16, 12] + [16, 12]        =>   [32, 2, 16, 12] + [1,  1, 16, 12]  ==>  [32, 2, 192] + [1, 1, 192]
* 7. [32, 2, 16, 12] + [32]            =>   [32, 2, 16, 12] + [32, 1,  1,  1]  ==>  [32, 384] + [32, 1]
* 8. [32, 2, 16, 12] + [2]             =>   [32, 2, 16, 12] + [ 1, 2,  1,  1]  ==>  [32, 2, 192] + [1, 2, 1]
* 9. [32, 2,  1,  1] + [1,  1, 16, 12] =>   [32, 2,  1,  1] + [1,  1, 16, 12]  ==>  [32, 2,   1] + [1, 1, 192]
*10. [32, 1,  1,  1] + [1,  2, 16, 12] =>   [32, 1,  1,  1] + [1,  2, 16, 12]  ==>  [32, 1] + [1, 192]
*11. [32, 1, 16, 12] + [32, 2,  1,  1] =>   [32, 1, 16, 12] + [32, 2,  1,  1]  ==>  [32, 1, 192] + [32, 2, 1]
*12. [32, 1, 16,  1] + [1,  2, 16, 12] =>   [32, 1, 16,  1] + [1,  2, 16, 12]  ==>   No support
*13. [32, 1, 16,  1] + [1,  2,  1, 12] =>   [32, 1, 16,  1] + [1,  2,  1, 12]  ==>   No support


* Process:
* 1. To compensate the lackage of input_tensors dimension;
* 2. To Merge the dimensions of input_tensors while the consequtive equal-dimensions appear;
* 3. To Merge the dimension of input_tensors while the consequtive 1-value-dimensions appear;
* 4. To calculate the strides of each input_tensor. 
*/
template <typename vec_t, typename T, int N>
void MergeDims(std::vector<vec_t> * in_arr, vec_t *out) {
    std::vector<vec_t> a = *in_arr;
    int out_size = out->size();
    
    // 1. 维度补充
    for (int j = 0; j < N; ++j) {
        std::reverse(a[j].begin(), a[j].end());
        if (a[j].size() < out_size ) {
            vec_t vec_temp(out_size, 1);
            int idx_in = 0, idx_out = 0;
            a[j].resize(out_size, 1);

            do {
                if (a[j][idx_in] == (*out)[idx_out] || a[j][idx_in] == 1 ) {
                    vec_temp[idx_out++] = a[j][idx_in++];
                } else {
                    idx_out++;
                }
            }while(idx_out < out_size);            
            std::copy(vec_temp.begin(), vec_temp.end(), a[j].begin());
        }
    }    
    DEBUG_LOG;


    void (*merge_ptr) (bool *, std::vector<vec_t> *, vec_t *, int, int);
    auto merge_same = []  (bool *equal, std::vector<vec_t> *in_arr, vec_t * out, int num, int i) ->void  {
            #pragma unroll 
            for (int j = 1; j < num; ++j) {
                *equal &= (*in_arr)[0][i] == (*in_arr)[j][i];
            }
        };
    auto merge_one = []  (bool *equal, std::vector<vec_t> *in_arr, vec_t * out, int num, int i) ->void {
            *equal &= (*in_arr)[0][i] == 1;
            if (*equal) {
                #pragma unroll
                for (int j = 1; j < num; ++j) {
                    *equal &= (*in_arr)[j][i] == (*out)[i];
                }
            }
        };
    merge_ptr = merge_same;
    DimsReorganise<T, vec_t, N>(&a, out, out_size, merge_ptr);

    // int i = 0;                            
    // while (i < out_size) {
    //     int cnt = 0;
    //     int low_idx = i;
    //     bool equal_flag = true;
    //     do{
    //         #pragma unroll 
    //         for (int j = 1; j < N; ++j) {
    //             equal_flag &= a[0][i] == a[j][i];
    //         }
    //         if (equal_flag) {
    //             i++; cnt++; 
    //         } else {
    //             break;
    //         }
    //     } while(i < out_size);
    //     TensorsReorganise<T, vec_t, N>(&a, out, out_size, i, low_idx, cnt);
    // }; 
    DEBUG_LOG;
    
    int min_idx = 0; 
    int min_val = std::accumulate(a[0].begin(), a[0].end(), 1, std::multiplies<T>());
    #pragma unroll
    for (int j=1; j < N; ++j) {
        int temp = std::accumulate(a[j].begin(), a[j].end(), 1, std::multiplies<T>());
        min_val = min_val > temp ? temp : min_val;
        if (min_val == temp) {
            min_idx = j;
        }
    }
    std::swap(a[0], a[min_idx]);

    merge_ptr = merge_one;
    DimsReorganise<T, vec_t, N>(&a, out, out_size, merge_ptr);

    // i = 0;
    // while (i < out_size) {
    //     int cnt = 0;
    //     int low_idx = i;
    //     bool equal_flag = true;
    //     do{
    //         equal_flag &= a[0][i] == 1;
    //         if (equal_flag) {
    //             #pragma unroll
    //             for (int j = 1; j < N; ++j) {
    //                 equal_flag &= a[j][i] == (*out)[i];
    //             }
    //         }
    //         if (equal_flag) {
    //             i++; cnt++; 
    //         } else {
    //             break;
    //         }
    //     } while(i < out_size);
    //     TensorsReorganise<T, vec_t, N>(&a, out, out_size, i, low_idx, cnt);
    // }; 
    DEBUG_LOG;

    std::cout << "stride: " << std::endl;
    std::vector<vec_t> in_stride(N, vec_t(out_size, 1));
    #pragma unroll
    for (int j = 0; j < N; ++j){
        for (int i = 0; i < out_size; ++i) {
            if (a[j][i] == 1) {
                in_stride[j][i] = 0;
            } else if (i != 1) {
                auto temp = std::accumulate(a[j].begin(), a[j].begin() + i, 1, std::multiplies<T>());
                in_stride[j][i] = temp;
            }
        }
    }
} 

// template <typename vec_t, typename T>
// auto func1(vec_t  arr) {
//     return std::accumulate(arr.begin(), arr.end(), 1, std::multiplies<T>());
// }

// template <typename vec_t, typename T>
// auto func2(vec_t arr) {
//         return std::accumulate(arr.begin(), arr.end(), 0, std::plus<T>());
// }


int main() {
    //1.1 自定义类型测试
    using vec_t = std::vector<int>;

    //3. 维度合并测试
    vec_t   a{32, 2};
    vec_t   b{32, 1, 16, 12};
    vec_t out{32, 2, 16, 12};

    std::vector<vec_t> in_arr;
    in_arr.emplace_back(a);
    in_arr.emplace_back(b);
    std::reverse(out.begin(), out.end());
    MergeDims<vec_t, int, 2>(&in_arr, &out);

    std::cout << std::endl;
    std::cout << "out:\t"; for(int x = 0; x < out.size(); ++x) { std::cout << out[x] << "\t";} std::cout << std::endl;
    auto loader = TensorLoader<int, vec_t>(b, out);
    std::cout << loader.Funcp(b)  << std::endl;
    return 0;
} 
