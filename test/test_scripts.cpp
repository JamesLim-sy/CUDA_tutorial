#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <numeric>

#define DEBUG_LOG {    \
    std::cout << out_size << std::endl;\
    std::cout << "a[0]:\t"; for (int x=0; x < out_size; ++x) {std::cout << a[0][x] << "\t"; } std::cout << std::endl;\
    std::cout << "a[1]:\t"; for (int x=0; x < out_size; ++x) {std::cout << a[1][x] << "\t"; } std::cout << std::endl;\
    std::cout << "out :\t"; for (int x=0; x < out_size; ++x) {std::cout << (*out)[x] << "\t"; } std::cout << std::endl;\
}

template <typename vec_t, typename T >
void MergeDims(std::vector<vec_t> * in_arr, vec_t *out) {
    std::vector<vec_t> a = *in_arr;
    auto out_size = out->size();
    vec_t vec_temp(out_size, 1);
    int N = a.size();

    // 1. 维度补充
    for (int i=0; i < N; ++i) {
        std::reverse(a[i].begin(), a[i].end());
        if (a[i].size() < out_size ) {
            int idx_in = 0, idx_out = 0;
            a[i].resize(out_size, 1);
            vec_temp.assign(out_size, 1);

            do {
                if (a[i][idx_in] == (*out)[idx_out] || a[i][idx_in] == 1 ) {
                    vec_temp[idx_out++] = a[i][idx_in++];
                } else {
                    idx_out++;
                }
            }while(idx_out < out_size);            
            std::copy(vec_temp.begin(), vec_temp.end(), a[i].begin());
        }
    }    
    DEBUG_LOG;

    // 2. 有连续相等得元素存在, 目前仅仅支持二元计算中的维度合并
    int i = 0;
    auto vec_reorganise = [] (vec_t *a, int l_idx, int m_idx) { 
        (*a)[m_idx-1] = std::accumulate(a->begin() + l_idx, a->begin() + m_idx,\
                         1, std::multiplies<T>());
        a->erase(a->begin() + l_idx, a->begin() + m_idx - 1); };

    while (i < out_size) {
        int cnt = 0;
        int low_idx = i;
        bool equal_flag = true;
        do{
            #pragma unroll 
            for (int j = 1; j < N; j++) {
                equal_flag &= a[0][i] == a[j][i];
            }
            if (equal_flag) {
                i++;
                cnt++; 
            } else {
                break;
            }
        } while(i < out_size);

        if (cnt > 1) {
            #pragma unroll
            for (int j=0; j < N; ++j) {
                vec_reorganise(&a[j], low_idx, i);
            } 
            vec_reorganise(out, low_idx, i);
            out_size -= --cnt;
            i -= cnt;
        } else if (cnt < 1) {
            i++;
        }
    }
    DEBUG_LOG;
    
    // 3. 连续 1 维度合并
    int min_idx = 0; 
    int min_val = std::accumulate(a[0].begin(), a[0].end(), 1, std::multiplies<T>());
    for (int j=1; j < N; ++j) {
        int temp = std::accumulate(a[j].begin(), a[j].end(), 1, std::multiplies<T>());
        min_val = min_val > temp ? temp : min_val;
        if (min_val == temp) {
            min_idx = j;
        }
    }
    std::swap(a[0], a[min_idx]);
    
    i = 0;
    while (i < out_size) {
        int cnt = 0;
        int low_idx = i;
        bool equal_flag = true;
        do{
            equal_flag &= a[0][i] == 1;
            if (equal_flag) {
                #pragma unroll
                for (int j = 1; j < N; ++j) {
                    equal_flag &= a[j][i] == (*out)[i];
                }
            }
            if (equal_flag) {
                i++;
                cnt++; 
            } else {
                break;
            }
        } while(i < out_size);

        if (cnt > 1) {
            #pragma unroll
            for (int j=0; j < N; ++j) {
                vec_reorganise(&a[j], low_idx, i);
            } 
            vec_reorganise(out, low_idx, i);
            out_size -= --cnt;
            i -= cnt;
        } else if (cnt < 1) {
            i++;
        }
    }
    DEBUG_LOG;
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
* 先进行维度补充, 再进行维度合并,
*/


int main() {
    //1. 向量化操作
    constexpr int num = 10;

    //1.1 自定义类型测试
    using vec_t = std::vector<int>;
    vec_t vec_data_1{1,2,3,4};
    vec_t vec_data;
    vec_data.reserve(num);

    //2. lambda 表达式
    auto vec_push = [&vec_data, num] (vec_t * a, int max_num) { 
                    for(int i; i < num; ++i) {
                        a->emplace_back(i);
                    } };
    vec_push(&vec_data, num);         


    //3. 维度合并测试
    vec_t   a{32, 2};
    vec_t   b{32, 1, 16, 12};
    vec_t out{32, 2, 16, 12};

    std::vector<vec_t> in_arr;
    in_arr.emplace_back(a);
    in_arr.emplace_back(b);
    std::reverse(out.begin(), out.end());
    MergeDims<vec_t, int>(&in_arr, &out);
} 