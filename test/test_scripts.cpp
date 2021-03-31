#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>

template <typename vec_t>
void functor(vec_t * vec_data) {
    for (int i : *vec_data) { std::cout << i << "\t";}
    std::cout << std::endl;
}


/*
* 5. [32, 2, 16, 12] + [32, 2 ] => [64, 192]    + [64, 1]
* 6. [32, 2, 16, 12] + [16, 12] => [32, 2, 192] + [1, 1, 192]
*/
template <int N, typename vec_t>
void MergeDims(std::vector<vec_t> * a, vec_t *c) {
    auto out_size = c->size();
    std::vector<int> v_idx(N, 0);
    
    #pragma unroll 
    for (vec_t v : *a) {
        std::reverse(v.begin(), v.end());

        if (v.size() != out_size ) {
            std::fill(v_idx.begin(), v_idx.end(), 0);
            int idx_in = 0, idx_out = 0;
            do {
                


            }while(0);
        }
    }
}


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

    //3. for 循环测试
    // for (int i : vec_data) { std::cout << i << "\t";}
    std::cout << std::endl;

    //4. std::reverse测试, 需要配合 #include <algorithm>
    std::reverse(vec_data.begin(), vec_data.end());
    // for (int i : vec_data) { std::cout << i << "\t";}
    std::cout << std::endl;

    //5. 传参测试
    functor(&vec_data);

    //6. 模板元编译完成 switch 操作


    //7. 维度合并测试
    vec_t a{3,1,5,4};
    vec_t b{2};
    vec_t c{3,2,5,4};
    std::reverse(c.begin(), c.end());

    std::vector<vec_t>in_arr;
    in_arr.reserve(2);
    in_arr.emplace_back(a);
    in_arr.emplace_back(b);
    MergeDims<2, vec_t>(&in_arr, &c);



}