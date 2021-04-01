#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>

template <typename vec_t>
void MergeDims(std::vector<vec_t> * a, vec_t *c) {
    auto out_size = c->size();
    std::vector<int> vec_temp(out_size);  
/*
    vec_t a {1,5,4};
    vec_t b {2};
    vec_t c {2,5,1};
    vec_t d {3,1,5};
    vec_t e {3,2};
    vec_t f {2,5};
    vec_t g {5,4};

       out  {3,2,5,4}; 
*/
    for (vec_t v : *a) {
        std::reverse(v.begin(), v.end());
    
        if (v.size() != out_size ) {
            std::fill(vec_temp.begin(), vec_temp.end(), 1);
            int idx_in = 0, idx_out = 0;
            v.reserve(out_size);
            do {
                if (v[idx_in] == (*c)[idx_out] ||
                    v[idx_in] == 1 ) {
                    vec_temp[idx_out++] = v[idx_in++];
                } else {
                    idx_out++;
                }
            }while(idx_out < out_size);
            v.assign(vec_temp.begin(), vec_temp.end());
        }
        for (int i=0; i < out_size; ++i) { std::cout << v[i] << "\t"; } std::cout << std::endl;
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

    //6. 模板元编译完成 switch 操作

    //7. 维度合并测试
    vec_t a{1,5,4};
    vec_t b{2};
    vec_t c{2,5,1};
    vec_t d{3,1,5};
    vec_t e{3,2};
    vec_t f{2,5};
    vec_t g{5,4};
    vec_t out{3,2,5,4};
    std::reverse(out.begin(), out.end());

    std::vector<vec_t> in_arr;
    // in_arr.emplace_back(a);
    // in_arr.emplace_back(b);
    // in_arr.emplace_back(c);
    // in_arr.emplace_back(d);
    in_arr.emplace_back(e);
    // in_arr.emplace_back(f);
    in_arr.emplace_back(g);
    MergeDims<vec_t>(&in_arr, &out);
}