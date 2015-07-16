#include <cmath>
#include <iostream>
#include <iomanip>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "/ghome/hzhangaq/DP-parellel-computing/CCode/src/dp_model.h"

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}


int
main() {

    size_t num_states = std::pow(n_capacity, n_dimension);

    float *h_current_values;
    float *h_future_values;
    dp_int *h_depletion;
    dp_int *h_order;

    checkCudaErrors(cudaHostAlloc((void **)&h_current_values,
                                  sizeof(float) * num_states,
                                  cudaHostAllocMapped));
    checkCudaErrors(cudaHostAlloc((void **)&h_future_values,
                                  sizeof(float) * num_states,
                                  cudaHostAllocMapped));
    checkCudaErrors(cudaHostAlloc((void **)&h_depletion,
                                  sizeof(dp_int) * num_states,
                                  cudaHostAllocMapped));
    checkCudaErrors(cudaHostAlloc((void **)&h_order,
                                  sizeof(dp_int) * num_states,
                                  cudaHostAllocMapped));

    float *d_current_values;
    float *d_future_values;
    dp_int *d_depletion;
    dp_int *d_order;

    cudaSetDeviceFlags(cudaDeviceMapHost);


    checkCudaErrors(cudaHostGetDevicePointer((void **)&d_current_values,
                                             (void *)h_current_values, 0));
    checkCudaErrors(cudaHostGetDevicePointer((void **)&d_future_values,
                                             (void *)h_future_values, 0));
    checkCudaErrors(cudaHostGetDevicePointer((void **)&d_depletion,
                                             (void *)h_depletion, 0));
    checkCudaErrors(cudaHostGetDevicePointer((void **)&d_order,
                                             (void *)h_order, 0));


    init_states(d_future_values);

//    std::cout << "depletion,order,value" << std::endl;

    for (int period = 0; period < n_period; period++) {

        iter_states(d_current_values,
                    d_depletion,
                    d_order,
                    d_future_values,
                    period);


  /*      for (int idx = 0; idx < num_states; idx++) {
            int exp = std::pow(n_capacity, n_dimension-1);
            int i = idx;
            for (int k = 0; k < n_dimension; k++) {
                if (k > 0) {
                    std::cout << ',';
                }
                std::cout << i / exp;
                i %= exp;
                exp /= n_capacity;
            }
            std::cout << '\t';
            std::cout << static_cast<int>(h_depletion[idx]) << ',';
            std::cout << static_cast<int>(h_order[idx]) << ',';
            std::cout << std::fixed << std::setprecision(4) << h_current_values[idx];
            std::cout << '\n';
        }
        std::cout << std::endl;  */

        float *tmp = d_future_values;
        d_future_values = d_current_values;
        d_current_values = tmp;

        tmp = h_future_values;
        h_future_values = h_current_values;
        h_current_values = tmp;
    }
    for (int idx = 0; idx < num_states; idx++) {
        int idxsum= 0;
        int idx_1 = idx;
        for (int i= n_dimension-1; i>= 0; i--){
            idxsum += idx_1 % n_capacity;
            idx_1 /= n_capacity;
        }
        if (idxsum <= initial_small){
        /*    int exp = std::pow(n_capacity, n_dimension-1);
            int i = idx;
            for (int k = 0; k < n_dimension; k++) {
                if (k > 0) {
                    std::cout << ',';
                }
                std::cout << i / exp;
                i %= exp;
                exp /= n_capacity;
            }
            std::cout << '\t';
            std::cout << static_cast<int>(d_depletion[idx]) << ',';
            std::cout << static_cast<int>(d_order[idx]) << ',';  */ 
            std::cout << std::fixed << std::setprecision(4) << d_future_values[idx];
            std::cout << '\n';
      }
    }
        std::cout << std::endl;



    checkCudaErrors(cudaFreeHost((void *)h_current_values));
    checkCudaErrors(cudaFreeHost((void *)h_future_values));
    checkCudaErrors(cudaFreeHost((void *)h_depletion));
    checkCudaErrors(cudaFreeHost((void *)h_order));

    return 0;
}
