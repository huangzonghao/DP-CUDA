#include <cmath>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "dynamic_programming.h"

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

    cudaHostGetDevicePointer((void **)&d_current_values,
                                             (void *)h_current_values, 0);
    cudaHostGetDevicePointer((void **)&d_future_values,
                                             (void *)h_future_values, 0);
    cudaHostGetDevicePointer((void **)&d_depletion,
                                             (void *)h_depletion, 0);
    cudaHostGetDevicePointer((void **)&d_order,
                                             (void *)h_order, 0);

    init_states(d_current_values);


    checkCudaErrors(cudaHostGetDevicePointer((void **)&d_current_values,
                                             (void *)h_current_values, 0));
    checkCudaErrors(cudaHostGetDevicePointer((void **)&d_future_values,
                                             (void *)h_future_values, 0));
    checkCudaErrors(cudaHostGetDevicePointer((void **)&d_depletion,
                                             (void *)h_depletion, 0));
    checkCudaErrors(cudaHostGetDevicePointer((void **)&d_order,
                                             (void *)h_order, 0));


    init_states(d_current_values);

    for (int i = 0; i < n_period; i++) {

        iter_states(d_current_values,
                    d_depletion,
                    d_order,
                    d_future_values);

        checkCudaErrors(cudaDeviceSynchronize());


        for (size_t current = num_states-100; current < num_states; current++) {
            std::cout << "Calculating state: " << current << " depletion: " << static_cast<int>(h_depletion[current]);
            std::cout << " order: " << static_cast<int>(h_order[current]) << " value: " << h_current_values[current];
            std::cout << std::endl;
        }
    }

    checkCudaErrors(cudaFreeHost((void *)h_current_values));
    checkCudaErrors(cudaFreeHost((void *)h_future_values));
    checkCudaErrors(cudaFreeHost((void *)h_depletion));
    checkCudaErrors(cudaFreeHost((void *)h_order));

    return 0;
}
