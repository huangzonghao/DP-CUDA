#include <cmath>
#include <iostream>

#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

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

    /*
    thrust::host_vector<float> h_current_values(num_states);
    thrust::host_vector<dp_int> h_depletion;
    thrust::host_vector<dp_int> h_order;

    thrust::device_vector<float> d_current_values = h_current_values;
    */

    float *d_current_values;
    float *d_future_values;
    dp_int *d_depletion;
    dp_int *d_order;

    // checkCudaErrors(cudaDeviceReset());
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

    /*
    init_states(thrust::raw_pointer_cast(d_current_values.data()));

    thrust::device_vector<float> d_future_values = d_current_values;

    thrust::device_vector<dp_int> d_depletion = thrust::host_vector<dp_int>(num_states);
    thrust::device_vector<dp_int> d_order = thrust::host_vector<dp_int>(num_states);
    */

    thrust::device_vector<float> d_demand_distribution = \
        thrust::host_vector<float>(demand_distribution,
                                   demand_distribution + (max_demand - min_demand) * sizeof(float));

    for (int i = 0; i < n_period; i++) {

        iter_states(d_current_values,
                    d_depletion,
                    d_order,
                    thrust::raw_pointer_cast(d_demand_distribution.data()),
                    d_future_values);
        /*
        iter_states(thrust::raw_pointer_cast(d_current_values.data()),
                    thrust::raw_pointer_cast(d_depletion.data()),
                    thrust::raw_pointer_cast(d_order.data()),
                    thrust::raw_pointer_cast(d_demand_distribution.data()),
                    thrust::raw_pointer_cast(d_future_values.data()));

        float *temp = d_future_values;
        d_future_values = d_current_values;
        d_current_values = temp;
        */

        checkCudaErrors(cudaDeviceSynchronize());

        // for (size_t current = 0; current < num_states; current++) {

        for (size_t current = num_states-100; current < num_states; current++) {
            std::cout << "Calculating state: " << current << " depetion: " << static_cast<int>(h_depletion[current]);
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
