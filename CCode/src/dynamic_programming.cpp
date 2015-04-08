#include <cmath>
#include <iostream>

#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "dynamic_programming.h"


int
main() {

    unsigned num_states = std::pow(n_capacity, n_dimension);

    // host_vector are vectors store in the main memory
    thrust::host_vector<float> h_current_values(num_states);
    thrust::host_vector<unsigned> h_depletion;
    thrust::host_vector<unsigned> h_order;

    // Copy the values from main memory to CUDA device memory
    thrust::device_vector<float> d_current_values = h_current_values;

    // First call of CUDA functions to initialize all values
    init_states(thrust::raw_pointer_cast(d_current_values.data()));

    // Copy back values to d_future_values
    thrust::device_vector<float> d_future_values = d_current_values;

    // vector in CUDA devices prefixed with d (device)
    // vector in main memory prefixed with h (host)
    thrust::device_vector<unsigned> d_depletion = thrust::host_vector<unsigned>(num_states);
    thrust::device_vector<unsigned> d_order = thrust::host_vector<unsigned>(num_states);

    // Copy the demand distribution from header configurations
    thrust::device_vector<float> d_demand_distribution = \
        thrust::host_vector<float>(demand_distribution,
                                   demand_distribution + (max_demand - min_demand) * sizeof(float));

    // Main loop
    for (unsigned i = 0; i < n_period; i++) {

        // The only computing step
        iter_states(thrust::raw_pointer_cast(d_current_values.data()),
                    thrust::raw_pointer_cast(d_depletion.data()),
                    thrust::raw_pointer_cast(d_order.data()),
                    thrust::raw_pointer_cast(d_demand_distribution.data()),
                    thrust::raw_pointer_cast(d_future_values.data()));

        // Iteration of DP
        d_future_values = d_current_values;

        // Copy the intermediate results back
        h_current_values = d_current_values;
        h_depletion = d_depletion;
        h_order = d_order;

        // Output for logging
        for (unsigned current = 0; current < pow(n_capacity, n_dimension); current++) {
            std::cout << "Calculating state: " << current << " depetion: " << h_depletion[current];
            std::cout << " order: " << h_order[current] << " value: " << h_current_values[current];
            std::cout << std::endl;
        }

    }

    return 0;
}
