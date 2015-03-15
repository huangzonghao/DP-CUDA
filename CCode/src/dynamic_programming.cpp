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

    thrust::host_vector<float> h_current_values(num_states);
    thrust::host_vector<unsigned> h_depletion;
    thrust::host_vector<unsigned> h_order;


    thrust::device_vector<float> d_current_values = h_current_values;

    init_states(thrust::raw_pointer_cast(d_current_values.data()));

    thrust::device_vector<float> d_future_values = d_current_values;

    thrust::device_vector<unsigned> d_depletion = thrust::host_vector<unsigned>(num_states);
    thrust::device_vector<unsigned> d_order = thrust::host_vector<unsigned>(num_states);

    for (unsigned i = 0; i < n_period; i++) {

        iter_states(thrust::raw_pointer_cast(d_current_values.data()),
                    thrust::raw_pointer_cast(d_depletion.data()),
                    thrust::raw_pointer_cast(d_order.data()),
                    thrust::raw_pointer_cast(d_future_values.data()));

        d_future_values = d_current_values;

        h_current_values = d_current_values;
        h_depletion = d_depletion;
        h_order = d_order;
    }

    return 0;
}
