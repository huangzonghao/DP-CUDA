#include <cmath>
#include <iostream>

#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>


const unsigned n_dimension = 3;
const unsigned n_capacity = 10;
const float unit_depletion = 1.0;

void init_states(float*, unsigned, unsigned);


int
main() {

    unsigned num_states = std::pow(n_capacity, n_dimension);

    thrust::host_vector<float> current_values(num_states);
    thrust::device_vector<float> device_values = current_values;

    init_states(thrust::raw_pointer_cast(device_values.data()),
                n_dimension, n_capacity);

    return 0;
}
