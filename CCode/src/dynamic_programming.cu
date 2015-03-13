#include <stdio.h>
#include <math.h>


__global__ void
kernel(float* states,
       unsigned n_dimension,
       unsigned n_capacity,
       unsigned d,
       unsigned c,
       unsigned batch_size) {

    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < batch_size) {
        unsigned current = c * batch_size + idx;
        unsigned parent = current - batch_size;

        states[current] = states[parent] + 1.0;

        printf("Computing %d: %.0f, referring %d\n", current, states[current], parent);
    }
}


void
init_states(float* device_values,
            unsigned n_dimension,
            unsigned n_capacity) {

    unsigned num_states = std::pow(n_capacity, n_dimension);

    for (unsigned d = 0; d < n_dimension; d++) {
        unsigned batch_size = pow(n_capacity, d);
        unsigned n_thread = 512;
        unsigned n_block = batch_size / n_thread + 1;
        for (unsigned c = 1; c < n_capacity; c++) {
            kernel<<<n_block, n_thread>>>(device_values,
                                          n_dimension,
                                          n_capacity,
                                          d, c, batch_size);

        }
    }
}
