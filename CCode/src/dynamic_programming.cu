#include <stdio.h>
#include <math.h>
#include <assert.h>

#include <cuda.h>

#include "dp_model.cu"


inline void
get_grid_dim(dim3* block_dim, dim3* grid_dim, size_t batch_size) {

    size_t n_block = batch_size / 512 + 1;

    assert(block_dim && grid_dim);
    *block_dim = dim3(512, 1, 1);
    *grid_dim = dim3(4096, n_block / 4096 + 1, 1);
}


// CUDA Kernel function for initialization
__global__ void
init_kernel(float *current_values,
            size_t d,
            size_t c,
            size_t batch_size) {

    size_t idx = get_thread_id();

    if (idx < batch_size) {
        size_t current = c * batch_size + idx;
        size_t parent = current - batch_size;

        current_values[current] = current_values[parent] + 1.0;
    }
}


// plain C function for interact with CUDA
void
init_states(float *current_values) {

    size_t num_states = std::pow(n_capacity, n_dimension);

    for (size_t d = 0; d < n_dimension; d++) {
        size_t batch_size = pow(n_capacity, d);

        dim3 block_dim, grid_dim;
        get_grid_dim(&block_dim, &grid_dim, batch_size);

        for (size_t c = 1; c < n_capacity; c++) {
            init_kernel<<<grid_dim, block_dim>>>(current_values,
                                                 d, c, batch_size);
        }
    }
}


// The CUDA kernel function for DP
__global__ void
iter_kernel(float *current_values,
            dp_int *depletion,
            dp_int *order,
            float *future_values,
            size_t d,
            size_t c,
            size_t batch_size) {

    size_t idx = get_thread_id();

    if (idx < batch_size) {

        size_t current = c * batch_size + idx;
        size_t parent = current - batch_size;

        if (depletion[parent] == 0) {

            optimize(current_values,
                     current,
                     // n_depletion: optimal point and range [min, max)
                     depletion,
                     0,
                     2,
                     // n_order: optimal point and range [min, max)
                     order,
                     0,
                     n_capacity,
                     // future utility for reference
                     future_values);

        } else /* (depletion[parent] != 0) */ {

            optimize(current_values,
                     current,
                     // n_depletion: optimal point and range [min, max)
                     depletion,
                     depletion[parent]+1,
                     depletion[parent]+2,
                     // n_order: optimal point and range [min, max)
                     order,
                     0,
                     n_capacity,
                     // future utility for reference
                     future_values);

        }

        __threadfence_system();
    }
}


// The plain C function to interact with CUDA
void
iter_states(float *current_values,
            dp_int *depletion,
            dp_int *order,
            float *future_values) {

    size_t num_states = std::pow(n_capacity, n_dimension);

    for (size_t d = 0; d < n_dimension; d++) {

        size_t batch_size = pow(n_capacity, d);

        dim3 block_dim, grid_dim;
        get_grid_dim(&block_dim, &grid_dim, batch_size);

        for (size_t c = 1; c < n_capacity; c++) {
            iter_kernel<<<grid_dim, block_dim>>>(current_values,
                                                 depletion,
                                                 order,
                                                 future_values,
                                                 d, c, batch_size);
        }
    }

    cudaDeviceSynchronize();
    cudaThreadSynchronize();

}
