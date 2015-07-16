#include <stdio.h>
#include <math.h>
#include <assert.h>

#include <cuda.h>

// Hack: including source files is bad in general!
// But we know it is not going to be used anywhere else
// If you do, write a header file and include it instead!
#include "dp_model.cu"


// Helper function to get CUDA thread id
// whenever we use __device__ function
__device__ inline size_t
get_thread_id() {

    size_t blockId = blockIdx.x +
                     blockIdx.y * gridDim.x +
                     gridDim.x * gridDim.y * blockIdx.z;
    return blockId * blockDim.x + threadIdx.x;
}


// Using these values for general CUDA GPU is just fine
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
        size_t parent = (c > 0) ? (current - batch_size) : 0;

        current_values[current] = current_values[parent] + 1.0;
    }
}


// Plain C function for interact with kernel
void
init_states(float *current_values) {

    size_t num_states = std::pow(n_capacity, n_dimension);

    // The very first state
    init_kernel<<<1, 1>>>(current_values,
                          0, 0, 1);

    for (size_t d = 0; d < n_dimension; d++) {
        size_t batch_size = pow(n_capacity, d);

        dim3 block_dim, grid_dim;
        get_grid_dim(&block_dim, &grid_dim, batch_size);

        for (size_t c = 1; c < n_capacity; c++) {
            init_kernel<<<grid_dim, block_dim>>>(current_values,
                                                 d, c, batch_size);
        }
    }

    cudaDeviceSynchronize();
    cudaThreadSynchronize();

}


// The CUDA kernel function for DP
__global__ void
iter_kernel(float *current_values,
            dp_int *depletion,
            dp_int *order,
            float *future_values,
            int period,
            size_t d,
            size_t c,
            size_t batch_size) {

    size_t idx = get_thread_id();

    if (idx < batch_size) {

        size_t current = c * batch_size + idx;
        int state[n_dimension+1] = {};
        decode(state, current);
        int h_depletion = 0;
        int h_order = 0;
        if (period < n_period-1) {
           if (n_drate - sum(state, n_dimension+1) > 1e-6){
              h_order = (int) n_drate - sum(state, n_dimension+1);
           }
        }
        else {
           for (int i = 1; i < n_dimension + 1; i++) {
               if (sum(state, i)- i* n_drate > h_depletion + 1e-6){
                  h_depletion = sum(state, i)- i* n_drate;
               }  
           }
           if (n_drate - h_depletion - sum(state, n_dimension+1) > 1e-6){
              h_order = (int) n_drate - h_depletion - sum(state, n_dimension+1);
           }

        }
        optimize(current_values,
                     // the state we are computing
                  current,
                     // n_depletion, min_depletion, max_depletion
                  depletion, h_depletion, h_depletion +1,
                     // n_order, min_order, max_order
                  order, h_order, h_order + 1,
                     // future utility for reference
                  future_values,
                  period);

     }
}


// Plain C function to interact with kernel
// The structure is essentially the same as init_states.
// If you feel confused, start from there!
void
iter_states(float *current_values,
            dp_int *depletion,
            dp_int *order,
            float *future_values,
            int period) {

    size_t num_states = std::pow(n_capacity, n_dimension);

    // The very first state 0,0,...,0
    iter_kernel<<<1, 1>>>(current_values,
                          depletion,
                          order,
                          future_values,
                          period,
                          0, 0, 1);

    for (size_t d = 0; d < n_dimension; d++) {

        size_t batch_size = pow(n_capacity, d);

        dim3 block_dim, grid_dim;
        get_grid_dim(&block_dim, &grid_dim, batch_size);

        for (size_t c = 1; c < n_capacity; c++) {
            iter_kernel<<<grid_dim, block_dim>>>(current_values,
                                                 depletion,
                                                 order,
                                                 future_values,
                                                 period,
                                                 d, c, batch_size);
        }
    }

    cudaDeviceSynchronize();
    cudaThreadSynchronize();

}

