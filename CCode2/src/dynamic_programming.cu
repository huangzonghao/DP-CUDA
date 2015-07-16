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
            size_t batch_idx,
            size_t batch_size) {

    size_t thread_idx = get_thread_id();

    if (thread_idx < batch_size) {

        size_t current = batch_idx * batch_size + thread_idx;
        size_t parent = current - batch_size;

        if (current == 0) {
            current_values[current] = 0.0;
        } else {
            current_values[current] = current_values[parent] + unit_salvage;
        }
    }
}


// Plain C function for interact with kernel
void
init_states(float *current_values) {

    size_t num_states = std::pow(n_capacity, n_dimension);

    // The very first state
    init_kernel<<<1, 1>>>(current_values, 0, 1);

    for (size_t d = 0; d < n_dimension; d++) {

        size_t batch_size = pow(n_capacity, d);

        dim3 block_dim, grid_dim;
        get_grid_dim(&block_dim, &grid_dim, batch_size);

        for (size_t batch_idx = 1; batch_idx < n_capacity; batch_idx++) {
            init_kernel<<<grid_dim, block_dim>>>(current_values,
                                                 batch_idx,
                                                 batch_size);
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
            size_t batch_idx,
            size_t batch_size) {

    size_t thread_idx = get_thread_id();

    if (thread_idx < batch_size) {
       // first update current_values

        size_t current = batch_idx * batch_size + thread_idx;
       // size_t parent = current - batch_size;

        int state[n_dimension+1] = {};
        decode(state, current);
        int currentsum = sum(state, n_dimension+1);
        int n_depletion = 0;
        int n_order = 0;

        float max_value = 0.0;
    
        struct Demand demand = demand_distribution_at_period[0];
     
        // Case 1: period < T-L-1;
        if (period < n_period- n_dimension){
               n_depletion= 0;
               n_order =0;
               if (n_capacity-1- currentsum >0){
                  n_order= n_capacity-1 - currentsum;
               }
               current_values[current] = stateValue(current, n_depletion, n_order, future_values,demand, period);  
               depletion[current] = (dp_int) n_depletion;
               order[current] = (dp_int) n_order;
        }
        // Case 2
        else {
           for (int i = 0; i <= currentsum; i++){        
               int j= 0;
               if (currentsum- i < n_capacity-1){
                  j = n_capacity-1- currentsum + i;
               }
               float expected_value = stateValue(current,i,j,future_values,demand, period) ;
                
                 // Simply taking the moving maximum
               if (expected_value > max_value + 1e-6) {
                    max_value = expected_value;
                    n_depletion = i;
                    n_order = j;
                 }
           }   
           current_values[current] = max_value;
            depletion[current] = (dp_int) n_depletion;
            order[current] = (dp_int) n_order;
         }
        
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
                          0, 1);

    for (size_t d = 0; d < n_dimension; d++) {

        size_t batch_size = pow(n_capacity, d);

        dim3 block_dim, grid_dim;
        get_grid_dim(&block_dim, &grid_dim, batch_size);

        for (size_t batch_idx = 1; batch_idx < n_capacity; batch_idx++) {
            iter_kernel<<<grid_dim, block_dim>>>(current_values,
                                                 depletion,
                                                 order,
                                                 future_values,
                                                 period,
                                                 batch_idx,
                                                 batch_size);
        }
    }

    cudaDeviceSynchronize();
    cudaThreadSynchronize();

}
