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
            float *aux_current_values,
            dp_int *depletion,
            dp_int *order,
            float *future_values,
            float *aux_future_values,
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
        int n_depletion1 = 0;
        int n_depletion2 = 0;
        int n_order1 = 0;
        int n_order2 = 0;

        float max_value1 = 0.0;
        float max_value2 = 0.0;

        struct Demand demand = demand_distribution_at_period[0];
        
        for (int i = 0; i <= currentsum; i++){        
            // Policy 1: order positive number q>0;
           
              if (currentsum- i < n_capacity){
                 int j1 = n_capacity- currentsum + i;
                 float expected_value1 = stateValue(current,i,j1,aux_future_values,demand, period) ;
                 
                       // Simply taking the moving maximum
                 if (expected_value1 > max_value1 + 1e-6) {
                    max_value1 = expected_value1;
                    n_depletion1 = i;
                    n_order1 = j1;
                 }
               }   
            // Policy 2: order q=0;
             int j2= 0;
             float expected_value2 = stateValue(current,i,j2,future_values,demand, period);
                        // Simply taking the moving maximum
             if (expected_value2 > max_value2 + 1e-6) {
                max_value2 = expected_value2;
                n_depletion2 = i;
                n_order2 = j2;
             }
         }  

    // Store the optimal point and value
         if (max_value1 > max_value2 + 1e-6 && n_order1 > 0){
            current_values[current] = max_value1;
            depletion[current] = (dp_int) n_depletion1;
            order[current] = (dp_int) n_order1;
         }
         else {
           current_values[current] = max_value2;
           depletion[current] = (dp_int) n_depletion2;
           order[current] = (dp_int) n_order2;
         }
   // then update aux_current_values
    int j =0;
    if (n_capacity- currentsum >0){
        j= n_capacity - currentsum;
    }
    aux_current_values[current] = stateValue(current, 0, j, aux_future_values,demand, period);  
    }
}


// Plain C function to interact with kernel
// The structure is essentially the same as init_states.
// If you feel confused, start from there!
void
iter_states(float *current_values,
            float *aux_current_values,
            dp_int *depletion,
            dp_int *order,
            float *future_values,
            float *aux_future_values,
            int period) {

    size_t num_states = std::pow(n_capacity, n_dimension);

    // The very first state 0,0,...,0
    iter_kernel<<<1, 1>>>(current_values,
                          aux_current_values,
                          depletion,
                          order,
                          future_values,
                          aux_future_values,
                          period,
                          0, 1);

    for (size_t d = 0; d < n_dimension; d++) {

        size_t batch_size = pow(n_capacity, d);

        dim3 block_dim, grid_dim;
        get_grid_dim(&block_dim, &grid_dim, batch_size);

        for (size_t batch_idx = 1; batch_idx < n_capacity; batch_idx++) {
            iter_kernel<<<grid_dim, block_dim>>>(current_values,
                                                 aux_current_values,
                                                 depletion,
                                                 order,
                                                 future_values,
                                                 aux_future_values,
                                                 period,
                                                 batch_idx,
                                                 batch_size);
        }
    }

    cudaDeviceSynchronize();
    cudaThreadSynchronize();

}
