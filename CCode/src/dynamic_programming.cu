#include <stdio.h>
#include <math.h>

#include <cuda.h>

#include "dynamic_programming.h"

__device__ inline size_t
getGlobalIdx_3D_1D() {
    size_t blockId = blockIdx.x +
                     blockIdx.y * gridDim.x +
                     gridDim.x * gridDim.y * blockIdx.z;
    return blockId * blockDim.x + threadIdx.x;
}


// CUDA Kernel function for initialization
__global__ void
init_kernel(float *current_values,
            size_t d,
            size_t c,
            size_t batch_size) {

    size_t idx = getGlobalIdx_3D_1D();

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
        size_t n_thread = 512;
        size_t n_block = batch_size / n_thread + 1;

        dim3 block_dim(n_thread, 1, 1);
        dim3 grid_dim(4096, n_block / 4096 + 1, 1);

    for (unsigned d = 0; d < n_dimension; d++) {
        unsigned batch_size = pow(n_capacity, d);
        unsigned n_block = batch_size / n_thread + 1;
        for (size_t c = 1; c < n_capacity; c++) {
            init_kernel<<<grid_dim, block_dim>>>(current_values,
                                                 d, c, batch_size);
        }
    }
}


__device__ inline int
sum(int *state, int length) {

    int acc = 0;
    for (int i = 0; i < length; i++) {
        acc += state[i];
    }
    return acc;
}


// Decode index and store the result into
// the state array by overwritting
// turning base 10 index into base ``n_capacity``
// without the last entry
// as it is always called with respect to today
__device__ inline void
decode(int *state, int index) {

    for (int i = n_dimension - 1; i >= 0; i--) {
        state[i] = index % n_capacity;
        index /= n_capacity;
    }
    state[n_dimension] = 0;
}


// The inverse function of ``decode``
// without the 0-th entry
// as it is always called with respect to future
__device__ inline int
encode(int *state) {

    int acc = 0;
    for (int i = 1; i < n_dimension + 1; i++) {
        acc *= n_capacity;
        acc += state[i];
    }
    return acc;
}


// The common component for both deplete and sell
// Original values in ``state`` are overwritten
__device__ inline int
substract(int *state, int length, int quantity) {

    int acc = 0;
    for (int i = 0; i < length; i++) {
        if (quantity <= state[i]) {
            acc += quantity;
            state[i] -= quantity;
            break;
        } else {
            quantity -= state[i];
            acc += state[i];
            state[i] = 0;
        }
    }
    return acc;
}


// Original values in ``state`` are overwritten
__device__ inline float
deplete(int *state, int quantity) {

    return unit_salvage * substract(state, n_dimension, quantity);
}


// Simple summation
__device__ inline float
hold(int *state) {

    return unit_hold * sum(state, n_dimension);
}


// Original values in ``state`` are overwritten
__device__ inline float
order(int *state, int quantity) {

    state[n_dimension] = quantity;
    return unit_order * quantity;
}


// Original values in ``state`` are overwritten
__device__ inline float
sell(int *state, int quantity) {

    return unit_price * substract(state, n_dimension+1, quantity);
}


// Original values in ``state`` are overwritten
__device__ inline float
dispose(int *state) {
    int disposal = state[0];
    state[0] = 0;
    return unit_disposal * disposal;
}


// Original values in ``state`` are overwritten
__device__ inline float
revenue(int *state,
        size_t current,
        int n_depletion,
        int n_order,
        int n_demand) {

    float depletion = deplete(state, n_depletion);
    float holding = hold(state);
    float ordering = order(state, n_order);
    float sales = sell(state, n_demand);
    float disposal = dispose(state);
    float revenue = depletion + holding + discount * (ordering + sales + disposal);

    return revenue;
}


__device__ void
optimize(float *current_values,
         size_t current,
         dp_int *depletion,
         int min_depletion,
         int max_depletion,
         dp_int *order,
         int min_order,
         int max_order,
         float *future_values) {

    // Allocate a memory buffer on stack
    // So we don't need to do it for every loop
    // The last dimension are used to store the ordering
    int state[n_dimension+1] = {};
    decode(state, current);

    int n_depletion = 0;
    int n_order = 0;
    float max_value = 0.0;

    for (int i = min_depletion; i < max_depletion; i++) {
        for (int j = min_order; j < max_order; j++) {

            float expected_value = 0.0;

            for (int k = min_demand; k < max_demand; k++) {

                // Initialize the ``state`` array
                // before each call of ``revenue()``
                decode(state, current);

                // By calling ``revenue()``, the state array
                // now stores the state for future
                float value = revenue(state, current, i, j, k);

                // And find the corresponding utility of future
                int future = encode(state);

                value += discount * future_values[future];

                expected_value += demand_distribution[k - min_demand] * value;
            }

            // Simply taking the moving maximum
            if (expected_value > max_value + 1e-6) {
                max_value = expected_value;
                n_depletion = i;
                n_order = j;
            }
        }
    }

    // Store the optimal point and value
    current_values[current] = max_value;
    depletion[current] = (dp_int) n_depletion;
    order[current] = (dp_int) n_order;
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

    size_t idx = getGlobalIdx_3D_1D();

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
        size_t n_thread = 512;
        size_t n_block = batch_size / n_thread + 1;

        dim3 block_dim(n_thread, 1, 1);
        dim3 grid_dim(4096, n_block / 4096 + 1, 1);

        for (size_t c = 1; c < n_capacity; c++) {
            iter_kernel<<<grid_dim, block_dim>>>(current_values,
                                                 depletion,
                                                 order,
                                                 future_values,
                                                 d, c, batch_size);
        }
    }
}
