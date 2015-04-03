#include <stdio.h>
#include <math.h>

#include <cuda.h>

#include "dynamic_programming.h"

__global__ void
init_kernel(float *current_values,
            unsigned d,
            unsigned c,
            unsigned batch_size) {

    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < batch_size) {
        unsigned current = c * batch_size + idx;
        unsigned parent = current - batch_size;

        current_values[current] = current_values[parent] + 1.0;
    }
}


void
init_states(float *current_values) {

    unsigned num_states = std::pow(n_capacity, n_dimension);

    for (unsigned d = 0; d < n_dimension; d++) {
        unsigned batch_size = pow(n_capacity, d);
        unsigned n_thread = 512;
        unsigned n_block = batch_size / n_thread + 1;
        for (unsigned c = 1; c < n_capacity; c++) {
            init_kernel<<<n_block, n_thread>>>(current_values,
                                               d, c, batch_size);
        }
    }
}


__device__ inline unsigned
sum(unsigned *state, unsigned length) {

    unsigned acc = 0;
    for (int i = 0; i < length; i++) {
        acc += state[i];
    }
    return acc;
}


__device__ inline void
decode(unsigned *state, unsigned index) {

    for (int i = n_dimension - 1; i >= 0; i--) {
        state[i] = index % n_capacity;
        index /= n_capacity;
    }
    state[n_dimension] = 0;
}


__device__ inline unsigned
encode(unsigned *state) {

    unsigned acc = 0;
    for (unsigned i = 1; i < n_dimension + 1; i++) {
        acc *= n_capacity;
        acc += state[i];
    }
    return acc;
}


__device__ inline unsigned
substract(unsigned *state, unsigned length, unsigned quantity) {

    unsigned acc = 0;
    for (unsigned i = 0; i < length; i++) {
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


__device__ inline float
deplete(unsigned *state, unsigned quantity) {

    return unit_salvage * substract(state, n_dimension, quantity);
}


__device__ inline float
hold(unsigned *state) {

    return unit_hold * sum(state, n_dimension);
}


__device__ inline float
order(unsigned *state, unsigned quantity) {

    state[n_dimension] = quantity;
    return unit_order * quantity;
}


__device__ inline float
sell(unsigned *state, unsigned quantity) {

    return unit_price * substract(state, n_dimension+1, quantity);
}


__device__ inline float
dispose(unsigned *state) {
    unsigned disposal = state[0];
    state[0] = 0;
    return unit_disposal * disposal;
}


__device__ inline float
revenue(unsigned *state,
        unsigned current,
        unsigned n_depletion,
        unsigned n_order,
        unsigned n_demand) {

    decode(state, current);

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
         unsigned current,
         unsigned *depletion,
         unsigned min_depletion,
         unsigned max_depletion,
         unsigned *order,
         unsigned min_order,
         unsigned max_order,
         const float *demand_pdf,
         unsigned min_demand,
         unsigned max_demand,
         float *future_values) {

    unsigned state[n_dimension+1] = {};
    decode(state, current);

    unsigned n_depletion = 0;
    unsigned n_order = 0;
    float max_value = 0.0;

    for (unsigned i = min_depletion; i < max_depletion; i++) {
        for (unsigned j = min_order; j < max_order; j++) {

            float expected_value = 0.0;

            for (unsigned k = min_demand; k < max_demand; k++) {

                float value = revenue(state, current, i, j, k);
                unsigned future = encode(state);
                value += discount * future_values[future];

                expected_value += demand_pdf[k - min_demand] * value;
            }
            if (expected_value > max_value + 1e-6) {
                max_value = expected_value;
                n_depletion = i;
                n_order = j;
            }
        }
    }

    current_values[current] = max_value;
    depletion[current] = n_depletion;
    order[current] = n_order;
}


__global__ void
iter_kernel(float *current_values,
            unsigned *depletion,
            unsigned *order,
            const float *demand_pdf,
            unsigned min_demand,
            unsigned max_demand,
            float *future_values,
            unsigned d,
            unsigned c,
            unsigned batch_size) {

    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < batch_size) {

        unsigned current = c * batch_size + idx;
        unsigned parent = current - batch_size;

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
                     // n_demand: probability distribution and range [min, max)
                     demand_pdf,
                     min_demand,
                     max_demand,
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
                     // n_demand: probability distribution and range [min, max)
                     demand_pdf,
                     min_demand,
                     max_demand,
                     // future utility for reference
                     future_values);

        }

    }
}


void
iter_states(float *current_values,
            unsigned *depletion,
            unsigned *order,
            const float *demand_pdf,
            float *future_values) {

    unsigned num_states = std::pow(n_capacity, n_dimension);

    for (unsigned d = 0; d < n_dimension; d++) {
        unsigned batch_size = pow(n_capacity, d);
        unsigned n_thread = 512;
        unsigned n_block = batch_size / n_thread + 1;
        for (unsigned c = 1; c < n_capacity; c++) {
            iter_kernel<<<n_block, n_thread>>>(current_values,
                                               depletion,
                                               order,
                                               demand_pdf,
                                               min_demand,
                                               max_demand,
                                               future_values,
                                               d, c, batch_size);
        }
    }
}
