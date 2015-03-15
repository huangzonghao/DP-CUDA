#include <stdio.h>
#include <math.h>

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

        printf("Computing %d: %.0f, referring %d\n", 
                current, current_values[current], parent);
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


__device__ unsigned
sum(unsigned *state, unsigned length) {

    unsigned acc = 0;
    for (int i = 0; i < length; i++) {
        acc += state[i];
    }
    return acc;
}


__device__ void
decode(unsigned *state, unsigned index) {

    for (int i = n_dimension - 1; i >= 0; i--) {
        state[i] = index % n_capacity;
        index /= n_capacity;
    }
    state[n_dimension] = 0;
}


__device__ unsigned
encode(unsigned *state) {

    unsigned acc = 0;
    for (unsigned i = 1; i < n_dimension + 1; i++) {
        acc *= n_capacity;
        acc += state[i];
    }
    return acc;
}


__device__ unsigned
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


__device__ float
deplete(unsigned *state, unsigned quantity) {

    return unit_salvage * substract(state, n_dimension, quantity);
}


__device__ float
hold(unsigned *state) {

    return unit_hold * sum(state, n_dimension);
}


__device__ float
order(unsigned *state, unsigned quantity) {

    state[n_dimension] = quantity;
    return unit_order * quantity;
}


__device__ float
sell(unsigned *state, unsigned quantity) {

    return unit_price * substract(state, n_dimension+1, quantity);
}


__device__ float
dispose(unsigned *state) {
    unsigned disposal = state[0];
    state[0] = 0;
    return unit_disposal * disposal;
}


__device__ float
revenue(unsigned *state,
        unsigned n_depletion,
        unsigned n_order,
        unsigned n_demand) {

    float depletion = deplete(state, n_depletion);
    float holding = hold(state);
    float ordering = order(state, n_order);
    float sales = sell(state, n_demand);
    float disposal = dispose(state);

    return depletion + holding + discount * (ordering + sales + disposal);
}


__global__ void
iter_kernel(float *current_values,
            unsigned *depletion,
            unsigned *order,
            float *future_values,
            unsigned d,
            unsigned c,
            unsigned batch_size) {

    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < batch_size) {
        unsigned current = c * batch_size + idx;
        unsigned parent = current - batch_size;

        current_values[current] = future_values[current];

        printf("Computing %d: %.2f, referring %d\n",
                current, current_values[current], parent);
    }
}


void
iter_states(float *current_values,
            unsigned *depletion,
            unsigned *order,
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
                                               future_values,
                                               d, c, batch_size);

        }
    }
}
