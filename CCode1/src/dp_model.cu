#include "dp_model.h"


// Handcraft array summation
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
         float *future_values,
         int period) {

    // Allocate a memory buffer on stack
    // So we don't need to do it for every loop
    // Last dimension are used to store the ordering
    // which could be used for sale
    int state[n_dimension+1] = {};
    decode(state, current);

    int n_depletion = 0;
    int n_order = 0;
    float max_value = 0.0;

    struct Demand demand = demand_distribution_at_period[0];

    for (int i = min_depletion; i < max_depletion; i++) {
        for (int j = min_order; j < max_order; j++) {

            float expected_value = 0.0;

            for (int k = demand.min_demand; k < demand.max_demand; k++) {

                // Always initialize state array before calling of revenue()
                // As the value is corrupted and can't be used again
                decode(state, current);

                // By calling revenue(), the state array
                // now stores the state for future
                float value = revenue(state, current, i, j, k);

                // And find the corresponding utility of future
                int future = encode(state);

                value += discount * future_values[future];

                expected_value += demand.distribution[k - demand.min_demand] * value;
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
