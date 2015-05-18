#ifndef DYNAMIC_PROGRAMMING_H
#define DYNAMIC_PROGRAMMING_H

#include <stdint.h>

// Use 8-bit integer to save space
// Remember to cast it to 32-bit int
// when you actually use it!
#define dp_int int_fast8_t

#ifndef __device__
#define __device__
#endif

#ifndef __constant__
#define __constant__
#endif

#define MAX_DISTRIBUTION_LENGTH 50


extern void init_states(float *);
extern void iter_states(float *,
                        dp_int *,
                        dp_int *,
                        float *,
                        int);


const unsigned n_period = 2;

const int n_dimension = 2;
const int n_capacity = 10;

const float unit_salvage = 1.0;
const float unit_hold = -0.5;
const float unit_order = -3.0;
const float unit_price = 5.0;
const float unit_disposal = -2.0;

const float discount = 0.95;

struct Demand {
    int min_demand;
    int max_demand;
    float distribution[MAX_DISTRIBUTION_LENGTH];
};


__device__ __constant__ struct Demand
demand_distribution_at_period[n_period] = {
    // Demand struct for 1st period
    {
        0,
        20,
        {
            0.00033546, 0.0026837,  0.0107348,  0.02862614, 0.05725229,
            0.09160366, 0.12213822, 0.13958653, 0.13958653, 0.12407692,
            0.09926153, 0.07219021, 0.0481268 , 0.02961649, 0.01692371,
            0.00902598, 0.00451299, 0.00212376, 0.00094389, 0.00039743
        }
    },
    // Demand struct for 2nd period
    {
        0,
        20,
        {
            0.00033546, 0.0026837,  0.0107348,  0.02862614, 0.05725229,
            0.09160366, 0.12213822, 0.13958653, 0.13958653, 0.12407692,
            0.09926153, 0.07219021, 0.0481268 , 0.02961649, 0.01692371,
            0.00902598, 0.00451299, 0.00212376, 0.00094389, 0.00039743
        }
    }
    // and so on
};


/*
 * Alternatively, if you don't want to supply the same distribution
 * multiple times, initialize Demand by yourself using macro and supply
 * it to the array constructor.
 *
 * Example:

#define demand1 { \
    0, \
    20, \
    { \
        0.00033546, 0.0026837,  0.0107348,  0.02862614, 0.05725229, \
        0.09160366, 0.12213822, 0.13958653, 0.13958653, 0.12407692, \
        0.09926153, 0.07219021, 0.0481268 , 0.02961649, 0.01692371, \
        0.00902598, 0.00451299, 0.00212376, 0.00094389, 0.00039743 \
    } \
}

__device__ __constant__ struct Demand
demand_distribution_at_period[n_period] = { demand1, demand1 };

 * Caveat:
 * Macros can only be one line, so use "\" to escape new line character.
 * Using variable is more readable, but nvcc has limit on constant variable
 * initialization, so consider it as an ugly hack.
 */

#endif /* DYNAMIC_PROGRAMMING_H */
