#ifndef DYNAMIC_PROGRAMMING_H
#define DYNAMIC_PROGRAMMING_H

#include <stdint.h>

#define dp_int int_fast8_t

#ifndef __device__
#define __device__
#endif

#ifndef __constant__
#define __constant__
#endif

extern void init_states(float *);
extern void iter_states(float *,
                        dp_int *,
                        dp_int *,
                        float *);


const unsigned n_period = 3;

const int n_dimension = 9;
const int n_capacity = 10;

const float unit_salvage = 1.0;
const float unit_hold = -0.5;
const float unit_order = -3.0;
const float unit_price = 5.0;
const float unit_disposal = -2.0;

const float discount = 0.95;

const int min_demand = 0;
const int max_demand = 20;

__device__ __constant__
const float demand_distribution[max_demand - min_demand] = {
    0.00033546, 0.0026837,  0.0107348,  0.02862614, 0.05725229,
    0.09160366, 0.12213822, 0.13958653, 0.13958653, 0.12407692,
    0.09926153, 0.07219021, 0.0481268 , 0.02961649, 0.01692371,
    0.00902598, 0.00451299, 0.00212376, 0.00094389, 0.00039743
};

#endif /* DYNAMIC_PROGRAMMING_H */
