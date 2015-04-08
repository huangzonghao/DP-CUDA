#ifndef DYNAMIC_PROGRAMMING_H
#define DYNAMIC_PROGRAMMING_H

extern void init_states(float *);
extern void iter_states(float *,
                        unsigned *,
                        unsigned *,
                        const float *,
                        float *);

// The number of n_thread for each block
// should be power of 2 and does not exceed 512
const unsigned n_thread = 512;

const unsigned n_period = 3;

const unsigned n_dimension = 6;
const unsigned n_capacity = 10;

const float unit_salvage = 1.0;
const float unit_hold = -0.5;
const float unit_order = -3.0;
const float unit_price = 5.0;
const float unit_disposal = -2.0;

const float discount = 0.95;

const unsigned min_demand = 0;
const unsigned max_demand = 20;

const float demand_distribution[max_demand - min_demand] = {
    0.00033546, 0.0026837,  0.0107348,  0.02862614, 0.05725229,
    0.09160366, 0.12213822, 0.13958653, 0.13958653, 0.12407692,
    0.09926153, 0.07219021, 0.0481268 , 0.02961649, 0.01692371,
    0.00902598, 0.00451299, 0.00212376, 0.00094389, 0.00039743
};

#endif /* DYNAMIC_PROGRAMMING_H */
