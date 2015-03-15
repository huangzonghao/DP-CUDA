#ifndef DYNAMIC_PROGRAMMING_H
#define DYNAMIC_PROGRAMMING_H

extern void init_states(float *);
extern void iter_states(float *,
                        unsigned *,
                        unsigned *,
                        float *);

const unsigned n_period = 1;

const unsigned n_dimension = 3;
const unsigned n_capacity = 10;

const float unit_salvage = 1.0;
const float unit_disposal = -2.0;
const float unit_hold = -0.5;
const float unit_order = -3.0;
const float unit_price = 5.0;

const float demand_rate = 8.0;
const float discount = 0.95;

#endif /* DYNAMIC_PROGRAMMING_H */
