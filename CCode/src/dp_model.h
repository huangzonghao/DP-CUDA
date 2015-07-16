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

#define MAX_DISTRIBUTION_LENGTH 60


extern void init_states(float *);
extern void iter_states(float *,
                        dp_int *,
                        dp_int *,
                        float *,
                        int);


const unsigned n_period = 15;

const int n_dimension = 2;
const int n_capacity = 29;
const int n_drate= 25;
const int cvalue = 100000;
const int initial_small= 100000;

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
demand_distribution_at_period[1] = {
    // Demand struct for 1st period
    {
        0,
        39,
        {1.38879439e-11,   3.47198597e-10,   4.33998246e-09,
         3.61665205e-08,   2.26040753e-07,   1.13020377e-06,
         4.70918235e-06,   1.68185084e-05,   5.25578388e-05,
         1.45993997e-04,   3.64984992e-04,   8.29511344e-04,
         1.72814863e-03,   3.32336276e-03,   5.93457635e-03,
         9.89096059e-03,   1.54546259e-02,   2.27273911e-02,
         3.15658209e-02,   4.15339749e-02,   5.19174686e-02,
         6.18065102e-02,   7.02346707e-02,   7.63420334e-02,
         7.95229515e-02,   7.95229515e-02,   7.64643764e-02,
         7.08003485e-02,   6.32145969e-02,   5.44953422e-02,
         4.54127851e-02,   3.66232138e-02,   2.86118858e-02,
         2.16756711e-02,   1.59379934e-02,   1.13842810e-02,
         7.90575071e-03,   5.34172345e-03,   3.51429174e-03}
    }
};

/*   // Demand struct for 2nd period
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
  {    0,
      20,
       {
            0.00033546, 0.0026837,  0.0107348,  0.02862614, 0.05725229,
            0.09160366, 0.12213822, 0.13958653, 0.13958653, 0.12407692,
            0.09926153, 0.07219021, 0.0481268 , 0.02961649, 0.01692371,
            0.00902598, 0.00451299, 0.00212376, 0.00094389, 0.00039743
        }
    },
{    0,
      20,
       {
            0.00033546, 0.0026837,  0.0107348,  0.02862614, 0.05725229,
            0.09160366, 0.12213822, 0.13958653, 0.13958653, 0.12407692,
            0.09926153, 0.07219021, 0.0481268 , 0.02961649, 0.01692371,
            0.00902598, 0.00451299, 0.00212376, 0.00094389, 0.00039743
        }
    },
    {    0,
      20,
       {
            0.00033546, 0.0026837,  0.0107348,  0.02862614, 0.05725229,
            0.09160366, 0.12213822, 0.13958653, 0.13958653, 0.12407692,
            0.09926153, 0.07219021, 0.0481268 , 0.02961649, 0.01692371,
            0.00902598, 0.00451299, 0.00212376, 0.00094389, 0.00039743
        }
    }


    // and so on
};*/


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
