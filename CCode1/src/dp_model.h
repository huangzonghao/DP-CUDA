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

const int n_dimension = 1;
const int n_capacity = 40;
const float n_drate = 35;
const int initial_small = 10000;
const int cvalue= 100000;

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
        51,
        {6.30511676e-16,   2.20679087e-14,   3.86188402e-13,
         4.50553135e-12,   3.94233993e-11,   2.75963795e-10,
         1.60978881e-09,   8.04894403e-09,   3.52141301e-08,
         1.36943839e-07,   4.79303438e-07,   1.52505639e-06,
         4.44808115e-06,   1.19756031e-05,   2.99390077e-05,
         6.98576847e-05,   1.52813685e-04,   3.14616411e-04,
         6.11754132e-04,   1.12691551e-03,   1.97210214e-03,
         3.28683689e-03,   5.22905869e-03,   7.95726323e-03,
         1.16043422e-02,   1.62460791e-02,   2.18697219e-02,
         2.83496394e-02,   3.54370493e-02,   4.27688526e-02,
         4.98969947e-02,   5.63353166e-02,   6.16167525e-02,
         6.53511012e-02,   6.72731924e-02,   6.72731924e-02,
         6.54044926e-02,   6.18691146e-02,   5.69847108e-02,
         5.11401251e-02,   4.47476095e-02,   3.81991788e-02,
         3.18326490e-02,   2.59102957e-02,   2.06104625e-02,
         1.60303597e-02,   1.21970128e-02,   9.08288190e-03,
         6.62293472e-03,   4.73066765e-03,   3.31146736e-03}
     }
};    
/*  {
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
    },
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
};  */


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
