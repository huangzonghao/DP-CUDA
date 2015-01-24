# -*- encoding: utf-8 -*-
from __future__ import division
from __future__ import print_function

from array import array
from cpython cimport array

import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY

cimport cython

cimport dp_state


@cython.boundscheck(False)
@cython.wraparound(False)
def optimize(current_index,
             np.ndarray[int, ndim=1, mode='c'] demands,
             np.ndarray[double, ndim=4, mode='c'] future_utility,
             double unit_salvage,
             double unit_hold,
             double unit_order,
             double unit_price,
             double unit_disposal,
             double discount,
             int n_capacity,
             int n_dimension,
             int max_hold,
             int verbosity=0):

    '''
    Exhaustive search of best n_depletion and n_order.
    Arguments:
        current_index: the state x we are optimizing on
        demands: an array of samples of demand
        future_utility: utility matrix of tomorrow
        verbosity: set larger than 0 to print optimal n_depletion and n_order.
    Return:
        maximum: the maximum utility of current_index
    '''
    cdef np.intp_t i, n_sample = demands.shape[0]
    cdef int n_depletion, n_order
    cdef int z, q
    cdef double revenue, objective

    cdef array.array current_state = array('i', current_index + (0,))
    cdef array.array state = array('i')

    cdef int holding = dp_state.csum(current_state, n_dimension + 1)
    cdef int at_least_deplete = dp_state.cmax(holding - max_hold, 0)

    cdef double maximum = -INFINITY

    for n_depletion in range(at_least_deplete, holding + 1):
        for n_order in range(n_capacity):
            objective = 0.0
            for i in range(n_sample):
                state[:] = current_state
                # The state is changed within state.revenue() call
                revenue = dp_state.revenue(state,
                                           n_depletion,
                                           n_order,
                                           demands[i],
                                           unit_salvage,
                                           unit_hold,
                                           unit_order,
                                           unit_price,
                                           unit_disposal,
                                           discount,
                                           n_capacity,
                                           n_dimension + 1)
                # Just look it up in last utility array
                objective += (revenue +
                              discount * future_utility[tuple(state[1:])])

            # Simply taking the maximum without any complex heuristics
            if objective > maximum:
                z, q = n_depletion, n_order
                maximum = objective

    if verbosity >= 10:
        print('Optimizing {}, result {}, value {}'.format(current_index,
                                                          (z, q), maximum / n_sample))
    return maximum / n_sample
