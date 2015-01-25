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
@cython.cdivision(True)
def optimize(int encoded_current_state,
             np.ndarray[int, ndim=1, mode='c'] demands,
             np.ndarray[double, ndim=1, mode='c'] future_utility,
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

    cdef int encoded_future_state

    cdef int holding
    cdef int at_least_deplete

    cdef double maximum = -INFINITY

    cdef array.array transient_state = array('i')
    cdef array.array current_state = array.clone(transient_state,
                                                 n_dimension+1, False)

    # initialize current_state
    dp_state.decode(current_state, encoded_current_state,
                    n_capacity, n_dimension)
    holding = dp_state.csum(current_state, n_dimension + 1)
    at_least_deplete = dp_state.cmax(holding - max_hold, 0)

    for n_depletion in range(at_least_deplete, holding + 1):
        for n_order in range(n_capacity):
            objective = 0.0
            for i in range(n_sample):
                transient_state[:] = current_state
                revenue = dp_state.revenue(transient_state,
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
                # The state is changed within state.revenue() call
                encoded_future_state = dp_state.encode(transient_state,
                                                    n_capacity,
                                                    n_dimension)
                objective += (revenue +
                            discount * future_utility[encoded_future_state])

            # Simply taking the maximum without any complex heuristics
            if objective > maximum:
                z, q = n_depletion, n_order
                maximum = objective

    if verbosity >= 10:
        print('Result {}, value {}'.format((z, q), maximum / n_sample))
    return maximum / n_sample
