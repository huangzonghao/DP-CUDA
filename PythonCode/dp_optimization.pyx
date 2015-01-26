# -*- encoding: utf-8 -*-
from __future__ import division
from __future__ import print_function

from datetime import datetime

import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY

cimport cython
from cython.view cimport array

cimport dp_state


cpdef task(encoded_states,
           demands,
           current_utility_ravel,
           future_utility_ravel,
           unit_salvage,
           unit_hold,
           unit_order,
           unit_price,
           unit_disposal,
           discount,
           n_capacity,
           n_dimension,
           max_hold,
           verbosity=0):
    for encoded_state in encoded_states:
        optimal_value = optimize(encoded_state,
                                 demands,
                                 future_utility_ravel,
                                 unit_salvage,
                                 unit_hold,
                                 unit_order,
                                 unit_price,
                                 unit_disposal,
                                 discount,
                                 n_capacity,
                                 n_dimension,
                                 max_hold,
                                 verbosity)
        current_utility_ravel[encoded_state] = optimal_value


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double optimize(int encoded_current_state,
                     int[:] demands,
                     double[:] future_utility,
                     double unit_salvage,
                     double unit_hold,
                     double unit_order,
                     double unit_price,
                     double unit_disposal,
                     double discount,
                     int n_capacity,
                     int n_dimension,
                     int max_hold,
                     int verbosity):

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

    cdef int i, n_sample = demands.shape[0]
    cdef int n_depletion, n_order
    cdef int z, q
    cdef double revenue, objective

    cdef int encoded_future_state

    cdef int holding
    cdef int at_least_deplete

    cdef double maximum = -INFINITY

    cdef int[:] transient_state = array(shape=(n_dimension+1,),
                                        itemsize=sizeof(int),
                                        format="i")
    cdef int[:] current_state = array(shape=(n_dimension+1,),
                                      itemsize=sizeof(int),
                                      format="i")

    with nogil:

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

    if verbosity >= 100:
        state = np.unravel_index(encoded_current_state, (n_capacity,)*n_dimension)
        print("[{}] Optimizing {}".format(datetime.now(),
                                            state), end=', ')
        print('Result {}, value {}'.format((z, q), maximum / n_sample))

    return maximum / n_sample
