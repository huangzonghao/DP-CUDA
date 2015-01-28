# -*- encoding: utf-8 -*-
# cython: boundscheck=False, wraparound=False, cdivision=True

from __future__ import division
from __future__ import print_function

from datetime import datetime

import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY

cimport cython
from cython.view cimport array

cimport dp_state


# printf in C is thread safe
cdef extern from "stdio.h":
    int printf(char *, ...) nogil


# Actual task dispatched
cpdef task(int[:] demands,
           double[:] current_utility_ravel,
           double[:] future_utility_ravel,
           double unit_salvage,
           double unit_hold,
           double unit_order,
           double unit_price,
           double unit_disposal,
           double discount,
           int n_capacity,
           int n_dimension,
           int max_hold,
           int job_number,
           int n_job,
           int verbosity=0):
    cdef int index = job_number

    # Allocate memory of state queue ahead
    # Avoid memory allocation inside inner loop
    cdef int[:] transient_state = array(shape=(n_dimension+1,),
                                        itemsize=sizeof(int),
                                        format="i")
    cdef int[:] current_state = array(shape=(n_dimension+1,),
                                      itemsize=sizeof(int),
                                      format="i")
    with nogil:
        # for index in range(job_number, n_capacity**n_dimension, n_jobs):
        while index < n_capacity**n_dimension:
            optimal_value = optimize(index,
                                    current_state,
                                    transient_state,
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
            current_utility_ravel[index] = optimal_value
            index += n_job


cdef double optimize(int encoded_current_state,
                     int[:] current_state,
                     int[:] transient_state,
                     int[:] demands,
                     double[:] future_utility_ravel,
                     double unit_salvage,
                     double unit_hold,
                     double unit_order,
                     double unit_price,
                     double unit_disposal,
                     double discount,
                     int n_capacity,
                     int n_dimension,
                     int max_hold,
                     int verbosity) nogil:

    '''
    Exhaustive search of best n_depletion and n_order.
    Arguments:
        encoded_current_index: the encoded state we are optimizing on
        current_state: buffer of length n_dimension+1 for buffering
        transient_state: same as current_state
        demands: an array of samples of demand
        future_utility_ravel: utility matrix of tomorrow to refer
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


    # initialize current_state
    dp_state.decode(current_state, encoded_current_state,
                    n_capacity, n_dimension)

    # Set limit of n_depletion
    # We are using hand written function in C to speed up
    holding = dp_state.csum(current_state, n_dimension + 1)
    at_least_deplete = dp_state.cmax(holding - max_hold, 0)

    for n_depletion in range(at_least_deplete, holding + 1):
        for n_order in range(n_capacity):
            objective = 0.0
            for i in range(n_sample):
                # Restore initial state
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
                                           n_dimension)
                # The state is changed within dp_state.revenue() call
                encoded_future_state = dp_state.encode(transient_state,
                                                       n_capacity,
                                                       n_dimension)
                objective += (revenue + discount * \
                              future_utility_ravel[encoded_future_state])

            # Simply taking the maximum without any complex heuristics
            if objective > maximum:
                z, q = n_depletion, n_order
                maximum = objective

    if verbosity >= 10:
        printf("State: %d, Result: (%d, %d), Value: %.2f\n",
               encoded_current_state, z, q, maximum / n_sample)

    # Instead of taking mean in each inner loop
    # We take it after optimization
    return maximum / n_sample
