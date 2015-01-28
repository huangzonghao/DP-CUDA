# -*- encoding: utf-8 -*-
# cython: boundscheck=False, wraparound=False, cdivision=True

cimport cython
from cpython cimport array


cdef inline int cmin(int a, int b) nogil:
    return a if a < b else b


cdef inline int cmax(int a, int b) nogil:
    return a if a > b else b


cdef inline int csum(int[:] state, int size) nogil:
    cdef int i
    cdef int acc = 0
    for i in range(size):
        acc += state[i]
    return acc


cdef void decode(int[:] state,
                 int encoded_state,
                 int n_capacity,
                 int n_dimension) nogil:
    cdef int i = 0
    for i in range(n_dimension-1, -1, -1):
        state[i] = encoded_state % n_capacity
        encoded_state /= n_capacity
    # Set last digit to 0, before ordering
    state[n_dimension] = 0


def py_decode(int[:] state,
              int encoded_state,
              int n_capacity,
              int n_dimension):
    decode(state, encoded_state, n_capacity, n_dimension)


cpdef int encode(int[:] state, int n_capacity, int n_dimension) nogil:
    cdef int i = 0
    cdef int acc = 0
    # Don't count on 1st digit (which is 0 any way after depletion)
    for i in range(1, n_dimension+1):
        acc *= n_capacity
        acc += state[i]
    return acc


cpdef int substract(int[:] state, int num, int size) nogil:
    cdef int i
    cdef int acc = 0
    for i in range(size):
        if num <= state[i]:
            acc += num
            state[i] -= num
            break
        else:
            num -= state[i]
            acc += state[i]
            state[i] = 0
    return acc


cdef inline double deplete(int[:] state, int n_depletion,
                    double unit_salvage, int n_dimension) nogil:
    return unit_salvage * substract(state, n_depletion, n_dimension)


cdef inline double hold(int[:] state, double unit_hold, int n_dimension) nogil:
    return unit_hold * csum(state, n_dimension)


cdef inline double order(int[:] state, int n_order,
                  double unit_order, int n_dimension) nogil:
    state[n_dimension] = n_order
    return unit_order * n_order


cdef inline double sell(int[:] state, int n_demand,
                 double unit_price, int n_dimension) nogil:
    return unit_price * substract(state, n_demand, n_dimension+1)


cdef inline double dispose(int[:] state, double unit_disposal) nogil:
    cdef int disposal = state[0]
    state[0] = 0
    return unit_disposal * disposal


cpdef inline double revenue(int[:] state,
                            int n_depletion,
                            int n_order,
                            int n_demand,
                            double unit_salvage,
                            double unit_hold,
                            double unit_order,
                            double unit_price,
                            double unit_disposal,
                            double discount,
                            int n_capacity,
                            int n_dimension) nogil:
    cdef double depletion = deplete(state, n_depletion, unit_salvage, n_dimension)
    cdef double holding = hold(state, unit_hold, n_dimension)
    cdef double ordering = order(state, n_order, unit_order, n_dimension)
    cdef double sales = sell(state, n_demand, unit_price, n_dimension)
    cdef double disposal = dispose(state, unit_disposal)

    return depletion + holding + discount * (ordering + sales + disposal)
