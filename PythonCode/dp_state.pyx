# -*- encoding: utf-8 -*-
from cpython cimport array


cdef int cmin(int a, int b):
    return a if a < b else b


cdef int cmax(int a, int b):
    return a if a > b else b


cdef int csum(int[:] state, int n_dimension):
    cdef int i
    cdef int acc = 0
    for i in range(n_dimension):
        acc += state[i]
    return acc


cpdef int substract(int[:] state, int num, int n_dimension):
    cdef int i
    cdef int acc = 0
    for i in range(n_dimension):
        if num <= state[i]:
            acc += num
            state[i] -= num
            break
        else:
            num -= state[i]
            acc += state[i]
            state[i] = 0
    return acc


cdef double deplete(int[:] state, int n_depletion, double unit_salvage, int n_dimension):
    return unit_salvage * substract(state, n_depletion, n_dimension)


cdef double hold(int[:] state, double unit_hold, int n_dimension):
    return unit_hold * csum(state, n_dimension)


cdef double order(int[:] state, int n_order, double unit_order, int n_dimension):
    state[n_dimension-1] = n_order
    return unit_order * n_order


cdef double sell(int[:] state, int n_demand, double unit_price, int n_dimension):
    return unit_price * substract(state, n_demand, n_dimension)


cdef double dispose(int[:] state, double unit_disposal):
    cdef int disposal = state[0]
    state[0] = 0
    return unit_disposal * disposal


cpdef double revenue(int[:] state,
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
                     int n_dimension):
    cdef double depletion = deplete(state, n_depletion, unit_salvage, n_dimension);
    cdef double holding = hold(state, unit_hold, n_dimension)
    cdef double ordering = order(state, n_order, unit_order, n_dimension)
    cdef double sales = sell(state, n_demand, unit_price, n_dimension)
    cdef double disposal = dispose(state, unit_disposal)

    return depletion + holding + discount * (ordering + sales + disposal)
"""
def StateFactory(double unit_salvage=0.,
                 double unit_hold=0.,
                 double unit_order=0.,
                 double unit_price=0.,
                 double unit_disposal=0.,
                 double discount=0.):
    '''
    State = StateFactory(config)
    state = State(4, 3, 2, 1)
    '''
    class State(object):
        '''
        state = [4, 3, 2, 1]
        n_depletion = 3
        n_order = 2
        n_demand = 0

        1st evening: deplete -> hold
        2nd day: order -> sales -> disposal

        [4, 3, 2, 1] -(deplete 3)-> [1, 3, 2, 1] (holding 7)
        [1, 3, 2, 1] -(order 1)-> [1, 3, 2, 1, 1] (sales 0)
        [1, 2, 2, 1, 1] -(disposal 1)-> [2, 2, 1, 1]
        '''
        def __init__(self, *args):
            self.state = list(args)

        def __iter__(self):
            return iter(self.state)

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                return self.state == other.state
            else:
                return list(self.state) == list(other)

        def __repr__(self):
            return repr(self.state)

        def __hash__(self):
            return hash(tuple(self.state))

        def __reduce__(self):
            return (self.__class__, tuple(self.state))

        def copy(self):
            return self.__class__(*self.state)

    return State
"""
