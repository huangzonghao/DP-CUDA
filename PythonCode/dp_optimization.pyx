# -*- encoding: utf-8 -*-
from __future__ import division
from __future__ import print_function

from array import array
from cpython cimport array

import numpy as np

import dp_state


def optimize(current_index,
             demands,
             future_utility,
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
    holding = sum(current_index)
    at_least_deplete = max(holding - max_hold, 0)

    # Initial value
    maximum = -np.infty

    for n_depletion in range(at_least_deplete, holding + 1):
        for n_order in range(n_capacity):
            objective = []
            for n_demand in demands:
                # Construct new State instance on each call of revenue
                state = array('i', current_index + (0,))
                # The state is changed within state.revenue() call
                revenue = dp_state.revenue(state,
                                           n_depletion,
                                           n_order,
                                           n_demand,
                                           unit_salvage,
                                           unit_hold,
                                           unit_order,
                                           unit_price,
                                           unit_disposal,
                                           discount,
                                           n_capacity,
                                           n_dimension+1)
                # Just look it up in last utility array
                objective.append(revenue +
                                 discount * future_utility[tuple(state[1:])])

            # Taking empirical expectation
            expectation = sum(objective) / len(objective)

            # Simply taking the maximum without any complex heuristics
            if expectation > maximum:
                z, q = n_depletion, n_order
                maximum = expectation

    if verbosity > 0:
        print('Optimizing {}, result {}, value {}'.format(current_index,
                                                          (z, q), maximum))
    return maximum
