# -*- encoding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import numpy as np

from dp_state import StateFactory


def optimize(current_index,
             demands,
             future_utility,
             unit_salvage,
             unit_hold,
             unit_order,
             unit_price,
             unit_disposal,
             discount,
             n_capacity,
             max_hold,
             verbosity=0):

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
    # The State class embedded with parameters
    State = StateFactory(unit_salvage,
                         unit_hold,
                         unit_order,
                         unit_price,
                         unit_disposal,
                         discount)

    holding = sum(current_index)
    at_least_deplete = max(holding - max_hold, 0)

    # Initial value
    maximum = -np.infty

    for n_depletion in range(at_least_deplete, holding + 1):
        for n_order in range(n_capacity):
            objective = []
            for n_demand in demands:
                # Construct new State instance on each call of revenue
                state = State(*current_index)
                # The state is changed within state.revenue() call
                revenue = state.revenue(n_depletion, n_order, n_demand)
                # Just look it up in last utility array
                objective.append(revenue +
                                 discount * future_utility[tuple(state.state)])

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
