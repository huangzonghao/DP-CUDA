# -*- encoding: utf-8 -*-
from __future__ import print_function

from ConfigParser import ConfigParser

import numpy as np
import scipy.stats

<<<<<<< Updated upstream
from dp_optimization import optimize


def simulation():

    # Read parameters from config file
    config = ConfigParser()
    config.read('./config.ini')

    unit_salvage = config.getfloat('Parameter', 'Salvage')
    unit_hold = config.getfloat('Parameter', 'Hold')
    unit_order = config.getfloat('Parameter', 'Order')
    unit_price = config.getfloat('Parameter', 'Price')
    unit_disposal = config.getfloat('Parameter', 'Disposal')
    discount = config.getfloat('Parameter', 'Discount')

    n_capacity = config.getint('State', 'Capacity')
    n_dimension = config.getint('State', 'Dimension')
    max_hold = config.getint('State', 'MaxHold')

    n_period = config.getint('Simulation', 'Period')
    n_sample = config.getint('Simulation', 'Sample')
    drate = config.getfloat('Simulation', 'DemandRate')

    verbosity = config.getint('Debug', 'Verbosity')

    # Construct demand matrix
    # demand_matrix = scipy.stats.poisson.rvs(drate, size=(n_period, n_sample))
    demand_matrix = np.ones((n_period, n_sample)) * drate
    demand_matrix = demand_matrix.astype(np.int32)

    if verbosity > 0:
        print('Start with demand sample: {}'.format(demand_matrix))

    shape = (n_capacity,) * n_dimension

    # Boundry values of utitility is just depletion of all remaining goods
    # Using a trick of np.where() to return all multi-indices, then sum them up
    future_utility = unit_salvage * \
        np.vstack(np.where(np.ones(shape))).sum(axis=0).reshape(shape)

    # Main loop
    for epoch, demands in enumerate(demand_matrix):
        current_utility = np.empty(shape)
        current_utility[:] = -np.infty

        iterator = np.nditer(future_utility, order='C', flags=['multi_index'])
        for _ in iterator:
            current_index = iterator.multi_index
            optimal_value = optimize(current_index,
                                     demands,
                                     future_utility,
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
            current_utility[current_index] = optimal_value
        current_utility = future_utility

    return current_utility


if __name__ == '__main__':
    simulation()
=======
from state import StateFactory


config = ConfigParser()
config.read('./config.ini')

# State class encapsulated with parameters
unit_salvage = config.getfloat('Parameter', 'Salvage')
unit_hold = config.getfloat('Parameter', 'Hold')
unit_order = config.getfloat('Parameter', 'Order')
unit_price = config.getfloat('Parameter', 'Price')
unit_disposal = config.getfloat('Parameter', 'Disposal')
discount = config.getfloat('Parameter', 'Discount')

n_capacity = config.getint('State', 'Capacity')
n_dimension = config.getint('State', 'Dimension')
max_hold = config.getint('State', 'MaxHold')

n_period = config.getint('Simulation', 'Period')
n_sample = config.getint('Simulation', 'Sample')
drate = config.getfloat('Simulation', 'DemandRate')


State = StateFactory(unit_salvage,
                     unit_hold,
                     unit_order,
                     unit_price,
                     unit_disposal,
                     discount)

demand_matrix = scipy.stats.poisson.rvs(drate, size=(n_period, n_sample))

shape = (n_capacity,) * n_dimension

# Using a trick of np.where() to return all indices
utility = unit_salvage * \
    np.vstack(np.where(np.ones(shape))).sum(axis=0).reshape(shape)

# Main loop
print demand_matrix
for epoch, demands in enumerate(demand_matrix):
    new_utility = np.empty(shape)
    new_utility[:] = -np.infty

    iterator = np.nditer(new_utility, flags=['multi_index'])
    for _ in iterator:
        hold = sum(iterator.multi_index)
        at_least_deplete = max(hold-max_hold, 0)
        for n_depletion in range(at_least_deplete, hold+1):
            for n_order in range(n_capacity):
                objective = []
                for n_demand in demands:
                    state = State(*iterator.multi_index)
                    # The state is changed within state.revenue() call
                    revenue = state.revenue(n_depletion, n_order, n_demand)
                    # Just look it up in last utility array
                    objective.append(revenue +
                                     discount * utility[tuple(state.state)])
                expectation = np.mean(objective)
                # Simply taking the maximum without any complex heuristics
                if not expectation < new_utility[iterator.multi_index]:
                    z, q = n_depletion, n_order
                    new_utility[iterator.multi_index] = expectation
        print iterator.multi_index, (z, q), new_utility[iterator.multi_index]
    utility = new_utility
    print

print 'Demand matrix:', demand_matrix
>>>>>>> Stashed changes
