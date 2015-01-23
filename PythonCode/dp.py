# -*- encoding: utf-8 -*-
from __future__ import print_function

from ConfigParser import ConfigParser

import numpy as np
import scipy.stats

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
