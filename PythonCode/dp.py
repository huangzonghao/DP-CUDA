# -*- encoding: utf-8 -*-
from __future__ import print_function

import datetime
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
    demand_matrix = scipy.stats.poisson.rvs(drate, size=(n_period,
                                                         n_sample)).astype(np.int32)

    shape = (n_capacity,) * n_dimension

    # Boundry values of utitility is just depletion of all remaining goods
    future_utility = unit_salvage * np.indices(shape).sum(axis=0)

    # Main loop
    for epoch in xrange(n_period):
        if verbosity > 0:
            print("[{}] Starting epoch {}".format(datetime.datetime.now(), epoch))

        current_utility = np.empty(shape)
        for current_index in np.ndindex(*shape):
            if verbosity > 0 and np.ravel_multi_index(current_index, shape) % 100 == 0:
                print("[{}] Optimizing {}".format(datetime.datetime.now(), current_index))

            optimal_value = optimize(current_index,
                                     demand_matrix[epoch],
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

        future_utility[:] = current_utility

    return future_utility


if __name__ == '__main__':
    simulation()
