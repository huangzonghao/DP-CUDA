# -*- encoding: utf-8 -*-
from __future__ import print_function

from datetime import datetime
from ConfigParser import ConfigParser

import numpy as np
import scipy.stats

from joblib import Parallel, delayed

from dp_optimization import task


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
    n_jobs = config.getint('Debug', 'NumberJobs')

    # Construct demand matrix
    demand_matrix = scipy.stats.poisson.rvs(drate, size=(n_period,
                                                         n_sample)
                                           ).astype(np.int32)

    shape = (n_capacity,) * n_dimension

    # Boundry values of utitility is just depletion of all remaining goods
    current_utility = np.empty(shape)
    future_utility = unit_salvage * np.indices(shape).sum(axis=0)

    dispatch = Parallel(n_jobs=n_jobs, backend='threading', verbose=verbosity)


    # Main loop
    for epoch in xrange(n_period):
        if verbosity > 0:
            print(("[{}] Starting epoch {} " +
                  "with demand {}").format(datetime.now(),
                                           epoch,
                                           demand_matrix[epoch]))

        current_utility_ravel = current_utility.ravel()
        future_utility_ravel = future_utility.ravel()

        dispatch(delayed(task)(demand_matrix[epoch],
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
                               index,
                               n_jobs,
                               verbosity) for index in range(n_jobs))

        future_utility[:] = current_utility

    return future_utility


if __name__ == '__main__':
    simulation()
