# -*- encoding: utf-8 -*-
from __future__ import print_function

from datetime import datetime
from ConfigParser import ConfigParser

import numpy as np
import scipy.stats

from joblib import Parallel, delayed

from dp_optimization import task


def simulation():

    # Read parameters from configuration file
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
    # Dimension of utility matrix
    shape = (n_capacity,) * n_dimension

    # Boundary values are just depletion of all remaining goods
    # So let's sum them up
    current_utility = np.empty(shape)
    future_utility = unit_salvage * np.indices(shape).sum(axis=0)

    # Thread pool for parallelization
    dispatch = Parallel(n_jobs=n_jobs, backend='threading')

    # Main loop
    for epoch in xrange(n_period):
        if verbosity > 0:
            print("[{}] Starting epoch {}".format(datetime.now(),
                                                  epoch), end=', ')
            print("with demand: ", ','.join(map(str, demand_matrix[epoch])))

        # Ravel utility matrix (to 1d) for fast access
        # Noted that no copy of actual data is made
        current_utility_ravel = current_utility.ravel()
        future_utility_ravel = future_utility.ravel()

        # We modify the current utility inside each task
        tasks = (delayed(task)(demand_matrix[epoch],
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
                               job_number,
                               n_jobs,
                               verbosity) for job_number in range(n_jobs))
        # Kick them off
        dispatch(tasks)

        # Copy them back and go to next epoch
        future_utility[:] = current_utility

    return future_utility


if __name__ == '__main__':
    simulation()
