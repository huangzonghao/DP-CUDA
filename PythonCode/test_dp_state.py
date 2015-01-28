# -*- encoding: utf-8 -*-
from array import array

import dp_state


def test_decode():
    # encoded = 12, n_capacity = 10, n_dimension = 2 (buffer_size = 3)
    state = array('i', [0, 0, 0])  # empty buffer
    dp_state.py_decode(state, 12, 10, 2)
    assert list(state) == [1, 2, 0]

    # encoded = 5, n_capacity = 3, n_dimension = 2 (buffer_size = 3)
    state = array('i', [0, 0, 0])  # empty buffer
    dp_state.py_decode(state, 5, 3, 2)
    assert list(state) == [1, 2, 0]

    return 'Good'


def test_encode():
    # n_capacity = 10, n_dimension = 2 (buffer_size = 3)
    assert dp_state.encode(array('i', [0, 1, 2]), 10, 2) == 12

    # n_capacity = 3, n_dimension = 2 (buffer_size = 3)
    assert dp_state.encode(array('i', [0, 1, 2]), 3, 2) == 5

    return 'Good'


def test_substract():
    # num = 2, size = 4
    state = array('i', [1, 1, 0, 0])
    assert dp_state.substract(state, 2, 4) == 2
    assert list(state) == [0, 0, 0, 0]

    # num = 3, size = 4
    state = array('i', [1, 1, 0, 0])
    assert dp_state.substract(state, 3, 4) == 2
    assert list(state) == [0, 0, 0, 0]

    # num = 2, size = 4
    state = array('i', [1, 0, 0, 3])
    assert dp_state.substract(state, 2, 4) == 2
    assert list(state) == [0, 0, 0, 2]

    return 'Good'


def test_revenue():
    unit_salvage = 1.0
    unit_hold = -0.5
    unit_order = -3.0
    unit_price = 5.0
    unit_disposal = -2.0
    discount = 0.95

    # Last digit in buffer is always reserved for demand (0)
    state = array('i', [2, 3, 4, 0])
    # n_deplete = 1, n_order = 2, n_demand = 3
    revenue = dp_state.revenue(state, 1, 2, 3,
                               unit_salvage,
                               unit_hold,
                               unit_order,
                               unit_price,
                               unit_disposal,
                               discount,
                               5, 3)  # n_capacity = 5, n_dimension = 3
    assert abs(5.55 - revenue < 1e-6)
    # First digit in result is always 0 after depletion
    assert list(state) == [0, 1, 4, 2]

    state = array('i', [1, 0, 0, 3, 0])
    # n_deplete = 1, n_order = 2, n_demand = 3
    revenue = dp_state.revenue(state, 1, 2, 3,
                               unit_salvage,
                               unit_hold,
                               unit_order,
                               unit_price,
                               unit_disposal,
                               discount,
                               4, 4)  # n_capacity = 4, n_dimension = 4
    assert list(state) == [0, 0, 0, 0, 2]
    assert abs(8.05 - revenue < 1e-6)

    return 'Good'


if __name__ == '__main__':
    print 'Testing state methods'
    print 'Testing decode...', test_decode()
    print 'Testing encode...', test_encode()
    print 'Testing subtract...', test_substract()
    print 'Testing revenue...', test_revenue()
    print 'All tests passed!'
