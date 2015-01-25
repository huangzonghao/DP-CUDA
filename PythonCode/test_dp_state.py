# -*- encoding: utf-8 -*-
from array import array

import dp_state


def test_decode():
    # n_capacity = 10, n_dimension = 2
    state = array('i', [0, 0, 0])  # empty array
    dp_state.py_decode(state, 12, 10, 2)
    assert list(state) == [1, 2, 0]

    # n_capacity = 3, n_dimension = 2
    state = array('i', [0, 0, 0])  # empty array
    dp_state.py_decode(state, 5, 3, 2)
    assert list(state) == [1, 2, 0]

    return 'Good'


def test_encode():
    # n_capacity = 10, n_dimension = 2
    assert dp_state.encode(array('i', [0, 1, 2]), 10, 2) == 12

    # n_capacity = 3, n_dimension = 2
    assert dp_state.encode(array('i', [0, 1, 2]), 3, 2) == 5

    return 'Good'


def test_substract():
    state = array('i', [1, 1, 0, 0])
    assert dp_state.substract(state, 2, 4) == 2
    assert list(state) == [0, 0, 0, 0]

    state = array('i', [1, 1, 0, 0])
    assert dp_state.substract(state, 3, 4) == 2
    assert list(state) == [0, 0, 0, 0]

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

    state = array('i', [2, 3, 4, 0])
    revenue = dp_state.revenue(state, 1, 2, 3,
                               unit_salvage,
                               unit_hold,
                               unit_order,
                               unit_price,
                               unit_disposal,
                               discount,
                               5, 4)
    assert abs(5.55 - revenue < 1e-6)
    assert list(state) == [0, 1, 4, 2]

    state = array('i', [1, 0, 0, 3, 0])
    revenue = dp_state.revenue(state, 1, 2, 3,
                               unit_salvage,
                               unit_hold,
                               unit_order,
                               unit_price,
                               unit_disposal,
                               discount,
                               4, 5)
    assert list(state) == [0, 0, 0, 0, 2]
    assert abs(8.05 - revenue < 1e-6)

    return 'Good'

'''

def test_children():
    State = StateFactory()

    state = State(2, 3, 4)
    assert list(state.children(5)) == [[3, 3, 4]]

    state = State(0, 1, 3)
    assert list(state.children(5)) == [[1, 1, 3], [0, 2, 3]]

    state = State(0, 0, 4)
    assert list(state.children(5)) == [[1, 0, 4], [0, 1, 4]]

    return 'Good'

'''

if __name__ == '__main__':
    print 'Testing state methods'
    print 'Testing decode...', test_decode()
    print 'Testing encode...', test_encode()
    print 'Testing subtract...', test_substract()
    print 'Testing revenue...', test_revenue()
    # print 'Testing children...', test_children()
    print 'All tests passed!'
