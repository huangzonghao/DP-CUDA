# -*- encoding: utf-8 -*-
from dp_state import StateFactory


def test_substract():
    State = StateFactory()

    state = State(1, 1, 0, 0)
    assert state.substract(2) == 2
    assert state == [0, 0, 0, 0]

    state = State(1, 1, 0, 0)
    assert state.substract(3) == 2
    assert state == [0, 0, 0, 0]

    state = State(1, 0, 0, 3)
    assert state.substract(2) == 2
    assert state == [0, 0, 0, 2]

    return 'Good'


def test_revenue():
    unit_salvage = 1.0
    unit_hold = -0.5
    unit_order = -3.0
    unit_price = 5.0
    unit_disposal = -2.0
    discount = 0.95

    State = StateFactory(unit_salvage,
                         unit_hold,
                         unit_order,
                         unit_price,
                         unit_disposal,
                         discount)

    state = State(2, 3, 4)
    assert abs(5.55 - state.revenue(1, 2, 3) < 1e-6)

    state = State(1, 0, 0, 3)
    assert abs(8.05 - state.revenue(1, 2, 3) < 1e-6)

    return 'Good'


def test_children():
    State = StateFactory()

    state = State(2, 3, 4)
    assert list(state.children(5)) == [[3, 3, 4]]

    state = State(0, 1, 3)
    assert list(state.children(5)) == [[1, 1, 3], [0, 2, 3]]

    state = State(0, 0, 4)
    assert list(state.children(5)) == [[1, 0, 4], [0, 1, 4]]

    return 'Good'


if __name__ == '__main__':
    print 'Testing state methods'
    print 'Testing subtract...', test_substract()
    print 'Testing revenue...', test_revenue()
    print 'Testing children...', test_children()
    print 'All tests passed!'
