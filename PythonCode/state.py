# -*- encoding: utf-8 -*-


def StateFactory(unit_salvage=0,
                 unit_hold=0,
                 unit_order=0,
                 unit_price=0,
                 unit_disposal=0,
                 discount=0):
    '''
    State = StateFactory(config)
    state = State(4, 3, 2, 1)
    '''
    class State(object):
        '''
        state = [4, 3, 2, 1]
        n_depletion = 3
        n_order = 2
        n_demand = 0

        1st evening: deplete -> hold
        2nd day: order -> sales -> disposal

        [4, 3, 2, 1] -(deplete 3)-> [1, 3, 2, 1] (holding 7)
        [1, 3, 2, 1] -(order 1)-> [1, 3, 2, 1, 1] (sales 0)
        [1, 2, 2, 1, 1] -(disposal 1)-> [2, 2, 1, 1]
        '''
        def __init__(self, *args):
            self.state = list(args)

        def __iter__(self):
            return iter(self.state)

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                return self.state == other.state
            else:
                return self.state == list(other)

        def __repr__(self):
            return repr(self.state)

        def __hash__(self):
            return hash(tuple(self.state))

        def __reduce__(self):
            return (self.__class__, tuple(self.state))

        def substract(self, num=0):
            acc = 0
            for i in range(len(self.state)):
                if num <= self.state[i]:
                    acc += num
                    self.state[i] -= num
                    break
                else:
                    num -= self.state[i]
                    acc += self.state[i]
                    self.state[i] = 0
            return acc

        def deplete(self, n_depletion=0):
            return unit_salvage * self.substract(n_depletion)

        def hold(self):
            return unit_hold * sum(self.state)

        def order(self, n_order=0):
            self.state.append(n_order)
            return unit_order * n_order

        def sell(self, n_demand=0):
            return unit_price * self.substract(n_demand)

        def dispose(self):
            return unit_disposal * self.state.pop(0)

        def revenue(self, n_depletion, n_order, n_demand):
            return (self.deplete(n_depletion) +
                    self.hold() +
                    discount * (self.order(n_order) +
                                self.sell(n_demand) +
                                self.dispose()))

        def children(self, capacity):
            for i, x in enumerate(self.state):
                if x != 0:
                    break
            for j, x in enumerate(self.state[:i+1]):
                new_state = self.state[:]
                assert x < capacity
                if x == capacity-1:
                    pass
                else:
                    new_state[j] += 1
                    yield State(*new_state)

        def copy(self):
            return self.__class__(*self.state)

    return State


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
