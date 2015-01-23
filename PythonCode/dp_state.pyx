# -*- encoding: utf-8 -*-


def StateFactory(double unit_salvage=0.,
                 double unit_hold=0.,
                 double unit_order=0.,
                 double unit_price=0.,
                 double unit_disposal=0.,
                 double discount=0.):
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

        def substract(self, int num):
            cdef int i, acc
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

        def deplete(self, int n_depletion):
            return unit_salvage * self.substract(n_depletion)

        def hold(self):
            return unit_hold * sum(self.state)

        def order(self, int n_order):
            self.state.append(n_order)
            return unit_order * n_order

        def sell(self, int n_demand):
            return unit_price * self.substract(n_demand)

        def dispose(self):
            return unit_disposal * self.state.pop(0)

        def revenue(self, int n_depletion, int n_order, int n_demand):
            return (self.deplete(n_depletion) +
                    self.hold() +
                    discount * (self.order(n_order) +
                                self.sell(n_demand) +
                                self.dispose()))

        def children(self, int capacity):
            cdef int i, j, x
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
