import numpy as np
from abc import abstractmethod


class EpsilonGreedyAgent:
    def __init__(self, k, epsilon):
        self.k = k
        self.epsilon = epsilon
        self.estimator_state = np.zeros(k)

    @abstractmethod
    def update(self,value,index):
        pass

    def initialize(self,initial_state=None):
        if initial_state is None:
            initial_state = np.zeros(self.k)
        else:
            assert(len(initial_state)==self.k)
        self.estimator_state = initial_state

    def step(self,bandit):
        greed = np.random.binomial(1, self.epsilon)
        max_index = np.argmax(self.estimator_state)
        if greed == 0:
            index = max_index
        else:
            index = np.random.randint(0,self.k)
        value = bandit(index)

        self.update(value,index)
        return value, index


class AverageEpsilonGreedyAgent(EpsilonGreedyAgent):

    @property
    def n_steps(self):
        assert self._n_steps is not None, "This agent needs to be initialized to access n_steps"
        return self._n_steps

    @n_steps.setter
    def n_steps(self,val):
        self._n_steps = val

    @property
    def i_step(self):
        assert self._i_step is not None, "This agent needs to be initialized to access step"
        return self._i_step

    @i_step.setter
    def i_step(self,val):
        self._i_step = val

    def __init__(self,k, epsilon):
        super(AverageEpsilonGreedyAgent, self).__init__(k,epsilon)
        self.n_steps = None
        self.i_step = None

    def initialize(self,initial_state=None,n_steps=100):
        super(AverageEpsilonGreedyAgent, self).initialize(initial_state=initial_state)
        self.n_steps = n_steps
        self.i_step = 0

    def update(self,value,index):
        assert self.step < self.n_steps, "No more steps available, reset the agent"
        self.estimator_state[index] += (value-self.estimator_state[index])/(self.i_step+1)


class RunningAverageEpsilonGreedyAgent(EpsilonGreedyAgent):

    def __init__(self,k, epsilon, alpha):
        super(RunningAverageEpsilonGreedyAgent, self).__init__(k,epsilon)
        self.alpha = alpha

    def update(self,value,index):
        self.estimator_state[index] += (value - self.estimator_state[index])*self.alpha


