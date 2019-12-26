import numpy as np
from numbers import Number


class NormalMultiArmedBandit:
    def __init__(self,means,stds,seed=None):
        assert len(means)==len(stds)
        self.k = len(means)
        self.means = np.array(means)
        self.stds = np.array(stds)

        if seed is not None:
            np.seed(seed)

    def __call__(self,indices):
        return np.random.normal(self.means[indices], self.stds[indices])

    @staticmethod
    def reset_seed(seed):
        np.random.seed(seed)


class StandardBandit(NormalMultiArmedBandit):
    def __init__(self,k=10,seed=None):
        means = np.random.normal(0.,1.,k)
        stds = [1.]*k
        super(StandardBandit, self).__init__(means, stds, seed=seed)
