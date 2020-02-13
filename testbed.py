import numpy as np

class Testbed:

    def __init__(self, num_arms, mean=0, var=1):

        self.num_arms = num_arms

        self.mean = mean
        self.var = var

        self.A = np.zeros(num_arms)     # Array to store action values
        self.optim = 0                  # Store optimal value for greedy

        self.reset()

    def reset(self):
        self.A = np.random.normal(self.mean, self.var, self.num_arms)
        self.optim = np.argmax(self.A)


