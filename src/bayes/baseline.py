from torch import nn 

class BayesModule(nn.Module):
    def __init__(self, params):
        self.params = params

    def step(self, X, y):
        return self.params