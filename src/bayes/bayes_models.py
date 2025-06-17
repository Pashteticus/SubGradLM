from torch import nn 
from transformers import AutoTokenizer, AutoModel

class BayesModule(nn.Module):
    def __init__(self, model_name, *args, **kwargs):
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.logs = dict()
        self.params = []

class BayesModel(BayesModule):
    def __init__(self, model_name):
        super().__init__()

    def step(self, loss, metric):
        pass # TODO 

class BayesEpoch(BayesModule):
    def __init__(self, model_name):
        super().__init__()

    def step(self, loss, metric):
        pass # TODO 
    
class BayesIter(BayesModule):
    def __init__(self, model_name, iters=50):
        super().__init__()

    def step(self, loss, metric):
        pass # TODO 