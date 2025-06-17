from torch import nn 
from transformers import AutoTokenizer, AutoModel

class BayesModule(nn.Module):
    def __init__(self, model_name, cnt_classes, *args, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.logs = dict()
        self.params = []
        for param in self.model.parameters():
            param.requires_grad = False

        self.cls = nn.Sequential(
            nn.Linear(768, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, cnt_classes)
        )

    def forward(self, x):
        x = self.model(x)['pooler_output']
        return self.cls(x)

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