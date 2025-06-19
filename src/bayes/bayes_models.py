from torch import nn 
from transformers import AutoTokenizer, AutoModel
import numpy as np 
from numpy.linalg import inv

def posterior(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
    return mu_s, cov_s

def kernel(X1, X2, l=1.0, sigma_f=1.0):
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

class BayesModule(nn.Module):
    def __init__(self, model_name, cnt_classes, optimizer, param, *args, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.optimizer = optimizer(self.model.parameters(), lr=param)
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
    def __init__(self, model_name, cnt_classes, optimizer, param):
        super().__init__(model_name, cnt_classes, optimizer, param)
        self.history = []
        self.params = [[param]]
        self.optimizer_class = optimizer

    def step(self, loss):
        self.history.append([loss])
        mu, cov = posterior(np.array([[loss/2]]), np.array(self.history), np.array(self.params))
        samples = np.abs(np.random.multivariate_normal(mu.ravel(), cov, 500))
        new_param = min(min(samples)[0], min(self.params))
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=new_param)
        self.params.append([new_param])