from torch import nn 
import torch 
from tqdm.auto import tqdm 


class CustomTrainer():
    def __init__(self, model, optimizer, criterion, metric, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer 
        self.criterion = criterion 
        self.metric = metric
        self.device = device

    def train_epoch(self, loader):
        self.model.train() 
        pbar = tqdm(loader, total=len(loader))
        logs = {
            "result": {
                'loss': [],
                'metric': []
            },
            "logs": {
                'loss': [],
                'metric': []
            }
        }
        total = 0
        for data, target in pbar:
            self.optimizer.zero_grad()
            data = data.reshape(data.shape[0], -1).to(self.device)
            target = target.to(self.device)
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()
            cur_metric = self.metric(target.detach().cpu(), outputs.detach().cpu().argmax(dim=1))
            pbar.set_description(f"Train Loss: {loss.item()} Train metric: {cur_metric}")
            logs['logs']['loss'].append(loss.item()*len(data))
            logs['logs']['metric'].append(cur_metric*len(data))
            total+=len(data)
        logs['result']['loss'] = sum(logs['logs']['loss']) / total 
        logs['result']['metric'] = sum(logs['logs']['metric']) / total
        return logs

    @torch.no_grad()
    def val_epoch(self, loader):
        self.model.eval()
        pbar = tqdm(loader, total=len(loader))
        logs = {
            "result": {
                'loss': [],
                'metric': []
            },
            "logs": {
                'loss': [],
                'metric': []
            }
        }
        total = 0
        for data, target in pbar:
            data = data.reshape(data.shape[0], -1).to(self.device)
            target = target.to(self.device)
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            cur_metric = self.metric(target.detach().cpu(), outputs.detach().cpu().argmax(dim=1))
            pbar.set_description(f"Val Loss: {loss.item()} Val metric: {cur_metric}")
            logs['logs']['loss'].append(loss.item())
            logs['logs']['metric'].append(cur_metric)
            logs['logs']['loss'].append(loss.item()*len(data))
            logs['logs']['metric'].append(cur_metric*len(data))
            total+=len(data)
        logs['result']['loss'] = sum(logs['logs']['loss']) / total 
        logs['result']['metric'] = sum(logs['logs']['metric']) / total
        return logs


    def run_train(self, train_loader, val_loader=None, n_epochs=5):
        pbar = tqdm(list(range(n_epochs)))
        last_log = None
        for epoch in pbar:
            train_logs = self.train_epoch(train_loader)
            last_log=train_logs
            if val_loader is not None:
                val_logs = self.val_epoch(val_loader)
                last_log=val_logs
                pbar.set_description(f"Train Loss: {train_logs['result']['loss']} Train Metric: {train_logs['result']['metric']}\nVal Loss: {val_logs['result']['loss']} Val Metric: {val_logs['result']['metric']}")
            else:
                pbar.set_description(f"Train Loss: {train_logs['result']['loss']} Train Metric: {train_logs['result']['metric']}")
        return last_log