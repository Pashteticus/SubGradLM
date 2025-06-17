from datasets import load_dataset 
from sklearn.model_selection import train_test_split
import pandas as pd 
from torch.utils.data import Dataset, DataLoader 
from sklearn.preprocessing import LabelEncoder as LE
import torch 

class CustomDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_len=512):
        self.X = [tokenizer(text, return_tensors='pt', padding='max_length', max_length=max_len, truncation=True)['input_ids'] for text in X.values]
        self.y = y
        self.transformer = LE().fit(self.y)
        self.y = self.transformer.transform(self.y)
        self.tokenizer = tokenizer 
        self.max_len = max_len

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, ids):
        return self.X[ids], torch.tensor(self.y[ids], dtype=torch.long)


def load_bigbenchhard(tokenizer, batch_size):
    data = load_dataset("maveriq/bigbenchhard", "causal_judgement")['train']
    data = pd.DataFrame(data)
    train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)
    train_dataset = CustomDataset(train_data['input'], train_data['target'], tokenizer) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataset = CustomDataset(test_data['input'], test_data['target'], tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False, drop_last=False)
    return train_loader, test_loader

def load_multiscam(tokenizer, batch_size):
    train_data = load_dataset("BothBosu/multi-agent-scam-conversation")['train']
    train_data = pd.DataFrame(train_data)
    test_data = load_dataset("BothBosu/multi-agent-scam-conversation")['test']
    test_data = pd.DataFrame(test_data)
    train_dataset = CustomDataset(train_data['dialogue'], train_data['personality'], tokenizer) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataset = CustomDataset(test_data['dialogue'], test_data['personality'], tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False, drop_last=False)
    return train_loader, test_loader

def load_llmops(tokenizer, batch_size):
    data = load_dataset("zenml/llmops-database")['train']
    data = pd.DataFrame(data)
    train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)
    train_dataset = CustomDataset(train_data['short_summary'], train_data['industry'], tokenizer) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataset = CustomDataset(test_data['short_summary'], test_data['industry'], tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False, drop_last=False)
    return train_loader, test_loader