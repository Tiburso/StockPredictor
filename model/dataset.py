"""In this file I will build the PyTorch dataset class to load data from the database in batches to feed the model.
I will also build a function to create the PyTorch DataLoader object to feed the data to the model.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

from db import get_db


class StockDataset(Dataset):
    def __init__(self, symbol, from_date, to_date, batch_size=64):
        self.db = get_db()
        self.symbol = symbol
        self.from_date = from_date
        self.to_date = to_date
        self.batch_size = batch_size

    def __len__(self):
        return len(self.db.stocks.find({"symbol": self.symbol}))

    def __getitem__(self, idx):
        data = self.db.stocks.find({"symbol": self.symbol})[idx]
        return torch.tensor(data["open"]), torch.tensor(data["close"])
