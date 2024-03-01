import torch
from torch.utils.data import Dataset

from db import get_db


class StockDataset(Dataset):
    def __init__(self, symbol, batch_size=64):
        self.db = get_db()
        self.symbol = symbol
        self.batch_size = batch_size

    def __len__(self):
        return self.db.stocks.count_documents({"symbol": self.symbol})

    def __getitem__(self, idx):
        data = self.db.stocks.find_one({"symbol": self.symbol}, skip=idx)

        data = torch.tensor(data["close"]).view(1, -1).float()

        return data, data
