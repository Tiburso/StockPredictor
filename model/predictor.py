import torch


class Predictor(torch.nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.fc = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)
