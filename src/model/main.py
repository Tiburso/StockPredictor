import torch
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split, KFold

# Metrics
from sklearn.metrics import mean_squared_error


from model.dataset import StockDataset
from model.predictor import Predictor


def train_test_dataloader(symbol, batch_size=64):
    dataset = StockDataset(symbol, batch_size)

    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2)

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def k_fold_dataloader(symbol, batch_size=64, n_splits=5):
    dataset = StockDataset(symbol, batch_size)

    kf = KFold(n_splits=n_splits)

    dataloaders = []
    for train_idx, test_idx in kf.split(dataset):
        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        dataloaders.append((train_loader, test_loader))

    return dataloaders


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
):
    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loop
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader)}, Test Loss: {test_loss/len(test_loader)}"
        )


def test(model: torch.nn.Module, criterion: torch.nn.Module, test_loader: DataLoader):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    print(f"Test Loss: {test_loss/len(test_loader)}")


def main():
    # Define the model
    model = Predictor()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Define the loss function
    criterion = torch.nn.MSELoss()

    # Define the data
    symbol = "AAPL"
    batch_size = 64

    # Get the dataloaders
    train_loader, test_loader = train_test_dataloader(symbol, batch_size)

    # Implement K-Fold Cross Validation
    dataloaders = k_fold_dataloader(symbol, batch_size, n_splits=5)

    for train_loader, test_loader in dataloaders:
        # Train the model
        train(model, optimizer, criterion, train_loader, test_loader)

    # Test the model
    test(model, criterion, test_loader)


if __name__ == "__main__":
    main()
