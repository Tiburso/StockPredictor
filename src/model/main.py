import torch
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split, KFold

# Metrics
from sklearn.metrics import mean_squared_error


from model.dataset import StockDataset
from model.predictor import LSTMModel


def train_test_dataloader(symbol, batch_size=64):
    dataset = StockDataset(symbol, batch_size)

    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2)

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def k_fold_dataloader(train_dataloader: DataLoader, batch_size=64, n_splits=5):
    kf = KFold(n_splits=n_splits)

    for train_idx, test_idx in kf.split(train_dataloader.dataset):
        train_dataset = Subset(train_dataloader.dataset, train_idx)
        test_dataset = Subset(train_dataloader.dataset, test_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        yield train_loader, test_loader


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
):
    """
    Trains the model for one epoch and evaluates it on the validation set.

    Args:
    - model: The PyTorch model to train.
    - optimizer: The optimizer to use for training.
    - criterion: The loss function to use for training.
    - train_loader: The DataLoader for the training set.
    - validation_loader: The DataLoader for the validation set.
    - device: The device to run the training on (CPU or GPU).

    Returns:
    - None
    """
    # Training loop
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            validation_loss += loss.item()

    validation_loss /= len(validation_loader)
    print(f"Validation Loss: {validation_loss:.4f}")


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model
    model = LSTMModel(input_size=1, hidden_size=32, num_layers=2, output_size=1)
    model.to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Define the loss function
    criterion = torch.nn.MSELoss()

    # Define the data
    symbol = "AAPL"
    batch_size = 64

    # Get the dataloaders
    train_dataloader, test_loader = train_test_dataloader(symbol, batch_size)

    # Implement K-Fold Cross Validation
    for train_loader, validation_loader in k_fold_dataloader(
        train_dataloader, batch_size, n_splits=5
    ):
        # Train the model
        train(model, optimizer, criterion, train_loader, validation_loader, device)

    # Test the model
    test(model, criterion, test_loader)


if __name__ == "__main__":
    main()
