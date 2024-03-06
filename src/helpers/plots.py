import numpy as np  # Linear algebra

import matplotlib.pyplot as plt  # Visualization
import matplotlib.dates as mdates  # Formatting dates
import seaborn as sns  # Visualization

import pandas as pd


def data_plot(df: pd.DataFrame):
    df_plot = df.copy()

    ncols = 2
    nrows = int(round(df_plot.shape[1] / ncols, 0))

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(14, 7))
    for i, ax in enumerate(fig.axes):
        sns.lineplot(data=df_plot.iloc[:, i], ax=ax)
        ax.tick_params(axis="x", rotation=30, labelsize=10, length=0)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    fig.tight_layout()
    plt.show()


def plot_loss(train_loss: list, valid_loss: list):
    num_epochs = len(train_loss)
    plt.plot(range(num_epochs), train_loss, label="Training Loss")
    plt.plot(range(num_epochs), valid_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_forecasting(
    test_data: pd.DataFrame,
    sequence_to_plot: np.ndarray,
    forecasted_values: np.ndarray,
    combined_index: pd.DatetimeIndex,
    scaler,
    forecast_horizon: int,
):
    # Test data
    plt.plot(
        test_data.index[-100:-forecast_horizon],
        test_data["open"][-100:-forecast_horizon],
        label="test_data",
        color="b",
    )
    # reverse the scaling transformation
    original_cases = scaler.inverse_transform(
        np.expand_dims(sequence_to_plot[-1], axis=0)
    ).flatten()

    # the historical data used as input for forecasting
    plt.plot(
        test_data.index[-forecast_horizon:],
        original_cases,
        label="actual values",
        color="green",
    )

    # Forecasted Values
    # reverse the scaling transformation
    forecasted_cases = scaler.inverse_transform(
        np.expand_dims(forecasted_values, axis=0)
    ).flatten()
    # plotting the forecasted values

    plt.plot(
        combined_index[-2 * forecast_horizon :],
        forecasted_cases,
        label="forecasted values",
        color="red",
    )

    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Time Series Forecasting")
    plt.grid(True)
