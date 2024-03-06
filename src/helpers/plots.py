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


def plot_loss(num_epochs: int, train_loss: list, valid_loss: list):
    plt.plot(range(num_epochs), train_loss, label="Training Loss")
    plt.plot(range(num_epochs), valid_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_forecasting(df: pd.DataFrame, forecast: pd.DataFrame, title: str):
    # Set the size of the plot
    plt.figure(figsize=(14, 7))

    plt.plot(df.index, df, label="Observed", color="black")
    plt.plot(forecast.index, forecast["mean"], label="Forecast", color="red")
    plt.fill_between(
        forecast.index,
        forecast["mean_ci_lower"],
        forecast["mean_ci_upper"],
        color="red",
        alpha=0.2,
    )

    plt.title(title)
    plt.legend()
    plt.show()
