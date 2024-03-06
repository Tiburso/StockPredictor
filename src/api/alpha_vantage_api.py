import dotenv
import os
import requests
import logging
import time

import pandas as pd

from db import get_db

dotenv.load_dotenv()

API_KEY = os.getenv("ALPHAADVANTAGE_API")

logger = logging.getLogger("API LOGGER")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def get_intraday_data(
    symbol, interval="15min", outputsize="compact", date=None
) -> dict:
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": API_KEY,
    }

    if date is not None:
        params["date"] = date

    response = requests.get(f"https://www.alphavantage.co/query", params=params)

    if response.status_code != 200:
        logger.error(f"Failed to fetch data for {symbol} on {date}")
        raise Exception(f"Failed to fetch data for {symbol} on {date}")

    data = response.json()

    if "Error Message" in data:
        logger.error(f"Failed to fetch data for {symbol} on {date}")
        raise Exception(f"Failed to fetch data for {symbol} on {date}")

    return data[f"Time Series ({interval})"]


def get_historical_intraday_data(symbol, from_date, to_date, interval="15min"):
    # Check date range
    date_range = pd.date_range(from_date, to_date, freq="ME")

    # Create a list of lists with the date ranges of size 5
    date_ranges = [[date_range[i : i + 5]] for i in range(0, len(date_range), 5)]

    # Call the API for each date range with a 1 minute delay
    for date_range in date_ranges:
        for date in date_range:
            try:
                data = get_intraday_data(symbol, interval, "full", date)
            except Exception as e:
                logger.error(e)
                time.sleep(60)
                continue

            yield data
            time.sleep(60)


def convert_to_df(data):
    df = pd.DataFrame(data).T
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    df.columns = [col.split(" ")[1] for col in df.columns]

    return df


def main():
    db = get_db()

    symbol = "AAPL"

    data = get_intraday_data(symbol, "15min", "full")
    df = convert_to_df(data)

    print(df.head())

    # get_historical_intraday_data(symbol, "2021-01", "2022-01")

    # for data in get_historical_intraday_data(symbol, "2021-01", "2022-0"):

    # db.stocks.insert_many(
    #     [
    #         {
    #             "symbol": symbol,
    #             "date": date,
    #             "open": float(data[date]["1. open"]),
    #             "high": float(data[date]["2. high"]),
    #             "low": float(data[date]["3. low"]),
    #             "close": float(data[date]["4. close"]),
    #             "volume": float(data[date]["5. volume"]),
    #         }
    #         for date in data
    #     ]
    # )


if __name__ == "__main__":
    main()
