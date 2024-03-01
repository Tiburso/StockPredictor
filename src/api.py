import dotenv
import os
import requests
import logging

import pandas as pd

from db import get_db

dotenv.load_dotenv()

API_KEY = os.getenv("ALPHAADVANTAGE_API")

logger = logging.getLogger("API LOGGER")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def get_stock_data(symbol, outputsize="compact", date=None):
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": "15min",
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

    return data[f"Time Series (15min)"]


def get_historical_data(symbol, from_date, to_date):
    for date in pd.date_range(from_date, to_date, freq="ME"):
        date = date.strftime("%Y-%m")

        logger.info(f"Fetching data for {symbol} on {date}")

        yield get_stock_data(symbol, date=date, outputsize="full")


def main():
    db = get_db()

    symbol = "AAPL"

    for data in get_historical_data(symbol, "2021-01", "2022-01"):
        db.stocks.insert_many(
            [
                {
                    "symbol": symbol,
                    "date": date,
                    "open": float(data[date]["1. open"]),
                    "high": float(data[date]["2. high"]),
                    "low": float(data[date]["3. low"]),
                    "close": float(data[date]["4. close"]),
                    "volume": float(data[date]["5. volume"]),
                }
                for date in data
            ]
        )
        # for date in data:
        #     db.stocks.update_one(
        #         {"symbol": symbol, "date": date},
        #         {
        #             "$set": {
        #                 "symbol": symbol,
        #                 "date": date,
        #                 "open": float(data[date]["1. open"]),
        #                 "high": float(data[date]["2. high"]),
        #                 "low": float(data[date]["3. low"]),
        #                 "close": float(data[date]["4. close"]),
        #                 "volume": float(data[date]["5. volume"]),
        #             }
        #         },
        #         upsert=True,
        #     )


if __name__ == "__main__":
    main()
