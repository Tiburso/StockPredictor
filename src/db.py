import pymongo
import pandas as pd


def get_db(url="localhost", port=27017, db_name="stocks"):
    client = pymongo.MongoClient(url, port)
    db = client[db_name]
    return db


# Insert ticker data into the appropriate collection
def insert_ticker_data(db, symbol, data):
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
