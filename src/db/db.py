import asyncio

from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from db.documents import Stock


async def init_db(url="localhost", port=27017, db_name="stocks"):
    url = f"mongodb://{url}:{port}"
    client = AsyncIOMotorClient(url)

    await init_beanie(database=client.get_database(db_name), document_models=[Stock])


# Insert ticker data into the appropriate collection
async def insert_stock_data(symbol, date, open, high, low, close, volume):
    stock = Stock(
        symbol=symbol,
        date=date,
        open=open,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )

    await stock.insert()


async def get_stocks(from_date, to_date):
    return await Stock.find(Stock.date >= from_date, Stock.date <= to_date).to_list()
