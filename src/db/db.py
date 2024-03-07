from pandas import DataFrame

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


async def get_stocks(
    symbol: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    limit: int | None = None,
):
    search_criteria = {}

    if symbol:
        search_criteria["symbol"] = symbol

    if from_date:
        search_criteria["date"] = {"$gte": from_date}

    if to_date:
        search_criteria["date"] = {"$lte": to_date}

    stocks = await Stock.find(search_criteria, limit=limit).to_list()

    df = DataFrame(
        [stock.model_dump(exclude=["id", "date"]) for stock in stocks],
        index=[stock.date for stock in stocks],
    )

    # Set volume as integer
    df["volume"] = df["volume"].astype(int)

    # Set index as datetime index
    df.index = df.index.astype("datetime64[ns]")

    return df
