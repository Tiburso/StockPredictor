from beanie import Document, Indexed


class Stock(Document):
    symbol: str = Indexed(unique=True)
    date: str = Indexed(unique=True)
    open: float
    high: float
    low: float
    close: float
    volume: float

    class Settings:
        name = "stock_data"
        indexes = ["symbol", "date"]
