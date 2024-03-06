from beanie import Document, Indexed


class Stock(Document):
    symbol: Indexed(str)
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
