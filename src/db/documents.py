from beanie import Document, Indexed


class Stock(Document):
    symbol: str
    date: str = Indexed()
    open: float
    high: float
    low: float
    close: float
    volume: float
