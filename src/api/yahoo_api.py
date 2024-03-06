import asyncio
import yfinance as yf
import pandas as pd
import dotenv
import logging

from db.db import init_db, insert_stock_data
from api.api_cache import session
from tickers import TICKERS

from datetime import date

dotenv.load_dotenv()

logger = logging.getLogger("API LOGGER")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


async def main():
    end_date = date.today().strftime(
        "%Y-%m-%d"
    )  # end date for our data retrieval will be current date

    start_date = "1990-01-01"  # Beginning date for our historical data retrieval

    await init_db()

    tickers = yf.Tickers(TICKERS, session=session)
    df: pd.DataFrame = tickers.download(
        start=start_date, end=end_date, group_by="ticker"
    )

    for ticker in TICKERS:
        logger.info(f"Inserting {ticker} into database")
        await insert_stock_data(df[ticker], ticker)


if __name__ == "__main__":
    asyncio.run(main())
