import yfinance as yf
import pandas as pd
import os
import dotenv
import requests
import logging
import time
from db.db import get_db

dotenv.load_dotenv()

logger = logging.getLogger("API LOGGER")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
