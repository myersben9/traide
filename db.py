import yfinance as yf
import requests
import pandas as pd
import numpy as np
import time
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Fetch API keys from environment
API_KEY = os.getenv("ALPHA_API_KEY")
MONGO_DB_URI = os.getenv("MONGO_DB_URI")

# MongoDB Connection
client = MongoClient(MONGO_DB_URI, server_api=ServerApi('1'))

