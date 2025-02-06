

# import dependencies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')
     

# loading AAPL data into a dataframe

stock = "AAPL"
start = "2015-01-01"
end = "2024-01-01"

df = yf.download(stock, start=start, end=end)
df.isnull().values.any()
df = df.dropna()

df.head()
     

import plotly.graph_objects as go

# making a copy of the data so you don't add unecessary columns (this is just for plotting purposes)
data = df.copy()

data['Lagged Close'] = data['Close'].shift(1)

fig = go.Figure(data=go.Scatter(x=data['Close'], y=data['Lagged Close'], mode='markers'))

fig.update_layout(
    title=f'Lag Plot of {stock} Closing Prices (1-Day Lag)',
    xaxis_title="Today's Close",
    yaxis_title="Previous Day's Close",
    template='plotly_white'
)


data = df.copy()

data['Lagged Close'] = data['Close'].shift(5)

fig = go.Figure(data=go.Scatter(x=data['Close'], y=data['Lagged Close'], mode='markers'))

fig.update_layout(
    title=f'Lag Plot of {stock} Closing Prices (5-Day Lag)',
    xaxis_title="Today's Close",
    yaxis_title="Previous Day's Close",
    template='plotly_white'
)

fig.show()
     
     
