import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
import pytz
from movers import TopMovers

# Initialize Dash app
app = dash.Dash(__name__)

# Get Top Movers2
top_movers = TopMovers(40)  # Adjust number of top movers
symbols = top_movers.symbols  # Get list of top movers

# Dashboard Layout
app.layout = html.Div([
    html.H1("Live Top Movers Dashboard", style={'text-align': 'center'}),

    dcc.Interval(id='interval-component', interval=5 * 60 * 1000, n_intervals=0),  # Auto-refresh every 5 minutes

    # Summary Table
    html.Div([
        html.H3("Stock Overview"),
        dash_table.DataTable(
            id='stock-table',
            columns=[
                {"name": "Symbol", "id": "Symbol"},
                {"name": "Short Name", "id": "Short Name"},
                {"name": "Last Price", "id": "Last Price"},
                {"name": "Change %", "id": "Change %"},
                {"name": "Volume", "id": "Volume"},
            ],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'},
        )
    ], style={'padding': '20px'}),

    # Stock Charts
    html.Div(id='charts')
])

# Fetch stock data and create charts
def fetch_stock_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1d', interval='5m', prepost=True)

        # Convert to PST timezone
        data.index = pd.to_datetime(data.index).tz_convert(pytz.timezone('US/Pacific')) 

        if data.empty or len(data) <1:
            return None

        short_name = ticker.info.get("shortName", symbol)  # Get short name

        # Create candlestick chart
        candlestick = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        )])
        candlestick.update_layout(title=f"{short_name} ({symbol}) Price", xaxis_title="Time", yaxis_title="Price")


        volume_chart = go.Figure(data=[go.Bar(
            x=data.index,
            y=data['Volume'],
            marker_color='blue'
        )])
        volume_chart.update_layout(title=f"{short_name} ({symbol}) Volume", xaxis_title="Time", yaxis_title="Volume")

        return html.Div([
            dcc.Graph(figure=candlestick),
            dcc.Graph(figure=volume_chart),
        ])
    except Exception as e:
        return None

# Fetch summary data
def fetch_stock_summary():
    table_data = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            # Get teh
            short_name = info.get("shortName", symbol)
            last_price = round(info.get("currentPrice", 0), 2)
            
            # Calculate change percent
            prev_close = info.get("regularMarketPreviousClose", 0)
            change_percent = round(((last_price - prev_close) / prev_close) * 100, 2)

            volume = info.get("volume", 0)

            table_data.append({
                "Symbol": symbol,
                "Short Name": short_name,
                "Last Price": last_price,
                "Change %": f"{change_percent}%",
                "Volume": f"{volume:,}"  # Format volume with commas
            })
        except Exception as e:
            continue
    return table_data

# Callback to update charts every 5 minutes
@app.callback(
    [Output('charts', 'children'), Output('stock-table', 'data')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    charts = [fetch_stock_data(symbol) for symbol in symbols]
    table_data = fetch_stock_summary()
    return charts, table_data

# Run Dash App
if __name__ == '__main__':
    app.run_server(debug=True)
