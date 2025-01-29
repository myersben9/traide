import yfinance as yf
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf
from yfinance import EquityQuery
import pandas as pd
import json

def screen_stocks():
    """
    Fetch stocks from Yahoo Finance based on predefined screening criteria.
    """
    try:
        # Define the screening criteria
        q = EquityQuery('and', [
            EquityQuery('gt', ['percentchange', 20]),  # Stocks with >20% change
            EquityQuery('eq', ['region', 'us']),       # US region
            EquityQuery('is-in', ['exchange', 'NMS', 'NYQ', 'NGM', 'ASE', 'PCX', 'YHD', 'NCM']),  # Specific exchanges
        ])
        
        # Fetch the screened stocks
        response = yf.screen(q, sortField='percentchange', sortAsc=True, size=200)
        
        # Return the list of stocks
        if response and 'quotes' in response:
            return response['quotes']
        else:
            print("No stocks found matching the criteria.")
            return []
    except Exception as e:
        print(f"Error fetching stocks: {e}")
        return []

def get_historical_data(symbol, period="30d", interval="1m"):
    """
    Fetch historical data for a stock.
    """
    stock = yf.Ticker(symbol)
    data = stock.history(period=period, interval=interval)
    return data

def calculate_target(data):
    """
    Calculate the target variable based on historical data.
    We will use the next day's percentage change for simplicity.
    """
    return (data['Close'].pct_change()).shift(-1)  # Predict next day's return

def collect_stock_data(symbols):
    """
    Collect both real-time and historical data for each symbol.
    """
    feature_data = []
    
    for symbol in symbols:
        try:
            # Fetch historical data (you can adjust the period as needed)
            historical_data = get_historical_data(symbol, period="30d", interval="1m")
            
            # Get the most recent market data (real-time)
            real_time_data = yf.Ticker(symbol).history(period="1d").iloc[-1]
            
            # Extract relevant features
            features = {
                'symbol': symbol,
                'regularMarketPrice': real_time_data['Close'],
                'regularMarketChangePercent': real_time_data['Close'] / historical_data['Close'][-2] - 1,  # Price change percentage
                'marketCap': real_time_data['Close'] * real_time_data['Volume'],  # Simple market cap estimate
                'regularMarketVolume': real_time_data['Volume'],
                'fiftyTwoWeekHigh': real_time_data['Close'] * 1.1,  # Placeholder
                'fiftyTwoWeekLow': real_time_data['Close'] * 0.9,  # Placeholder
                'regularMarketDayHigh': real_time_data['High'],
                'regularMarketDayLow': real_time_data['Low'],
                'regularMarketPreviousClose': historical_data['Close'][-2],  # Previous day's close
                'averageDailyVolume3Month': historical_data['Volume'].mean(),  # 3-month average volume
            }
            
            # Calculate the target value
            target = calculate_target(historical_data)
            features['target'] = target[-1]  # Next day's return (target for prediction)
            
            feature_data.append(features)
        
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    
    return feature_data

def train_model(data):
    """
    Train a RandomForestClassifier on the collected data.
    """
    df = pd.DataFrame(data)
    
    # Remove rows with missing values
    df = df.dropna()
    
    # Features and target
    X = df.drop(columns=['symbol', 'target'])  # Features
    y = df['target']  # Target
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    return model

def predict_best_stock(model, stock_data):
    """
    Predict which stock has the highest potential for gains using the trained model.
    """
    predictions = []
    for stock in stock_data:
        prediction = model.predict([stock.drop(columns=['symbol', 'target']).values])
        predictions.append((stock['symbol'], prediction[0]))  # Predict the next day's return
    
    # Sort predictions by the highest predicted return
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    return predictions

# Main function to run the pipeline
def run_pipeline():
    # Step 1: Fetch the stock data using the screener
    screener_stocks = screen_stocks()
    
    if screener_stocks:
        # Step 2: Collect both real-time and historical data for the screened stocks
        symbols = [stock['symbol'] for stock in screener_stocks]
        stock_data = collect_stock_data(symbols)
        
        # Step 3: Train the model on the collected data
        model = train_model(stock_data)
        
        # Step 4: Predict which stock has the highest potential for gains
        predictions = predict_best_stock(model, stock_data)
        
        # Print the predictions (top 5 for example)
        print("Predicted Top 5 Stocks with Highest Potential for Gains:")
        for symbol, prediction in predictions[:5]:
            print(f"{symbol}: Predicted Gain: {prediction:.2%}")
        
        # Save results to a file
        output_file = 'stock_predictions.json'
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=4)
        print(f"Predictions saved to {output_file}")
        
    else:
        print("No stocks found from the screener.")

# Run the pipeline
run_pipeline()
