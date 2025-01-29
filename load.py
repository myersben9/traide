import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras import Sequential
import keras

# Load the dataset
df = pd.read_csv('stock_screener_results.csv')

# Feature engineering: Add more features if needed
# Example: Calculate RSI, MACD, etc.

# Define the target variable (e.g., 1 if 5d_percent_change > 10%, else 0)
df['target'] = (df['5d_percent_change'] > 10).astype(int)

# Select features and target
features = ['marketCap', '5d_percent_change', '5d_avg_volume']  # Add more features
X = df[features]
y = df['target']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Define the model
model = Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Generate predictions
predictions = model.predict(X_test)

# Convert X_test to DataFrame and get the original index from df
X_test_df = pd.DataFrame(X_test, columns=features)  # Make sure column names are kept
df_test = df.iloc[X_test_df.index]  # Use the original indices of the test set
df_test.loc[:, 'prediction'] = predictions

# Rank stocks by prediction score
df_test = df_test.sort_values(by='prediction', ascending=False)
print(df_test[['symbol', 'prediction', '5d_percent_change']].head())
best_stock = df_test.iloc[0]
print(f"Best stock to invest in: {best_stock['symbol']} (Prediction Score: {best_stock['prediction']:.2f})")