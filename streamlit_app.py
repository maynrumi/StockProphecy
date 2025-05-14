import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import pickle
from datetime import date, datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error


FEATURE_COLS = [
    'Open', 'High', 'Low', '20MA', '50MA', 'RSI', 'MACD', 'Signal_Line',
    'MACD_Histogram', 'Middle_Band', 'Upper_Band', 'Lower_Band', 'Close_Lag1',
    'Close_Lag2', 'Close_Lag3', 'Close_Lag5', 'Close_Lag10'
]

# Cache the loading of models and scalers for performance
@st.cache_resource
def load_assets():
    models = {}
    scalers = {}
    stock_list = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', '^DJI', '^FTSE', 'U11.SI', 'D05.SI', 'O39.SI', 'J36.SI', 'Z74.SI']
    
    for stock in stock_list:
        try:
            models[stock] = pickle.load(open(f'/StockProphecy/models/{stock}_model_lr1.pkl', 'rb'))
            scalers[stock] = pickle.load(open(f'/StockProphecy/scalers/{stock}_scaler.pkl', 'rb'))
        except FileNotFoundError:
            st.error(f"Model or scaler file for {stock} not found.")
            return None, None, None
    return models, scalers, stock_list

# Compute technical indicators manually
def compute_features(data):
    # Moving Averages
    data['20MA'] = data['Close'].rolling(window=20).mean()
    data['50MA'] = data['Close'].rolling(window=50).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
    
    
    # Bollinger Bands
    data['Middle_Band'] = data['Close'].rolling(window=20).mean()  
    rolling_std = data['Close'].rolling(window=20).std() 
    print("Type of rolling std:", type(rolling_std))           
    data['Upper_Band'] = data['Middle_Band'] + 2 * rolling_std 
    upper_band_calc = data['Middle_Band'] + 2 * rolling_std
    print("Type of upper_band_calc:", type(upper_band_calc))    
    data['Lower_Band'] = data['Middle_Band'] - 2 * rolling_std


    
    # Lagged Closes
    data['Close_Lag1'] = data['Close'].shift(1)
    data['Close_Lag2'] = data['Close'].shift(2)
    data['Close_Lag3'] = data['Close'].shift(3)
    data['Close_Lag5'] = data['Close'].shift(5)
    data['Close_Lag10'] = data['Close'].shift(10)
    
    return data


# Fetch stock data from yfinance
@st.cache_data
def load_stock_data(stock):
    try:
        start_date = pd.to_datetime('2024-03-01')
        end_date = datetime.today()
        # Download data
        data = yf.download(stock, start=start_date, end=end_date, progress=False)
        if data.empty:
            st.error(f"No data fetched for {stock}.")
            return None
        
        # Check if columns are multi-index and flatten if necessary
        if isinstance(data.columns, pd.MultiIndex):
            # Extract columns for the specific stock (second level of MultiIndex)
            data = data.xs(stock, axis=1, level=1)
            print("After flattening multi-index - Columns:", data.columns.tolist())
        
        # Reset index to make 'Date' a column
        data.reset_index(inplace=True)
        print("After reset_index - Columns:", data.columns.tolist())
        
        # Select desired columns
        data = data[['Date', 'Open', 'High', 'Low', 'Close']]
        print("After column selection - Columns:", data.columns.tolist())
        print("Type of data['Close']:", type(data['Close']))
        
        # Compute features (assuming this function exists)
        data = compute_features(data)
        
        # Filter and clean data
        data = data[data['Date'] >= pd.to_datetime('2024-04-04')]
        data = data.dropna()
        
        return data
    except Exception as e:
        st.error(f"Error fetching data for {stock}: {e}")
        return None

def predict_stock(stock, model, scaler, data, period):
    # Feature columns in training order, excluding 'Close'
    feature_cols = FEATURE_COLS
    
    # Initialize simulated dataset with enough historical data for feature computation
    N = 50  # Sufficient for '50MA' and lagged features
    sim_data = data.tail(N).copy()
    
    # List to store predictions
    predictions = []
    
    # Prediction loop for the specified period
    for _ in range(period):
        # Prepare input data from the last row of sim_data
        input_data_df = sim_data[feature_cols].tail(1)
        input_scaled = scaler.transform(input_data_df)
        
        # Predict the next closing price
        pred = model.predict(input_scaled)[0]
        predictions.append(pred)
        
        # Create a new row with the predicted 'Close' and placeholders
        new_date = sim_data['Date'].iloc[-1] + timedelta(days=1)
        new_row = {
            'Date': new_date,
            'Open': pred,  # Placeholder: assume same as Close
            'High': pred,  # Placeholder
            'Low': pred,   # Placeholder
            'Close': pred
        }
        
        # Append the new row to sim_data
        sim_data = pd.concat([sim_data, pd.DataFrame([new_row])], ignore_index=True)
        
        # Recompute all features on the updated sim_data
        sim_data = compute_features(sim_data)
    
    return predictions



# Updated plot_candlestick function
def plot_candlestick(stock, data, predictions, period, interval):

    # Generate future dates for predictions
    future_dates = pd.date_range(start=data['Date'].iloc[-1] + timedelta(days=1), periods=period)
    
    # Prepare historical data based on selected interval
    if interval == "Day":
        historical = data[['Date', 'Open', 'High', 'Low', 'Close']]
    else:
        freq = 'W' if interval == "Week" else 'M'
        historical = data.resample(freq, on='Date').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).reset_index()
    
    # Prepare future predictions based on selected interval
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': predictions})
    if interval != "Day":
        freq = 'W' if interval == "Week" else 'M'
        future_resampled = future_df.resample(freq, on='Date').last().dropna().reset_index()
    else:
        future_resampled = future_df
    
    # Calculate the Y-axis range
    hist_min = historical['Low'].min()  # Minimum historical low
    hist_max = historical['High'].max()  # Maximum historical high
    pred_min = future_resampled['Predicted_Close'].min()  # Minimum predicted close
    pred_max = future_resampled['Predicted_Close'].max()  # Maximum predicted close
    overall_min = min(hist_min, pred_min)  # Overall minimum
    overall_max = max(hist_max, pred_max)  # Overall maximum
    padding = 0.05 * (overall_max - overall_min)  # Add 5% padding
    yaxis_min = overall_min - padding
    yaxis_max = overall_max + padding
    
    # Create the figure
    fig = go.Figure()
    
    # Add historical candlestick trace
    fig.add_trace(go.Candlestick(
        x=historical['Date'],
        open=historical['Open'],
        high=historical['High'],
        low=historical['Low'],
        close=historical['Close'],
        name='Historical'
    ))
    
    # Add predicted close prices trace
    fig.add_trace(go.Scatter(
        x=future_resampled['Date'],
        y=future_resampled['Predicted_Close'],
        mode='lines+markers',
        name='Predicted Close',
        line=dict(color='red', width=2)
    ))
    
    # Update the layout with the calculated Y-axis range
    fig.update_layout(
        title=f'{stock} Stock Price Prediction ({interval})',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=True,
        template='plotly_white',
        height=600,
        yaxis_range=[yaxis_min, yaxis_max]  # Set the Y-axis range explicitly
    )
    
    return fig

#Technical indicator charts
def plot_interactive_candlestick(stock, data, interval):
    # Determine the frequency for resampling
    if interval == "Week":
        freq = 'W'
    elif interval == "Month":
        freq = 'M'
    else:
        freq = None  # Daily, no resampling
    
    if freq:
        # Resample the data
        resampled = data.resample(freq, on='Date').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna().reset_index()
        # Compute features on resampled data
        resampled = compute_features(resampled)
    else:
        resampled = data.copy()
    
    # Create subplots: 3 rows, shared x-axes with slightly increased spacing
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05,  # Increased from 0.02 for clarity
                        row_heights=[0.5, 0.2, 0.3],
                        subplot_titles=('Price', 'RSI', 'MACD'))
    
    # Candlestick trace
    fig.add_trace(go.Candlestick(
        x=resampled['Date'],
        open=resampled['Open'],
        high=resampled['High'],
        low=resampled['Low'],
        close=resampled['Close'],
        name='Candlestick'
    ), row=1, col=1)
    
    # Moving Averages
    fig.add_trace(go.Scatter(
        x=resampled['Date'],
        y=resampled['20MA'],
        mode='lines',
        name='20MA',
        line=dict(color='blue')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=resampled['Date'],
        y=resampled['50MA'],
        mode='lines',
        name='50MA',
        line=dict(color='orange')
    ), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=resampled['Date'],
        y=resampled['Upper_Band'],
        mode='lines',
        name='Upper Band',
        line=dict(color='gray', dash='dash')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=resampled['Date'],
        y=resampled['Lower_Band'],
        mode='lines',
        name='Lower Band',
        line=dict(color='gray', dash='dash')
    ), row=1, col=1)
    
    # RSI trace
    fig.add_trace(go.Scatter(
        x=resampled['Date'],
        y=resampled['RSI'],
        mode='lines',
        name='RSI',
        line=dict(color='purple')
    ), row=2, col=1)
    
    # MACD traces
    fig.add_trace(go.Scatter(
        x=resampled['Date'],
        y=resampled['MACD'],
        mode='lines',
        name='MACD',
        line=dict(color='green')
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=resampled['Date'],
        y=resampled['Signal_Line'],
        mode='lines',
        name='Signal Line',
        line=dict(color='red')
    ), row=3, col=1)
    
    fig.add_trace(go.Bar(
        x=resampled['Date'],
        y=resampled['MACD_Histogram'],
        name='MACD Histogram',
        marker_color='blue'
    ), row=3, col=1)
    
    # Update layout
    fig.update_layout(
        title=f'{stock} Technical Indicators ({interval})',
        xaxis_title='Date',
        template='plotly_white',
        height=800,
        showlegend=True
    )
    
    # Update y-axes titles
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='RSI', row=2, col=1)
    fig.update_yaxes(title_text='MACD', row=3, col=1)
    
    # Explicitly control range sliders
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)  # Disable for Price
    fig.update_xaxes(rangeslider_visible=False, row=2, col=1)  # Disable for RSI
    fig.update_xaxes(rangeslider_visible=True, row=3, col=1)   # Enable for MACD
    
    return fig


# Streamlit app
def main():
    st.set_page_config(layout="wide")
    
    # Set up input widgets
    st.logo("https://raw.githubusercontent.com/maynrumi/StockProphecy/main/forecast_img.png", 
        icon_image="https://raw.githubusercontent.com/maynrumi/StockProphecy/main/mayan%20log.PNG")
    


    # Load models, scalers, and stock list
    models, scalers, stock_list = load_assets()
    if models is None:
        return
    
    # User interface
    with st.sidebar:
        st.title("Stock Prophecy App")
        selected_stock = st.sidebar.selectbox("Select Stock", stock_list)
        prediction_period_options = [7, 15, 30]
        prediction_period = st.sidebar.selectbox("Prediction Period (days)", prediction_period_options)
        interval = st.sidebar.selectbox("Chart Interval", ["Day", "Week", "Month"])

    
    if selected_stock and prediction_period:
        # Load data
        data = load_stock_data(selected_stock)
        if data is None:
            return
        
        st.subheader("Technical Indicators Chart")
        indicators_fig = plot_interactive_candlestick(selected_stock, data, interval)
        st.plotly_chart(indicators_fig, use_container_width=True)

        # Make predictions
        predictions = predict_stock(selected_stock, models[selected_stock], scalers[selected_stock], data, prediction_period)
        
        # Display chart
        st.subheader("Prediction Chart")
        fig = plot_candlestick(selected_stock, data, predictions, prediction_period, interval)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display raw predictions
        st.subheader("Predicted Prices")
        future_dates = pd.date_range(start=data['Date'].iloc[-1] + timedelta(days=1), periods=prediction_period)
        pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': predictions})
        st.dataframe(pred_df)

if __name__ == "__main__":
    main()
