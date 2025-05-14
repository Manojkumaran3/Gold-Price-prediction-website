import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from django.conf import settings

class GoldPriceModel:
    """Core gold price prediction model combining N-BEATS and Prophet"""
    
    def __init__(self):
        self.data = self._load_data()
        self.scaler = MinMaxScaler()
        self.nbeats = self._init_nbeats()
        self.prophet = None
        
    def _load_data(self):
        """Load and preprocess the gold price data"""
        file_path = os.path.join(settings.BASE_DIR, 'gold_app', 'data', 'gold_price.csv')
        df = pd.read_csv(file_path)
        
        # Clean and prepare data
        df = df[['date', 'price']].dropna()
        df['date'] = pd.to_datetime(df['date'])
        df.columns = ['ds', 'y']
        return df.sort_values('ds')
    
    def _init_nbeats(self):
        """Initialize N-BEATS model architecture"""
        class NBeatsBlock(nn.Module):
            def __init__(self, input_size, theta_size):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, theta_size)
                )
                self.backcast = nn.Linear(theta_size, input_size)
                self.forecast = nn.Linear(theta_size, 1)
                
            def forward(self, x):
                x = x.view(-1, self.layers[0].in_features)  # Ensure correct shape
                theta = self.layers(x)
                return self.backcast(theta), self.forecast(theta)
        
        class NBeats(nn.Module):
            def __init__(self, input_size=30):
                super().__init__()
                self.input_size = input_size
                self.blocks = nn.ModuleList([
                    NBeatsBlock(input_size, input_size + 1) for _ in range(3)
                ])
                
            def forward(self, x):
                # Ensure correct input shape
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                elif x.dim() == 2 and x.shape[0] != 1:
                    x = x.unsqueeze(0)
                
                forecast = torch.zeros(x.shape[0], 1)
                for block in self.blocks:
                    backcast, block_forecast = block(x)
                    x = x - backcast
                    forecast = forecast + block_forecast
                return forecast
        
        return NBeats()
    
    def train_models(self):
        """Train both prediction models"""
        # Prepare N-BEATS data
        window_size = 30
        scaled_data = self.scaler.fit_transform(self.data['y'].values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(len(scaled_data) - window_size):
            X.append(scaled_data[i:i+window_size].flatten())  # Flatten the window
            y.append(scaled_data[i+window_size])
        
        X_train = torch.FloatTensor(np.array(X))
        y_train = torch.FloatTensor(np.array(y))
        
        # Train N-BEATS
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.nbeats.parameters(), lr=0.001)
        
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self.nbeats(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        
        # Train Prophet
        self.prophet = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        self.prophet.fit(self.data)
    
    def predict(self, target_date):
        """Make prediction for a specific date"""
        target_date = pd.to_datetime(target_date)
        last_date = self.data['ds'].max()
        days_ahead = (target_date - last_date).days
        
        if days_ahead <= 0:
            return None
        
        # Prophet prediction
        future = self.prophet.make_future_dataframe(periods=days_ahead)
        prophet_forecast = self.prophet.predict(future)
        prophet_pred = prophet_forecast.iloc[-1]['yhat']
        
        # N-BEATS prediction
        last_window = self.scaler.transform(
            self.data['y'].values[-30:].reshape(-1, 1)
        ).flatten()  # Flatten the window
        
        predictions = []
        current_window = last_window.copy()
        
        for _ in range(days_ahead):
            with torch.no_grad():
                input_tensor = torch.FloatTensor(current_window).unsqueeze(0)
                pred = self.nbeats(input_tensor).item()
                predictions.append(pred)
                # Update window with new prediction
                current_window = np.append(current_window[1:], pred)
        
        nbeats_pred = self.scaler.inverse_transform(
            np.array(predictions[-1]).reshape(-1, 1)
        )[0][0]
        
        # Ensemble prediction
        return round((prophet_pred + nbeats_pred) / 2, 2)