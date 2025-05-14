import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import render
from django.conf import settings
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import logging
from functools import wraps
import time
from django.core.cache import cache

logger = logging.getLogger(__name__)

# Configuration
YFINANCE_TIMEOUT = getattr(settings, 'YFINANCE_TIMEOUT', 15)
CACHE_TIMEOUT = getattr(settings, 'CACHE_TIMEOUT', 300)  # 5 minutes

def handle_prediction_errors(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:  # Changed from yf.YFinanceError
            logger.error(f"Market data error: {str(e)}", exc_info=True)
            return JsonResponse({
                'error': 'Market data service is currently unavailable',
                'code': 'market_data_failure'
            }, status=503)
        except (ValueError, TypeError) as e:
            logger.error(f"Validation error: {str(e)}", exc_info=True)
            return JsonResponse({
                'error': 'Invalid request parameters',
                'code': 'invalid_parameters'
            }, status=400)
        except RuntimeError as e:
            logger.error(f"Model runtime error: {str(e)}", exc_info=True)
            return JsonResponse({
                'error': 'Prediction service is currently unavailable',
                'code': 'model_failure'
            }, status=503)
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            return JsonResponse({
                'error': 'An unexpected error occurred',
                'code': 'server_error'
            }, status=500)
    return wrapper

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, theta_size, basis_function, layers, layer_size):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_size, layer_size)] + 
                                  [nn.Linear(layer_size, layer_size) for _ in range(layers-1)])
        self.basis_function = basis_function
        self.theta_layer = nn.Linear(layer_size, theta_size)
        
    def forward(self, x):
        block_input = x
        for layer in self.layers:
            block_input = torch.relu(layer(block_input))
        theta = self.theta_layer(block_input)
        return self.basis_function(theta)

class NBeats(nn.Module):
    def __init__(self, input_size=30, output_size=1, blocks=3, layers=4, layer_size=128):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.blocks = nn.ModuleList([
            NBeatsBlock(
                input_size=input_size,
                theta_size=2*(input_size + output_size),
                basis_function=self.trend_basis,
                layers=layers,
                layer_size=layer_size
            ) for _ in range(blocks)
        ])
        
        self.backcast_linear = nn.Linear(2*(input_size + output_size), input_size)
        self.forecast_linear = nn.Linear(2*(input_size + output_size), output_size)
        
    def trend_basis(self, theta):
        backcast = self.backcast_linear(theta)
        forecast = self.forecast_linear(theta)
        return backcast, forecast
        
    def forward(self, x):
        x = x.view(-1, self.input_size)
        residuals = x.flip(dims=(1,))
        forecast = torch.zeros(x.shape[0], self.output_size)
        
        for block in self.blocks:
            block_backcast, block_forecast = block(residuals)
            residuals = residuals - block_backcast
            forecast = forecast + block_forecast
            
        return forecast

def prepare_nbeats_data(data, window_size=30, horizon=1):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(scaled_data) - window_size - horizon + 1):
        X.append(scaled_data[i:i+window_size, 0])
        y.append(scaled_data[i+window_size:i+window_size+horizon, 0])
        
    return np.array(X), np.array(y), scaler

def train_nbeats(X_train, y_train, epochs=50, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NBeats(input_size=X_train.shape[1], output_size=y_train.shape[1])
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.FloatTensor(y_train).to(device)
    
    # Early stopping
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        # Early stopping check
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    return model

def get_yfinance_data(ticker, period="1mo"):
    cache_key = f"yfinance_{ticker}_{period}"
    cached_data = cache.get(cache_key)
    
    if cached_data:
        return cached_data
    
    try:
        # Updated yfinance usage without timeout in constructor
        data = yf.Ticker(ticker).history(
            period=period
        )
        cache.set(cache_key, data, CACHE_TIMEOUT)
        return data
    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker}: {str(e)}")
        raise

@handle_prediction_errors
def get_current_market_data():
    start_time = time.time()
    
    try:
        # Get gold data with fallback
        gold_data = get_yfinance_data("GC=F", "1mo")
        if gold_data.empty:
            raise Exception("No data returned for gold prices")
            
        current_gold = gold_data['Close'].iloc[-1]
        prev_day_gold = gold_data['Close'].iloc[-2]
        prev_week_gold = gold_data['Close'].iloc[-6] if len(gold_data) > 6 else current_gold
        prev_month_gold = gold_data['Close'].iloc[0]

        # Get other indicators with fallback
        def safe_yf(ticker):
            try:
                data = get_yfinance_data(ticker, "1d")['Close']
                return data.iloc[-1] if not data.empty else None
            except:
                return None

        oil = safe_yf("CL=F")
        usd_inr = safe_yf("INR=X")
        sp500 = safe_yf("^GSPC")
        treasury = safe_yf("^TNX")
        treasury = treasury / 100 if treasury else None

        result = {
            'gold_price': current_gold,
            'oil_price': oil if oil is not None else None,
            'usd_inr': usd_inr if usd_inr is not None else None,
            'sp500': sp500 if sp500 is not None else None,
            'treasury_yield': treasury if treasury is not None else None,
            'daily_change': current_gold - prev_day_gold,
            'weekly_change': current_gold - prev_week_gold,
            'monthly_change': current_gold - prev_month_gold,
            'daily_pct': ((current_gold - prev_day_gold) / prev_day_gold) * 100,
            'weekly_pct': ((current_gold - prev_week_gold) / prev_week_gold) * 100,
            'monthly_pct': ((current_gold - prev_month_gold) / prev_month_gold) * 100,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Market data fetched in {time.time() - start_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Error in get_current_market_data: {str(e)}")
        raise

@handle_prediction_errors
@require_http_methods(["GET"])
def market_data(request):
    data = get_current_market_data()
    return JsonResponse(data)

@handle_prediction_errors
@csrf_exempt
@require_http_methods(["POST"])
def predict_price(request):
    start_time = time.time()
    data = json.loads(request.body)
    
    # Validate prediction date
    try:
        prediction_date = datetime.strptime(data['prediction_date'], '%Y-%m-%d').date()
        today = datetime.now().date()
        
        if prediction_date <= today:
            raise ValueError("Prediction date must be in the future")
            
        horizon = (prediction_date - today).days
        if horizon > 365:
            raise ValueError("Prediction date cannot be more than 1 year in the future")
    except ValueError as e:
        logger.error(f"Invalid prediction date: {str(e)}")
        raise

    # Get historical data with fallback
    try:
        hist_data = get_yfinance_data("GC=F", "5y")
        if hist_data.empty:
            raise Exception("No historical data returned")
    except Exception as e:
        logger.error(f"Failed to fetch historical data: {str(e)}")
        raise

    # Prophet prediction with fallback
    prophet_pred = None
    try:
        prophet_df = hist_data.reset_index()[['Date', 'Close']]
        prophet_df.columns = ['ds', 'y']
        
        prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        prophet_model.fit(prophet_df)
        future = prophet_model.make_future_dataframe(periods=365)
        forecast = prophet_model.predict(future)
        prophet_pred = forecast[forecast['ds'].dt.date == prediction_date]['yhat'].values[0]
    except Exception as e:
        logger.error(f"Prophet prediction failed, using fallback: {str(e)}")
        prophet_pred = hist_data['Close'].iloc[-1]

    # N-BEATS prediction with fallback
    nbeats_pred = None
    try:
        window_size = min(30, len(hist_data))
        close_prices = hist_data['Close']
        
        if len(close_prices) < window_size + horizon:
            raise RuntimeError("Insufficient data for N-BEATS prediction")
        
        X, y, scaler = prepare_nbeats_data(close_prices, window_size=window_size, horizon=horizon)
        
        if len(X) == 0 or len(y) == 0:
            raise RuntimeError("Insufficient data after preprocessing")
        
        nbeats_model = train_nbeats(X, y, epochs=50)
        
        recent_data = close_prices[-window_size:].values
        recent_data_scaled = scaler.transform(recent_data.reshape(-1, 1))
        input_tensor = torch.FloatTensor(recent_data_scaled.T).unsqueeze(0)
        
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pred_scaled = nbeats_model(input_tensor.to(device)).cpu().numpy()
        
        nbeats_pred = scaler.inverse_transform(pred_scaled)[0][0]
    except Exception as e:
        logger.error(f"N-BEATS prediction failed, using fallback: {str(e)}")
        nbeats_pred = hist_data['Close'].iloc[-1]

    # Combine predictions with dynamic weighting
    try:
        # Calculate recent model performance for dynamic weighting
        prophet_mae = calculate_model_mae(hist_data, model_type='prophet')
        nbeats_mae = calculate_model_mae(hist_data, model_type='nbeats')
        
        total_mae = prophet_mae + nbeats_mae
        prophet_weight = (1 - prophet_mae/total_mae) * 0.7  # Base weight 70% with adjustment
        nbeats_weight = (1 - nbeats_mae/total_mae) * 0.7
        
        # Normalize weights
        weight_sum = prophet_weight + nbeats_weight
        prophet_weight /= weight_sum
        nbeats_weight /= weight_sum
        
        final_pred = (nbeats_weight * nbeats_pred + prophet_weight * prophet_pred)
    except:
        # Fallback to equal weighting if MAE calculation fails
        final_pred = (0.5 * nbeats_pred + 0.5 * prophet_pred)

    logger.info(f"Prediction generated in {time.time() - start_time:.2f}s")
    
    return JsonResponse({
        'prediction': final_pred,
        'model_used': 'N-BEATS & Prophet Ensemble',
        'prediction_date': prediction_date.strftime('%Y-%m-%d'),
        'model_weights': {
            'nbeats': round(nbeats_weight, 2) if 'nbeats_weight' in locals() else 0.5,
            'prophet': round(prophet_weight, 2) if 'prophet_weight' in locals() else 0.5
        },
        'timestamp': datetime.now().isoformat()
    })

def calculate_model_mae(hist_data, model_type, test_period=30):
    """Calculate Mean Absolute Error for model validation"""
    try:
        test_data = hist_data.iloc[-test_period*2:-test_period]
        actual_values = hist_data.iloc[-test_period:]['Close'].values
        
        if model_type == 'prophet':
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
            train_df = hist_data.iloc[:-test_period].reset_index()[['Date', 'Close']]
            train_df.columns = ['ds', 'y']
            model.fit(train_df)
            
            future = model.make_future_dataframe(periods=test_period)
            forecast = model.predict(future)
            pred_values = forecast['yhat'].values[-test_period:]
        else:  # N-BEATS
            window_size = min(30, len(hist_data)-test_period)
            X, y, scaler = prepare_nbeats_data(
                hist_data.iloc[:-test_period]['Close'],
                window_size=window_size,
                horizon=1
            )
            model = train_nbeats(X, y, epochs=30)
            
            test_input = hist_data.iloc[-test_period-window_size:-test_period]['Close'].values
            test_input_scaled = scaler.transform(test_input.reshape(-1, 1))
            input_tensor = torch.FloatTensor(test_input_scaled.T).unsqueeze(0)
            
            with torch.no_grad():
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                pred_scaled = model(input_tensor.to(device)).cpu().numpy()
            
            pred_values = scaler.inverse_transform(pred_scaled)[0]
        
        return np.mean(np.abs(actual_values - pred_values))
    except Exception as e:
        logger.error(f"Failed to calculate MAE for {model_type}: {str(e)}")
        return 1.0  # Return default MAE if calculation fails

def pred(request):
    return render(request, 'predictor/pred.html')

from django.contrib import messages
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.hashers import make_password, check_password
from .models import AboutUs, User, ContactMessage
from .forms import ContactForm


def base(request):
    return render(request, "predictor/base.html")


def index(request):
    return render(request, "predictor/index.html")


def about(request):
    about_record = AboutUs.objects.first()
    about_content = about_record.description if about_record else "No content available."
    return render(request, 'predictor/about.html', {'about_content': about_content})



def contact_view(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Your message has been sent successfully!')
            return redirect('contact')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = ContactForm()
    return render(request, 'predictor/contact.html', {'form': form})


def service(request):
    return render(request, "predictor/service.html")


def team(request):
    return render(request, "predictor/team.html")


def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        try:
            user = User.objects.get(email=email)
            if check_password(password, user.password):
                request.session['user_id'] = user.id
                messages.success(request, f"Welcome {user.first_name}!")
                return redirect('index')
            else:
                messages.error(request, "Invalid email or password")
                return redirect('login')
        except User.DoesNotExist:
            messages.error(request, "Invalid email or password")
            return redirect('login')

    return render(request, "predictor/login.html")


def register_view(request):
    if request.method == 'POST':
        fname = request.POST.get('first_name')
        lname = request.POST.get('last_name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        cpassword = request.POST.get('confirm_password')
        phone = request.POST.get('phone')

        if password != cpassword:
            messages.error(request, "Passwords do not match")
            return redirect('register')

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered")
            return redirect('register')

        hashed_password = make_password(password)
        User.objects.create(
            first_name=fname,
            last_name=lname,
            email=email,
            password=hashed_password,
            phone=phone
        )
        messages.success(request, "Registration successful! Please login.")
        return redirect('login')

    return render(request, 'predictor/register.html')


def logout_view(request):
    if 'user_id' in request.session:
        del request.session['user_id']
    messages.success(request, "You have been logged out successfully.")
    return redirect('login')




'''
from django.shortcuts import render
from django.http import JsonResponse
from .predictions import GoldPriceModel
import json
from datetime import datetime

# Initialize model when Django starts (moved to a better location)
try:
    gold_model = GoldPriceModel()
    gold_model.train_models()
except Exception as e:
    print(f"Failed to initialize model: {str(e)}")
    gold_model = None

def prediction_page(request):
    """Render the prediction interface"""
    return render(request, 'predictor/inr_pred.html')

def predict_price(request):
    """Handle prediction API requests"""
    if gold_model is None:
        return JsonResponse(
            {'error': 'Prediction model not initialized'},
            status=500
        )
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            target_date = data['prediction_date']
            
            # Validate date
            try:
                pred_date = datetime.strptime(target_date, '%Y-%m-%d').date()
                if pred_date <= datetime.now().date():
                    return JsonResponse(
                        {'error': 'Prediction date must be in the future'},
                        status=400
                    )
            except ValueError:
                return JsonResponse(
                    {'error': 'Invalid date format (YYYY-MM-DD required)'},
                    status=400
                )
            
            # Get prediction
            prediction = gold_model.predict(target_date)
            if prediction is None:
                return JsonResponse(
                    {'error': 'Failed to generate prediction'},
                    status=500
                )
            
            return JsonResponse({
                'prediction': prediction,
                'prediction_date': target_date,
                'timestamp': datetime.now().isoformat()
            })
            
        except json.JSONDecodeError:
            return JsonResponse(
                {'error': 'Invalid JSON data'},
                status=400
            )
        except KeyError:
            return JsonResponse(
                {'error': 'Missing prediction_date field'},
                status=400
            )
        except Exception as e:
            return JsonResponse(
                {'error': str(e)},
                status=500
            )
    
    return JsonResponse(
        {'error': 'Only POST requests are accepted'},
        status=405
    )'''