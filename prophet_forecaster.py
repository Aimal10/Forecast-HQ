from prophet import Prophet
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_holiday_events():
    """
    Create a DataFrame of US holidays for Prophet
    """
    # Get current year
    current_year = pd.Timestamp.now().year

    # Create holiday events for 5 years (2 years back and 2 years forward)
    years = range(current_year - 2, current_year + 3)

    holidays_list = []
    for year in years:
        holidays_list.extend([
            {'holiday': 'New Year\'s Day', 'ds': f'{year}-01-01'},
            {'holiday': 'Martin Luther King Jr. Day', 'ds': f'{year}-01-{15 + (7 - pd.Timestamp(f"{year}-01-15").dayofweek) % 7}'},
            {'holiday': 'Presidents\' Day', 'ds': f'{year}-02-{15 + (7 - pd.Timestamp(f"{year}-02-15").dayofweek) % 7}'},
            {'holiday': 'Memorial Day', 'ds': f'{year}-05-{25 + (7 - pd.Timestamp(f"{year}-05-25").dayofweek) % 7}'},
            {'holiday': 'Independence Day', 'ds': f'{year}-07-04'},
            {'holiday': 'Labor Day', 'ds': f'{year}-09-{1 + (7 - pd.Timestamp(f"{year}-09-01").dayofweek) % 7}'},
            {'holiday': 'Thanksgiving', 'ds': f'{year}-11-{22 + (7 - pd.Timestamp(f"{year}-11-22").dayofweek) % 7}'},
            {'holiday': 'Christmas', 'ds': f'{year}-12-25'},
            # Black Friday (day after Thanksgiving)
            {'holiday': 'Black Friday', 'ds': pd.Timestamp(f'{year}-11-{22 + (7 - pd.Timestamp(f"{year}-11-22").dayofweek) % 7}') + pd.Timedelta(days=1)},
            # Cyber Monday (Monday after Thanksgiving)
            {'holiday': 'Cyber Monday', 'ds': pd.Timestamp(f'{year}-11-{22 + (7 - pd.Timestamp(f"{year}-11-22").dayofweek) % 7}') + pd.Timedelta(days=4)},
        ])

    holidays = pd.DataFrame(holidays_list)

    # Add windows for holiday effects
    holidays['lower_window'] = 0  # No extra days before the holiday
    holidays['upper_window'] = 1  # One extra day after the holiday

    return holidays

def process_and_forecast(df, periods, interval):
    """
    Process data and generate forecast using Prophet with holiday effects
    """
    # Create holidays DataFrame
    holidays = create_holiday_events()

    # Initialize Prophet model with holidays
    model = Prophet(
        interval_width=interval,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        holidays=holidays,
        changepoint_prior_scale=0.05
    )

    # Fit the model
    model.fit(df)

    # Create future dataframe
    future = model.make_future_dataframe(periods=periods)

    # Generate forecast
    forecast = model.predict(future)

    # Ensure no negative values in forecast
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)

    return forecast, model