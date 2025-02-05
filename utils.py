import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer

def load_and_validate_data(file, validate_only=False):
    """
    Load and validate uploaded data file with proper encoding and missing value handling
    """
    try:
        # Try reading with different encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        df = None

        for encoding in encodings:
            try:
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file, encoding=encoding)
                else:
                    df = pd.read_excel(file)
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                raise ValueError(f"Error reading file: {str(e)}")

        if df is None:
            raise ValueError("Unable to read the file. Please ensure it's properly formatted.")

        # Convert potential date columns
        for col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                continue

        if validate_only:
            return df

        # Handle missing values using forward fill then backward fill
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(method='bfill')

        # If still have missing values, use mean imputation
        if df[numeric_columns].isna().any().any():
            imputer = SimpleImputer(strategy='mean')
            df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

        # Ensure we have enough data points
        if len(df) < 10:
            raise ValueError("Not enough data points. Please provide at least 10 data points for forecasting.")

        # Sort by date and reset index
        df = df.sort_values('ds').reset_index(drop=True)

        return df

    except Exception as e:
        raise ValueError(f"Error processing file: {str(e)}\n\n"
                      "Please ensure your file:\n"
                      "1. Contains a date column\n"
                      "2. Has numeric columns for forecasting\n"
                      "3. Is properly formatted (CSV or Excel)\n"
                      "4. Has sufficient data points (minimum 10)")

def calculate_metrics(model, actual_data):
    """
    Calculate MAE and RMSE for the forecast
    """
    predictions = model.predict(actual_data[['ds']])
    mae = mean_absolute_error(actual_data['y'], predictions['yhat'])
    rmse = np.sqrt(mean_squared_error(actual_data['y'], predictions['yhat']))
    return mae, rmse