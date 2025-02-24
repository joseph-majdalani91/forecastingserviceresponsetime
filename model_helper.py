import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import timedelta
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def download_odata(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data['value'])
        return df
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return None

def describe_dataframe(df):
    """
    Describe all columns in a DataFrame with:
      - Missing values and their percentage.
      - Unique values count and their percentage.
      - Basic statistics for numerical columns.
      - Mode and frequency for categorical columns.
      
    """
    
    description = []
    total_rows = len(df)
    # Convert empty strings to null
    df = df.replace("", pd.NA)
    
    for col in df.columns:
        missing_values = df[col].isna().sum()
        missing_percentage = (missing_values / total_rows) * 100
        
        # Check for dictionary/complex type columns
        if df[col].apply(lambda x: isinstance(x, dict)).any():
            col_type = "Dictionary/Complex"
            unique_values = "N/A"
            unique_percentage = "N/A"
            top_value = "N/A"
            top_count = "N/A"
            stats = "Skipped"
        
        # Categorical: object type or fewer than 20 unique values
        elif df[col].dtype == 'object' or df[col].nunique() < 20:
            col_type = "Categorical"
            unique_values = df[col].nunique()
            unique_percentage = round((unique_values / total_rows) * 100, 2)
            top_value = df[col].mode()[0] if not df[col].mode().empty else None
            top_count = df[col].value_counts().iloc[0] if unique_values > 0 else 0
            stats = "N/A"
        
        # Numerical columns
        else:
            col_type = "Numerical"
            unique_values = df[col].nunique()
            unique_percentage = round((unique_values / total_rows) * 100, 0)
            top_value = "N/A"
            top_count = "N/A"
            stats = {
                "Mean": df[col].mean(),
                "Min": df[col].min(),
                "Max": df[col].max(),
                "Std": df[col].std()
            }
        
        description.append({
            "Column Name": col,
            "Type": col_type,
            "Missing Values": missing_values,
            "Missing %": round(missing_percentage, 0),
            "Unique Values": unique_values,
            "Unique %": unique_percentage,
            "Most Frequent Value": top_value,
            "Frequency": top_count,
            "Statistics": stats,
        })
    
    return pd.DataFrame(description)

def impute_missing_coordinates(df):
    df = df.copy()

    # Fill unspecified park_borough with "MANHATTAN"
    df['park_borough'].fillna("MANHATTAN", inplace=True)

    # Function to find replacement coordinates
    def get_replacement_coords(borough):
        subset = df[df['park_borough'] == borough].dropna(subset=['latitude', 'longitude'])
        return subset[['latitude', 'longitude']].sample(1).values[0] if not subset.empty else [np.nan, np.nan]

    # Iterate over rows with missing coordinates
    for idx, row in df[df[['latitude', 'longitude']].isnull().any(axis=1)].iterrows():
        lat, lon = get_replacement_coords(row['park_borough'])
        df.at[idx, 'latitude'] = lat
        df.at[idx, 'longitude'] = lon

    return df

# Mapping: Open-Meteo weather codes to reduced weather categories.
REDUCED_WEATHER_MAP = {
    # Clear
    0: "Clear", 1: "Clear", 2: "Clear", 3: "Clear",
    # Fog
    45: "Fog", 48: "Fog",
    # Rain (drizzle, rain, rain showers)
    51: "Rain", 53: "Rain", 55: "Rain", 56: "Rain", 57: "Rain",
    61: "Rain", 63: "Rain", 65: "Rain", 66: "Rain", 67: "Rain",
    80: "Rain", 81: "Rain", 82: "Rain",
    # Snow (snowfall, snow showers)
    71: "Snow", 73: "Snow", 75: "Snow", 77: "Snow",
    85: "Snow", 86: "Snow",
    # Thunderstorms
    95: "Thunderstorm", 96: "Thunderstorm", 99: "Thunderstorm"
}

# Severity ranking: higher number = worse condition.
WEATHER_SEVERITY = {
    "Clear": 0,
    "Fog": 1,
    "Rain": 2,
    "Snow": 3,
    "Thunderstorm": 4,
    "Unknown": 1  # Default severity for unmapped codes.
}

def reduce_weather_code(code):
    """Map a single weather code to a reduced category."""
    return REDUCED_WEATHER_MAP.get(code, "Unknown")

def parse_date_str(date_str):
    """Convert a date-time string to a datetime.date object."""
    return pd.to_datetime(date_str).date()

def compute_end_date(start_date, offset_days=1):
    """
    For a 3-day window, compute the end date as start_date + 2 days.
    (This covers created_date and the following 2 days.)
    """
    return start_date + timedelta(days=offset_days)

def fetch_weather_data(latitude, longitude, start_date, end_date, timezone="America/New_York"):
    """
    Fetch daily weather codes between start_date and end_date (inclusive)
    from Open-Meteo’s Archive API using the specified time zone.
    
    Returns a list of (date_str, code) pairs.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "daily": "weathercode",
        "timezone": timezone
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"Error fetching weather data for lat={latitude}, lon={longitude} from {start_date} to {end_date}: {e}")
        return []
    
    if "daily" in data and "time" in data["daily"] and "weathercode" in data["daily"]:
        dates = data["daily"]["time"]
        codes = data["daily"]["weathercode"]
        return list(zip(dates, codes))
    
    return []

def get_worst_weather_for_row(row):
    """
    For a given row, fetch weather data for a 3-day window starting from 'created_date',
    map each day's code to a reduced category, and return the worst-case (most severe)
    weather condition based on a predefined severity ranking.
    """
    start_date = parse_date_str(row["created_date"])
    end_date = compute_end_date(start_date, offset_days=2)
    
    daily_data = fetch_weather_data(
        latitude=row["latitude"],
        longitude=row["longitude"],
        start_date=start_date,
        end_date=end_date,
        timezone="America/New_York"
    )
    
    # Map each day's weather code to its reduced category.
    categories = [reduce_weather_code(code) for _, code in daily_data]
    
    # Choose the worst-case weather by severity ranking.
    if categories:
        worst = max(categories, key=lambda cat: WEATHER_SEVERITY.get(cat, 0))
    else:
        worst = "Unknown"
    return worst

def enrich_df_with_worst_weather(df):
    """
    For each row in the DataFrame, determine the worst-case weather condition
    over the 3-day window and attach it as a new column 'weather_condition'.
    """
    return df.assign(weather_condition=df.apply(get_worst_weather_for_row, axis=1))

def get_season(date):
    # get the time of the year during which the service request was created
    month = date.month
    if month in (12, 1, 2, 3):
        return "Winter"
    elif month in (4, 5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    elif month in (9, 10, 11):
        return "Fall"
    return "Unknown"

def add_season_column(df):
    df = df.copy()
    df["created_date"] = pd.to_datetime(df["created_date"])
    df["season"] = df["created_date"].apply(get_season)
    return df

def count_active(row, df):
    # Filter rows for the same agency, created before the current row's created_date,
    # and not yet closed at the current row's closed_date.
    active = df[
        (df['agency'] == row['agency']) &
        (df['created_date'] < row['created_date']) &
        (df['closed_date'] > row['created_date'])
    ]
    return active.shape[0]