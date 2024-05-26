import os
import pandas as pd
from config import Config

pd.options.mode.chained_assignment = None  # default='warn'


def clip_values(dataset: pd.DataFrame, factor: float = 1.5, is_historical: bool = True) -> pd.DataFrame:
    if is_historical:
        for column_name in Config.HISTORICAL_NORMALIZED_NUMERICAL_FEATURES+Config.USED_TARGET:
            q1 = dataset[column_name].quantile(0.25)
            q3 = dataset[column_name].quantile(0.75)
            iqr = (q3 - q1)
            clip_val = q3 + (factor * iqr)
            dataset[column_name] = dataset[column_name].clip(upper=clip_val)
    else:
        for column_name in Config.FORECAST_NORMALIZED_NUMERICAL_FEATURES+Config.USED_TARGET:
            q1 = dataset[column_name].quantile(0.25)
            q3 = dataset[column_name].quantile(0.75)
            iqr = (q3 - q1)
            clip_val = q3 + (factor * iqr)
            dataset[column_name] = dataset[column_name].clip(upper=clip_val)
    return dataset


# LOADING RAW DATA FILES
print('Loading raw data...')
try:
    weather_station_to_county: pd.DataFrame = pd.read_csv('raw_data/weather_station_to_county_mapping.csv')
    forecast_weather: pd.DataFrame = pd.read_csv('raw_data/forecast_weather.csv')
    historical_weather: pd.DataFrame = pd.read_csv('raw_data/historical_weather.csv')
    train: pd.DataFrame = pd.read_csv('raw_data/train.csv')
except FileNotFoundError:
    print('One of required data files are missing!')
    exit(1)
print('Raw data loaded')

# DATA PREPARATION
print('Preparing weather_station_to_county...')
# Preparation of 'weather_station_to_county_mapping.csv'
# # Removing rows with empty 'county' column and rounding latitude to 1 decimal place
weather_station_to_county.dropna(subset=['county'], inplace=True)
weather_station_to_county['latitude'] = weather_station_to_county['latitude'].round(1)

# Preparation of 'forecast_weather.csv'
print('Preparing forecast_weather...')
# Removing columns 'hours_ahead', 'origin_datetime', 'cloudcover_mid', 'cloudcover_low', 'cloudcover_high',
# 'data_block_id', '10_metre_v_wind_component', '10_metre_u_wind_component', 'dewpoint'
forecast_weather.drop(
    labels=['hours_ahead', 'origin_datetime', 'cloudcover_mid', 'cloudcover_low', 'cloudcover_high', 'data_block_id',
            '10_metre_v_wind_component', '10_metre_u_wind_component', 'dewpoint'], axis='columns', inplace=True)

# Joining and grouping 'forecast_weather' and 'weather_station_to_county'
# Inner join on 'forecast_weather' and 'weather_station_to_county' on 'longitude' and 'latitude'
forecast_mapped: pd.DataFrame = pd.merge(forecast_weather, weather_station_to_county, how='inner',
                                         left_on=['latitude', 'longitude'], right_on=['latitude', 'longitude'])

# Group by on 'forecast_mapped' on 'county' and 'forecast_datetime'
forecast_mapped = forecast_mapped.groupby(["county", "forecast_datetime"]).agg(
    {"latitude": "first", "longitude": "first", "temperature": "mean", "cloudcover_total": "mean",
     "direct_solar_radiation": "mean", "surface_solar_radiation_downwards": "mean", "snowfall": "mean",
     "total_precipitation": "mean", "county_name": "first"})

# All numerical values rounded to 2 decimal places
forecast_mapped[['temperature', 'cloudcover_total',
                 'direct_solar_radiation', 'surface_solar_radiation_downwards', 'snowfall',
                 'total_precipitation']] = forecast_mapped[['temperature', 'cloudcover_total',
                                                            'direct_solar_radiation',
                                                            'surface_solar_radiation_downwards', 'snowfall',
                                                            'total_precipitation']].round(2)

# Preparation of 'historical_weather.csv'
print('Preparing historical_weather...')
# Removing columns 'winddirection_10m', 'windspeed_10m', 'cloudcover_high', 'cloudcover_mid', 'cloudcover_low',
# 'surface_pressure', 'data_block_id', 'dewpoint'
historical_weather.drop(
    labels=['winddirection_10m', 'windspeed_10m', 'cloudcover_high', 'cloudcover_mid', 'cloudcover_low',
            'surface_pressure', 'data_block_id', 'dewpoint'], axis='columns', inplace=True)

# Joining and grouping 'historical_weather' and 'weather_station_to_county'
# Inner join on 'historical_weather' and 'weather_station_to_county' on 'longitude' and 'latitude'
historical_mapped: pd.DataFrame = pd.merge(historical_weather, weather_station_to_county, how='inner',
                                           left_on=['latitude', 'longitude'], right_on=['latitude', 'longitude'])

# Group by on 'historical_mapped' on 'county' and 'datetime'
historical_mapped = historical_mapped.groupby(["county", "datetime"]).agg(
    {"latitude": "first", "longitude": "first", "temperature": "mean", "rain": "mean",
     "snowfall": "mean",
     "cloudcover_total": "mean", "shortwave_radiation": "mean", "direct_solar_radiation": "mean",
     "diffuse_radiation": "mean", "county_name": "first"})

# All numerical values rounded to 2 decimal places
historical_mapped[
    ["temperature", "rain", "snowfall", "cloudcover_total", "shortwave_radiation", "direct_solar_radiation",
     "diffuse_radiation"]] = historical_mapped[
    ["temperature", "rain", "snowfall", "cloudcover_total", "shortwave_radiation", "direct_solar_radiation",
     "diffuse_radiation"]].round(2)

# Preparation of 'train.csv'
print('Joining with training data...')
# Removing columns 'prediction_unit_id', 'row_id', 'data_block_id'
train.drop(labels=['prediction_unit_id', 'row_id', 'data_block_id'], axis='columns', inplace=True)

# Joining and then preparing 'forecast_weather' with 'train'
# Inner join on 'train' and 'forecast_mapped'
forecast_weather_joined: pd.DataFrame = pd.merge(train, forecast_mapped, how='inner', left_on=['county', 'datetime'],
                                                 right_on=['county', 'forecast_datetime'])

# Remove rows with empty values in 'target' column
forecast_weather_joined.dropna(subset=['target'], inplace=True)

# Split into 'forecast_weather_consumption' and 'forecast_weather_production'
forecast_weather_consumption: pd.DataFrame = forecast_weather_joined[forecast_weather_joined['is_consumption'] == 1]
forecast_weather_production: pd.DataFrame = forecast_weather_joined[forecast_weather_joined['is_consumption'] == 0]

# Removing column 'is_consumption' in both datasets
forecast_weather_consumption.drop(labels='is_consumption', axis='columns', inplace=True)
forecast_weather_production.drop(labels='is_consumption', axis='columns', inplace=True)

# Joining and then preparing 'historical_weather' with 'train'
# Inner join on 'train' and 'historical_mapped'
historical_weather_joined: pd.DataFrame = pd.merge(train, historical_mapped, how='inner',
                                                   left_on=['county', 'datetime'],
                                                   right_on=['county', 'datetime'])

# Remove rows with empty values in 'target' column
historical_weather_joined.dropna(subset=['target'], inplace=True)

# Split into 'historical_weather_consumption' and 'historical_weather_production'
historical_weather_consumption: pd.DataFrame = historical_weather_joined[
    historical_weather_joined['is_consumption'] == 1]
historical_weather_production: pd.DataFrame = historical_weather_joined[
    historical_weather_joined['is_consumption'] == 0]

# Removing column 'is_consumption' in both datasets
historical_weather_consumption.drop(labels='is_consumption', axis='columns', inplace=True)
historical_weather_production.drop(labels='is_consumption', axis='columns', inplace=True)

# Split by 'is_business' column
historical_weather_consumption_business: pd.DataFrame = historical_weather_consumption[
    historical_weather_consumption['is_business'] == 1]
historical_weather_consumption_private: pd.DataFrame = historical_weather_consumption[
    historical_weather_consumption['is_business'] == 0]
historical_weather_production_business: pd.DataFrame = historical_weather_production[
    historical_weather_production['is_business'] == 1]
historical_weather_production_private: pd.DataFrame = historical_weather_production[
    historical_weather_production['is_business'] == 0]

# Removing column 'is_business' in all datasets
historical_weather_consumption_business.drop(labels='is_business', axis='columns', inplace=True)
historical_weather_consumption_private.drop(labels='is_business', axis='columns', inplace=True)
historical_weather_production_business.drop(labels='is_business', axis='columns', inplace=True)
historical_weather_production_private.drop(labels='is_business', axis='columns', inplace=True)

# Clipping values outside 1.5 IQR
forecast_weather_production = clip_values(dataset=forecast_weather_production, is_historical=False)
forecast_weather_consumption = clip_values(dataset=forecast_weather_consumption, is_historical=False)

historical_weather_consumption_business = clip_values(dataset=historical_weather_consumption_business)
historical_weather_consumption_private = clip_values(dataset=historical_weather_consumption_private)
historical_weather_production_business = clip_values(dataset=historical_weather_production_business)
historical_weather_production_private = clip_values(dataset=historical_weather_production_private)

print('Creating prepped_data directory...')
try:
    os.mkdir('prepped_data', mode=0o777)
except FileExistsError:
    print("Directory 'prepped_data' already exists")

print('Saving prepared data to .csv...')
forecast_weather_production.to_csv('prepped_data/forecast_weather_production.csv', index=False)
forecast_weather_consumption.to_csv('prepped_data/forecast_weather_consumption.csv', index=False)
historical_weather_consumption_business.to_csv('prepped_data/historical_weather_consumption_business.csv', index=False)
historical_weather_consumption_private.to_csv('prepped_data/historical_weather_consumption_private.csv', index=False)
historical_weather_production_business.to_csv('prepped_data/historical_weather_production_business.csv', index=False)
historical_weather_production_private.to_csv('prepped_data/historical_weather_production_private.csv', index=False)
print('Data saved')
