import pandas as pd

# Loading 'weather_station_to_county_mapping.csv'
weather_station_to_county: pd.DataFrame = pd.read_csv('raw_data/weather_station_to_county_mapping.csv')

# Removing rows with empty 'county' column and rounding latitude to 1 decimal place
weather_station_to_county.dropna(subset=['county'], inplace=True)
weather_station_to_county['latitude'] = weather_station_to_county['latitude'].round(1)

# Loading 'forecast_weather.csv'
forecast_weather: pd.DataFrame = pd.read_csv('raw_data/forecast_weather.csv')

# Removing columns 'hours_ahead', 'origin_datetime', 'cloudcover_mid', 'cloudcover_low', 'cloudcover_high',
# 'data_block_id', '10_metre_v_wind_component', '10_metre_u_wind_component'
forecast_weather.drop(
    labels=['hours_ahead', 'origin_datetime', 'cloudcover_mid', 'cloudcover_low', 'cloudcover_high', 'data_block_id',
            '10_metre_v_wind_component', '10_metre_u_wind_component'], axis='columns', inplace=True)

# Inner join on 'forecast_weather' and 'weather_station_to_county' on 'longitude' and 'latitude'
forecast_mapped: pd.DataFrame = pd.merge(forecast_weather, weather_station_to_county, how='inner',
                                         left_on=['latitude', 'longitude'], right_on=['latitude', 'longitude'])

# Group by on 'forecast_mapped' on 'county' and 'forecast_datetime'
forecast_mapped = forecast_mapped.groupby(["county", "forecast_datetime"]).agg(
    {"latitude": "first", "longitude": "first", "temperature": "mean", "dewpoint": "mean", "cloudcover_total": "mean",
     "direct_solar_radiation": "mean", "surface_solar_radiation_downwards": "mean", "snowfall": "mean",
     "total_precipitation": "mean", "county_name": "first"})

# All numerical values rounded to 2 decimal places
forecast_mapped[['temperature', 'dewpoint', 'cloudcover_total',
                 'direct_solar_radiation', 'surface_solar_radiation_downwards', 'snowfall',
                 'total_precipitation']] = forecast_mapped[['temperature', 'dewpoint', 'cloudcover_total',
                                                            'direct_solar_radiation',
                                                            'surface_solar_radiation_downwards', 'snowfall',
                                                            'total_precipitation']].round(2)

# Loading 'train.csv'
train: pd.DataFrame = pd.read_csv('raw_data/train.csv')

# Splitting 'train' into 'train_consumption' and 'train_production'
train_consumption: pd.DataFrame = train[train['is_consumption'] == 1]
train_production: pd.DataFrame = train[train['is_consumption'] == 0]

# Inner join on 'train_consumption' and 'forecast_weather'
train_consumption = pd.merge(train_consumption, forecast_mapped, how='inner',
                             left_on=['county', 'datetime'], right_on=['county', 'forecast_datetime'])

# Inner join on 'train_production' and 'forecast_weather'
train_production = pd.merge(train_production, forecast_mapped, how='inner',
                            left_on=['county', 'datetime'], right_on=['county', 'forecast_datetime'])

# Remove column 'is_consumption'
train_consumption.drop(labels='is_consumption', axis='columns', inplace=True)
train_production.drop(labels='is_consumption', axis='columns', inplace=True)

# Remove rows with empty values in 'target' column
train_consumption.dropna(subset=['target'], inplace=True)
train_production.dropna(subset=['target'], inplace=True)

# Saving prepared datasets to .csv
train_production.to_csv('prepped_data/train_production.csv', index=False)
train_consumption.to_csv('prepped_data/train_consumption.csv', index=False)
