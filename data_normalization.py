import os
import pandas as pd
from sklearn import preprocessing
from config import Config


def normalize_data(dataset: pd.DataFrame, is_historical: bool = True) -> pd.DataFrame:
    """Method normalizing both categorical and numerical data"""
    min_max_scaler = preprocessing.MinMaxScaler()
    if is_historical:
        dataset[Config.HISTORICAL_NORMALIZED_NUMERICAL_FEATURES + Config.USED_TARGET] = min_max_scaler.fit_transform(
            dataset[Config.HISTORICAL_NORMALIZED_NUMERICAL_FEATURES + Config.USED_TARGET])

        dataset_dummies: pd.DataFrame = pd.get_dummies(
            data=dataset[Config.HISTORICAL_ENCODED_CATEGORICAL_FEATURES],
            columns=Config.HISTORICAL_ENCODED_CATEGORICAL_FEATURES)

        dataset.drop(labels=Config.HISTORICAL_ENCODED_CATEGORICAL_FEATURES, axis='columns', inplace=True)
        dataset = pd.concat([dataset, dataset_dummies], axis=1)
    else:
        dataset[Config.FORECAST_NORMALIZED_NUMERICAL_FEATURES + Config.USED_TARGET] = min_max_scaler.fit_transform(
            dataset[Config.FORECAST_NORMALIZED_NUMERICAL_FEATURES + Config.USED_TARGET])

        dataset_dummies: pd.DataFrame = pd.get_dummies(
            data=dataset[Config.FORECAST_ENCODED_CATEGORICAL_FEATURES],
            columns=Config.FORECAST_ENCODED_CATEGORICAL_FEATURES)

        dataset.drop(labels=Config.FORECAST_ENCODED_CATEGORICAL_FEATURES, axis='columns', inplace=True)
        dataset = pd.concat([dataset, dataset_dummies], axis=1)

    return dataset


# Loading datasets
print('Loading data...')
try:
    forecast_weather_production: pd.DataFrame = pd.read_csv('prepped_data/forecast_weather_production.csv')
    forecast_weather_consumption: pd.DataFrame = pd.read_csv('prepped_data/forecast_weather_consumption.csv')
    historical_weather_consumption_business: pd.DataFrame = pd.read_csv(
        'prepped_data/historical_weather_consumption_business.csv')
    historical_weather_consumption_private: pd.DataFrame = pd.read_csv(
        'prepped_data/historical_weather_consumption_private.csv')
    historical_weather_production_business: pd.DataFrame = pd.read_csv(
        'prepped_data/historical_weather_production_business.csv')
    historical_weather_production_private: pd.DataFrame = pd.read_csv(
        'prepped_data/historical_weather_production_private.csv')
except FileNotFoundError:
    print('One of required data files are missing!')
    exit(1)
print('Data loaded')

# Data normalization
print('Normalizing features...')
forecast_weather_production = normalize_data(dataset=forecast_weather_production, is_historical=False)
forecast_weather_consumption = normalize_data(dataset=forecast_weather_consumption, is_historical=False)

historical_weather_consumption_business = normalize_data(dataset=historical_weather_consumption_business)
historical_weather_consumption_private = normalize_data(dataset=historical_weather_consumption_private)
historical_weather_production_business = normalize_data(dataset=historical_weather_production_business)
historical_weather_production_private = normalize_data(dataset=historical_weather_production_private)

print('Creating normalized_data directory...')
try:
    os.mkdir('normalized_data', mode=0o777)
except FileExistsError:
    print("Directory 'normalized_data' already exists")

print('Saving normalized data to .csv...')
forecast_weather_production.to_csv('normalized_data/forecast_weather_production.csv', index=False)
forecast_weather_consumption.to_csv('normalized_data/forecast_weather_consumption.csv', index=False)

historical_weather_consumption_business.to_csv('normalized_data/historical_weather_consumption_business.csv',
                                               index=False)
historical_weather_consumption_private.to_csv('normalized_data/historical_weather_consumption_private.csv',
                                              index=False)
historical_weather_production_business.to_csv('normalized_data/historical_weather_production_business.csv',
                                              index=False)
historical_weather_production_private.to_csv('normalized_data/historical_weather_production_private.csv',
                                             index=False)
print('Data saved')
