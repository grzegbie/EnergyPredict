import os
import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from config import Config


def fit_model(dataset: pd.DataFrame, is_historical: bool = False, is_business: bool = False) -> object:
    """Method fitting model to training data and displaying metrics"""
    print('Making train/test split...')
    dataset_train, dataset_test = train_test_split(dataset, test_size=Config.TEST_SIZE)

    if not is_historical:
        dataset_train_features = dataset_train[Config.FORECAST_USED_FEATURES]
        dataset_test_features = dataset_test[Config.FORECAST_USED_FEATURES]
    else:
        if is_business:
            dataset_train_features = dataset_train[Config.HISTORICAL_BUSINESS_USED_FEATURES]
            dataset_test_features = dataset_test[Config.HISTORICAL_BUSINESS_USED_FEATURES]
        else:
            dataset_train_features = dataset_train[Config.HISTORICAL_PRIVATE_USED_FEATURES]
            dataset_test_features = dataset_test[Config.HISTORICAL_PRIVATE_USED_FEATURES]

    dataset_train_target = dataset_train[Config.USED_TARGET]
    dataset_train_target = np.ravel(dataset_train_target)

    dataset_test_target = dataset_test[Config.USED_TARGET]
    dataset_test_target = np.ravel(dataset_test_target)

    # Model fitting
    print('Fitting model...')
    regressor = RandomForestRegressor(n_estimators=Config.N_ESTIMATORS, max_depth=Config.MAX_DEPTH, random_state=0)
    regressor.fit(dataset_train_features, dataset_train_target)

    print('Training set R2 score : ',
          metrics.r2_score(dataset_train_target, regressor.predict(dataset_train_features)))
    print('Validation set R2 score : ',
          metrics.r2_score(dataset_test_target, regressor.predict(dataset_test_features)))

    return regressor


def save_model(regressor: object, filename: str) -> None:
    """Method saving model to pickle file"""
    print('Saving model to', filename)
    with open(filename, 'wb') as file:
        pickle.dump(regressor, file)


# FORECAST DATA MODELS
print('Loading forecast data...')
try:
    forecast_weather_production: pd.DataFrame = pd.read_csv('standardized_data/forecast_weather_production.csv')
    forecast_weather_consumption: pd.DataFrame = pd.read_csv('standardized_data/forecast_weather_consumption.csv')
except FileNotFoundError:
    print('One of required data files are missing!')
    exit(1)
print('Forecast data loaded')

print('\nFitting model for forecast_weather_production')
forecast_weather_production_model = fit_model(forecast_weather_production, is_historical=False, is_business=False)
print('\nFitting model for forecast_weather_consumption')
forecast_weather_consumption_model = fit_model(forecast_weather_consumption, is_historical=False, is_business=False)

print('\nCreating models directory...')
try:
    os.mkdir('models', mode=0o777)
except FileExistsError:
    print("Directory 'models' already exists")

save_model(forecast_weather_production_model, 'models/forecast_weather_production_model.pkl')
save_model(forecast_weather_consumption_model, 'models/forecast_weather_consumption_model.pkl')


# HISTORICAL DATA MODELS
print('\nLoading historical data...')
try:
    historical_weather_consumption_business: pd.DataFrame = pd.read_csv(
        'standardized_data/historical_weather_consumption_business.csv')
    historical_weather_consumption_private: pd.DataFrame = pd.read_csv(
        'standardized_data/historical_weather_consumption_private.csv')
    historical_weather_production_business: pd.DataFrame = pd.read_csv(
        'standardized_data/historical_weather_production_business.csv')
    historical_weather_production_private: pd.DataFrame = pd.read_csv(
        'standardized_data/historical_weather_production_private.csv')
except FileNotFoundError:
    print('One of required data files are missing!')
    exit(1)
print('Historical data loaded')

print('\nFitting model for historical_weather_consumption_business')
historical_weather_consumption_business_model = fit_model(historical_weather_consumption_business, is_historical=True,
                                                          is_business=True)
print('\nFitting model for historical_weather_consumption_private')
historical_weather_consumption_private_model = fit_model(historical_weather_consumption_private, is_historical=True,
                                                         is_business=False)
print('\nFitting model for historical_weather_production_business')
historical_weather_production_business_model = fit_model(historical_weather_production_business, is_historical=True,
                                                         is_business=True)
print('\nFitting model for historical_weather_production_private')
historical_weather_production_private_model = fit_model(historical_weather_production_private, is_historical=True,
                                                        is_business=False)

print('Creating models directory...')
try:
    os.mkdir('models', mode=0o777)
except FileExistsError:
    print("Directory 'models' already exists")

save_model(historical_weather_consumption_business_model, 'models/historical_weather_consumption_business_model.pkl')
save_model(historical_weather_consumption_private_model, 'models/historical_weather_consumption_private_model.pkl')
save_model(historical_weather_production_business_model, 'models/historical_weather_production_business_model.pkl')
save_model(historical_weather_production_private_model, 'models/historical_weather_production_private_model.pkl')
