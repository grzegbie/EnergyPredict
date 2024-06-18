# EnergyPredict
Data source on Kaggle: <a href="https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/data">Predict Energy Behavior of Prosumers</a>

## How to use
1. Project needs `forecast_weather.csv'`, `historical_weather.csv`, `train.csv` and `weather_station_to_county_mapping.csv`files in directory `raw_data` in project directory to work properly.
2. Use scripts in order: 
   1. `data_preparation.py`
   2. `data_normalization.py`
   3. `model_training.py`
3. Trained models are saved in `models` directory in project directory