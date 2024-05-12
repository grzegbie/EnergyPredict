class Config:
    FORECAST_USED_NUMERICAL_FEATURES = ['latitude', 'longitude', 'temperature', 'cloudcover_total',
                                        'direct_solar_radiation', 'surface_solar_radiation_downwards', 'snowfall',
                                        'total_precipitation']

    FORECAST_USED_CATEGORICAL_FEATURES = ['is_business', 'product_type']

    USED_TARGET = ['target']

    HISTORICAL_USED_NUMERICAL_FEATURES = ['latitude', 'longitude', 'temperature', 'rain', 'snowfall',
                                          'cloudcover_total', 'shortwave_radiation', 'direct_solar_radiation',
                                          'diffuse_radiation']

    HISTORICAL_USED_CATEGORICAL_FEATURES = ['product_type']
