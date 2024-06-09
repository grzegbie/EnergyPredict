class Config:
    # Data preparation and standardization parameters
    FORECAST_NORMALIZED_NUMERICAL_FEATURES = ['latitude', 'longitude', 'temperature', 'cloudcover_total',
                                              'direct_solar_radiation', 'surface_solar_radiation_downwards', 'snowfall',
                                              'total_precipitation']

    FORECAST_CLIPPED_NUMERICAL_FEATURES = ['latitude', 'longitude', 'temperature',
                                           'direct_solar_radiation', 'surface_solar_radiation_downwards', 'snowfall',
                                           'total_precipitation']

    FORECAST_ENCODED_CATEGORICAL_FEATURES = ['is_business', 'product_type']

    HISTORICAL_NORMALIZED_NUMERICAL_FEATURES = ['latitude', 'longitude', 'temperature', 'rain', 'snowfall',
                                                'cloudcover_total', 'shortwave_radiation', 'direct_solar_radiation',
                                                'diffuse_radiation']

    HISTORICAL_CLIPPED_NUMERICAL_FEATURES = ['latitude', 'longitude', 'temperature', 'rain', 'snowfall',
                                             'shortwave_radiation', 'direct_solar_radiation',
                                             'diffuse_radiation']

    HISTORICAL_ENCODED_CATEGORICAL_FEATURES = ['product_type']

    FORECAST_USED_FEATURES = ['latitude', 'longitude', 'temperature', 'cloudcover_total', 'direct_solar_radiation',
                              'surface_solar_radiation_downwards', 'snowfall', 'total_precipitation', 'is_business_0',
                              'is_business_1', 'product_type_0', 'product_type_1', 'product_type_2', 'product_type_3']

    HISTORICAL_BUSINESS_USED_FEATURES = ['latitude', 'longitude', 'temperature', 'rain', 'snowfall', 'cloudcover_total',
                                         'shortwave_radiation', 'direct_solar_radiation', 'diffuse_radiation',
                                         'product_type_0', 'product_type_1', 'product_type_2', 'product_type_3']

    HISTORICAL_PRIVATE_USED_FEATURES = ['latitude', 'longitude', 'temperature', 'rain', 'snowfall', 'cloudcover_total',
                                        'shortwave_radiation', 'direct_solar_radiation', 'diffuse_radiation',
                                        'product_type_1', 'product_type_2', 'product_type_3']

    USED_TARGET = ['target']

    # Model fitting parameters
    TEST_SIZE = 0.2

    N_ESTIMATORS = 100

    MAX_DEPTH = 13
