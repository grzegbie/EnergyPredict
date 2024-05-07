class Config:
    USED_NUMERICAL_FEATURES = ['latitude', 'longitude', 'temperature', 'dewpoint', 'cloudcover_total',
                               'direct_solar_radiation', 'surface_solar_radiation_downwards', 'snowfall',
                               'total_precipitation']

    USED_CATEGORICAL_FEATURES = ['is_business', 'product_type']

    USED_TARGET = ['target']

    # @property
    # def USED_NUMERICAL_FEATURES(self):
    #     return ['latitude', 'longitude', 'temperature', 'dewpoint', 'cloudcover_total',
    #             'direct_solar_radiation', 'surface_solar_radiation_downwards', 'snowfall',
    #             'total_precipitation']
    #
    # @property
    # def USED_CATEGORICAL_FEATURES(self):
    #     return ['is_business', 'product_type']
    #
    # @property
    # def USED_TARGET(self):
    #     return ['target']
    #
    # @property
    # def ALL_COLUMNS(self):
    #     return self.USED_TARGET + self.USED_CATEGORICAL_FEATURES + self.USED_NUMERICAL_FEATURES
