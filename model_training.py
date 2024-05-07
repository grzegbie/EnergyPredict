import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Loading the production data
production_features = pd.read_csv('normalized_data/production_features.csv')
production_targets = pd.read_csv('normalized_data/production_target.csv')
production_targets = np.ravel(production_targets)

# Model fitting
regressor = RandomForestRegressor(n_estimators=100, max_depth=14, random_state=0)
regressor.fit(production_features, production_targets)
print('Energy production R2 score: ', regressor.score(production_features, production_targets))


# Loading the consumption data
consumption_features = pd.read_csv('normalized_data/consumption_features.csv')
consumption_targets = pd.read_csv('normalized_data/consumption_target.csv')

# Model fitting
regressor = RandomForestRegressor(n_estimators=100, max_depth=14, random_state=0)
model = regressor.fit(consumption_features, consumption_targets)
print('Energy consumption R2 score: ', regressor.score(consumption_features, consumption_targets))
