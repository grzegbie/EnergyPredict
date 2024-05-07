import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

# PRODUCTION MODEL
# Loading the production data
production_features = pd.read_csv('normalized_data/production_features.csv')
production_targets = pd.read_csv('normalized_data/production_target.csv')

# Splitting features and target datasets into train and test subsets
production_train_features = production_features.sample(frac=0.8, random_state=200)
production_test_features = production_features.drop(production_train_features.index)
production_train_target = production_targets.sample(frac=0.8, random_state=200)
production_test_target = production_targets.drop(production_train_target.index)

production_train_target = np.ravel(production_train_target)
production_test_target = np.ravel(production_test_target)

# Model fitting
production_regressor = RandomForestRegressor(n_estimators=100, max_depth=14, random_state=0)
(production_regressor.fit(production_train_features, production_train_target))

# Saving model to .pkl
with open('production_model.pkl', 'wb') as file:
    pickle.dump(production_regressor, file)

# Model R2 metric check on train and test subsets
print('Production training set R2 score : ',
      metrics.r2_score(production_train_target,
                       production_regressor.predict(production_train_features)) * 100)
print('Production validation set R2 score : ',
      metrics.r2_score(production_test_target,
                       production_regressor.predict(production_test_features)) * 100)

# CONSUMPTION MODEL
# Loading the consumption data
consumption_features = pd.read_csv('normalized_data/consumption_features.csv')
consumption_targets = pd.read_csv('normalized_data/consumption_target.csv')

# Splitting features and target datasets into train and test subsets
consumption_train_features = consumption_features.sample(frac=0.8, random_state=200)
consumption_test_features = consumption_features.drop(consumption_train_features.index)
consumption_train_target = consumption_targets.sample(frac=0.8, random_state=200)
consumption_test_target = consumption_targets.drop(consumption_train_target.index)

consumption_train_target = np.ravel(consumption_train_target)
consumption_test_target = np.ravel(consumption_test_target)

# Model fitting
consumption_regressor = RandomForestRegressor(n_estimators=100, max_depth=14, random_state=0)
(consumption_regressor.fit(consumption_train_features, consumption_train_target))

# Saving model to .pkl
with open('consumption_model.pkl', 'wb') as file:
    pickle.dump(consumption_regressor, file)

# Model R2 metric check on train and test subsets
print('Consumption training set R2 score : ',
      metrics.r2_score(consumption_train_target,
                       consumption_regressor.predict(consumption_train_features)) * 100)
print('Consumption validation set R2 score : ',
      metrics.r2_score(consumption_test_target,
                       consumption_regressor.predict(consumption_test_features)) * 100)
