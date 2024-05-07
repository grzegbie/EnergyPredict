import pandas as pd
from sklearn import preprocessing
from config import Config

# Loading datasets
production = pd.read_csv('prepped_data/train_production.csv')
consumption = pd.read_csv('prepped_data/train_consumption.csv')

# Data standardization

# Split production dataset based on column data type - categorical, numerical and target
production_num = production[Config.USED_NUMERICAL_FEATURES]
production_cat = production[Config.USED_CATEGORICAL_FEATURES]
production_target = production[Config.USED_TARGET]

# Split consumption dataset based on column data type - categorical, numerical and target
consumption_num = consumption[Config.USED_NUMERICAL_FEATURES]
consumption_cat = consumption[Config.USED_CATEGORICAL_FEATURES]
consumption_target = consumption[Config.USED_TARGET]

# Standardize numerical features
min_max_scaler = preprocessing.MinMaxScaler()
production_num_std = min_max_scaler.fit_transform(production_num)
consumption_num_std = min_max_scaler.fit_transform(consumption_num)

production_num_std = pd.DataFrame(production_num_std, columns=Config.USED_NUMERICAL_FEATURES)
consumption_num_std = pd.DataFrame(consumption_num_std, columns=Config.USED_NUMERICAL_FEATURES)

production_features = pd.concat([production_num_std, production_cat], axis=1)
consumption_features = pd.concat([consumption_num_std, consumption_cat], axis=1)

# Save standardized data to .csv
production_features.to_csv('normalized_data/production_features.csv', index=False)
consumption_features.to_csv('normalized_data/consumption_features.csv', index=False)
production_target.to_csv('normalized_data/production_target.csv', index=False)
consumption_target.to_csv('normalized_data/consumption_target.csv', index=False)


# # Standardize categorical features with LabelEncoding
# production_is_business = production_cat['is_business']
# production_product_type = production_cat['product_type']
#
# consumption_is_business = consumption_cat['is_business']
# consumption_product_type = consumption_cat['product_type']
#
# label_encoder = LabelEncoder()
# production_is_business = label_encoder.fit_transform(production_is_business)
# production_product_type = label_encoder.fit_transform(production_product_type)
#
# production_is_business = pd.DataFrame(production_is_business, columns=['is_business'])
# production_product_type = pd.DataFrame(production_product_type, columns=['product_type'])
#
