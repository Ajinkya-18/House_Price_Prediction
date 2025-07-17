from utils import load_data, preprocess_data, save_model, train_model, test_model


# load data
df = load_data('data\housing.csv')
# print(df.shape)


# preprocess data
x_train, x_val, y_train, y_val = preprocess_data(df, target='median_house_value')
# print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)


# model training
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

rfr = RandomForestRegressor(max_depth=10, min_samples_split= 20, 
                            min_samples_leaf= 10, random_state=42)
hgbr = HistGradientBoostingRegressor(max_iter=200, random_state=42)

rfr = train_model(rfr, x_train, y_train)
hgbr = train_model(hgbr, x_train, y_train)


# # validate model performance
rfr_score = test_model(model=rfr, x_test=x_val, y_test=y_val)
print(rfr_score*100)

hgbr_score = test_model(model=hgbr, x_test=x_val, y_test=y_val)
print(hgbr_score*100)


# Save trained models
# save_model(hgbr, 'models/hgbr_best_model.joblib')





