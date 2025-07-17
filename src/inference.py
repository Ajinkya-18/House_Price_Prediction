from utils import get_inference_data, load_model

# suppress warnings
import warnings
warnings.filterwarnings('ignore')


# get inference data
data = get_inference_data()


# Load trained model
model = load_model('models/RandomForestRegressor.joblib')


# Make predictions on inference data
y_hat = model.predict(data)


# Display the predicted result
# print(data)
print(f'House Valuation: {round(y_hat[0], 1)}')


