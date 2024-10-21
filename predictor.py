import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import load

# Load the dataset to get feature names
data = pd.read_csv("ResOverallData2_encoded.csv")
feature_names = data.drop(['HeatingEnergykwh', 'CoolingEnergy'], axis=1).columns

# Load the model and scaler (ensure these files exist)
model_RF = load('finalized_model_multivariate.sav')
#scaler = load('scaler_file.sav')  # If you saved the scaler earlier

# Initialize an empty list to store user inputs
user_inputs = []

# Prompt user to input values for each feature
print("Please enter values for the following features:")
for feature in feature_names:
    user_value = input(f"{feature}: ")
    user_inputs.append(float(user_value))


user_inputs_df = pd.DataFrame([user_inputs], columns=feature_names)
user_inputs_array = user_inputs_df.values
#user_inputs_scaled = scaler.transform(user_inputs_df)
#predicted_output = model_RF.predict(user_inputs_scaled)
predicted_output = model_RF.predict(user_inputs_array)

# Display the output values
print("Predicted Heating Energy (kWh):", predicted_output[0][0])
print("Predicted Cooling Energy:", predicted_output[0][1])

