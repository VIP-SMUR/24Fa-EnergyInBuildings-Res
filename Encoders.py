import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Read the CSV file
df = pd.read_csv('ResOverallData2.csv')

# Step 2: Initialize LabelEncoder for both columns
label_encoder_building = LabelEncoder()
label_encoder_orientation = LabelEncoder()

# Step 3: Fit and transform the 'BuildingType' column
df['BuildingType'] = label_encoder_building.fit_transform(df['BuildingType'])

# Step 4: Fit and transform the 'Orientation' column
df['Orientation'] = label_encoder_orientation.fit_transform(df['Orientation'])

# Step 5: Save the updated DataFrame to a new CSV
df.to_csv('ResOverallData2_encoded.csv', index=False)

# (Optional) Print a success message
print("Encoded data saved to 'ResOverallData2_encoded.csv'")
