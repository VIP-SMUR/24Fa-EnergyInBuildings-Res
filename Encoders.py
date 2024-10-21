import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Read the CSV file
df = pd.read_csv('ResOverallData2.csv')
df = df.drop('bldg_id', axis=1)
df = df.dropna(axis=1, how='all')
# Step 2: Initialize LabelEncoder for both columns
label_encoder_building = LabelEncoder()
label_encoder_orientation = LabelEncoder()

# Step 3: Fit and transform the 'BuildingType' column
df['BuildingType'] = label_encoder_building.fit_transform(df['BuildingType'])

# Step 4: Fit and transform the 'Orientation' column
df['Orientation'] = label_encoder_orientation.fit_transform(df['Orientation'])

#Save mappings

building_mapping = pd.DataFrame({
    'Original': label_encoder_building.classes_,
    'Encoded': range(len(label_encoder_building.classes_))
})

orientation_mapping = pd.DataFrame({
    'Original': label_encoder_orientation.classes_,
    'Encoded': range(len(label_encoder_orientation.classes_))
})

building_mapping.to_csv('BuildingType_mapping.csv', index=False)
orientation_mapping.to_csv('Orientation_mapping.csv', index=False)
print("Mappings saved to 'BuildingType_mapping.csv' and 'Orientation_mapping.csv'")
# Step 5: Save the updated DataFrame to a new CSV
df.to_csv('ResOverallData2_encoded.csv', index=False)

# (Optional) Print a success message
print("Encoded data saved to 'ResOverallData2_encoded.csv'")
