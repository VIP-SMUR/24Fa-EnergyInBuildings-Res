# a) Load libraries
import pandas as pd
from pandas import set_option
from pandas import DataFrame

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from keras.layers import BatchNormalization, Dropout, Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score

from pickle import dump


# b) Load dataset
# Load and view each file in the dataset
data = pd.read_csv('ResOverallData2_encoded.csv', header=0)
print(data.shape)
print(data.head(10))
print(data.dtypes)

# Define features and label (Heating)
X = data.drop(columns=['HeatingEnergykwh', 'CoolingEnergy'], axis = 1)
y = data[['HeatingEnergykwh', 'CoolingEnergy']]
print(X.shape, y.shape)

# a) Descriptive statistics
set_option("display.precision", 3)
print(data.describe())

# checking categorical features
cat_features = data.select_dtypes(include='O').keys()
# display variabels
cat_features

# b) Data visualizations
# Box and Whisker Plots of a few features

plt.rcParams["figure.figsize"] = [10, 6]
plt.rcParams["figure.autolayout"] = True

raw_plot = pd.DataFrame(data)
raw_plot.plot(kind='box', subplots=True, layout=(4,3), sharex=False, sharey=False)
plt.show()


# create the stack bar chart of Heating and Cooling Load
fig, ax = plt.subplots()
ax.bar(np.arange(len(data['HeatingEnergykwh'])), data['HeatingEnergykwh'], label='HeatingEnergykwh')

ax.bar(np.arange(len(data['HeatingEnergykwh'])), data['CoolingEnergy'], bottom=data['CoolingEnergy'], label='CoolingEnergy')

# add labels, title and legend
ax.set_xlabel('Number of Samples')
ax.set_ylabel('Energy Consumption (kWh)')
ax.set_title('Heating and Cooling Energy Consumption')
ax.legend()

# show the plot
plt.show()

# Create a figure with subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Plot the density for 'HeatingEnergykwh'
sns.kdeplot(data['HeatingEnergykwh'], ax=axes[0], color='steelblue', fill=True)
axes[0].set_title('HeatingEnergykwh')

# Plot the density for 'CoolingEnergy'
sns.kdeplot(data['CoolingEnergy'], ax=axes[1], color='orange', fill=True)
axes[1].set_title('CoolingEnergy')

# Adjust the layout and spacing of the subplots
fig.tight_layout()

plt.show()

# correlation matrix between all pairs of attributes
corr_matrix = data.corr(method='pearson')

# Create a heatmap using seaborn
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, vmin=-1, vmax=1, annot=True, cmap="coolwarm", ax=ax)

plt.show()


# a) Data Cleaning
# summarize the number of unique values in each column
print(data.nunique())


# Identify columns with missing values and count the number of missing values
data.columns[data.isnull().any()]
print(data.isnull().sum())

# Correlation with 'HeatingEnergykwh'
corr_heating = data.select_dtypes(include=np.number).corrwith(data['HeatingEnergykwh']).sort_values(ascending=False)
# Correlation with 'CoolingEnergy'
corr_cooling = data.select_dtypes(include=np.number).corrwith(data['CoolingEnergy']).sort_values(ascending=False)

# filter columns with low correlation
corr_threshold = 0.05
low_corr_columns_heating = corr_heating[corr_heating.abs() < corr_threshold].index
low_corr_columns_cooling = corr_cooling[corr_cooling.abs() < corr_threshold].index

# Combine low-correlation columns from both outputs
low_corr_columns = low_corr_columns_heating.union(low_corr_columns_cooling)

# remove outliers from numerical columns with low correlation
for column in low_corr_columns:
    lower = data[column].quantile(0.01)
    upper = data[column].quantile(0.99)
    data[column] = data[column].clip(lower, upper)

# b) Split-out dataset into train and validation sets

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=1)

print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)


# c) Data Transforms
# Standardize the dataset by rescaling the distribution of values

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)

# a) Spot check algorithms using cross-validation technique
num_folds = 8
seed = 8

# Select 5 most popular linear and tree-based algorithms for evaluation
models = []
models.append(('LR', LinearRegression()))
models.append(('EN', ElasticNet()))
models.append(('RF', RandomForestRegressor()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('SVR', MultiOutputRegressor(SVR(gamma='auto'))))


# Neural Network algorithms
# create keras Sequential model
def baseline_model():
    model = Sequential()
    model.add(Dense(300, input_shape = (8, ), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(200, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(100, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(2))
    # Compile model
    model.compile(optimizer = 'adam',
                loss = 'mean_squared_error',
                metrics=['mse'])
    return model

# Build model
model_NN = baseline_model()
models.append(('NN', model_NN))

# Define a callback for early stopping if the validation loss does not improve for 10 consecutive
# epochs (patience=10).
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

results = []
alg_names = []

# Initialize the KerasRegressor estimator with early stopping
estimator = KerasRegressor(model=model_NN, batch_size=128, verbose=1, callbacks=[early_stopping])

for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)

    # modify loop for NN
    if name != 'NN':
        cvs = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    else:
        #validation_data = (X_val, y_val)
        cvs = cross_val_score(estimator, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')

    results.append(cvs)
    alg_names.append(name)
    output = "%s: %f (%f)" % (name, cvs.mean(), cvs.std())
    print(output)

# b) Compare algorithms and the NN model
# Distribution of accuracy values calculated across 8 cross-validation folds.
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot()
plt.boxplot(results)
ax.set_xticklabels(alg_names)
plt.ylabel("MSE")
plt.show()

# Evaluate performance of Random Forest algorithm on validation data
model_RF = RandomForestRegressor()
model_RF.fit(X_train, y_train)
y_pred_RF = model_RF.predict(X_val)
mae_pred_RF = mean_absolute_error(y_val, y_pred_RF)
print("Mean Absolute Error of predicted data: ", mae_pred_RF)

# Evaluate performance of NN algorithm on validation data
model_NN.fit(X_train, y_train)
y_pred_NN = model_NN.predict(X_val)
print(y_pred_NN.shape)
mae_pred_NN = mean_absolute_error(y_val, y_pred_NN)
print("Mean Absolute Error of predicted data: ", mae_pred_NN)


# Define baseline mean_absolute_error of y_val in the data set
y_mean = y_val.mean().values  # Calculate the mean for each output column
y_mean = np.tile(y_mean, (len(y_val), 1))
mae_ori = mean_absolute_error(y_val, y_mean)
print("Mean Absolute Error of original data: ", mae_ori)


# a) Get best model parameters
model_params = model_RF.get_params()

# Print the model's parameters
print(model_params)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot the actual and predicted heating load
ax1.bar(np.arange(len(y_val)), y_val.iloc[:, 0], label='Actual heating load', color='orange')
ax1.scatter(np.arange(len(y_pred_RF)), y_pred_RF[:, 0], label='Predicted data', color='red')
ax1.set_ylabel('Heating Load')
ax1.legend()

# Show the plot
plt.show()

# b) Save model for later use
# save the model to disk
with open('scaler_file.sav', 'wb') as scaler_file:
    dump(scaler, scaler_file)
filename = 'finalized_model_multivariate.sav'
dump(model_RF, open(filename, 'wb'))

"""
if __name__ == "__main__":
    # Adjust these based on your dataset
    dataset_path = 'ResOverallData2_encoded.csv'
    output_columns = ['HeatingEnergykwh', 'CoolingEnergy']  # Adjust output columns as needed
"""
