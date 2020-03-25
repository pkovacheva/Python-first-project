import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib

# Load the dataset
file_name = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv'
df_houses = pd.read_csv(file_name)

# Clean the data
df_houses.drop("id", axis = 1, inplace=True)
df_houses.drop("Unnamed: 0", axis = 1, inplace=True)
df_houses.drop("zipcode", axis = 1, inplace=True)
df_houses['date'] =  pd.to_datetime(df_houses['date'], infer_datetime_format=True)
mean=df_houses['bathrooms'].mean()
df_houses['bathrooms'].replace(np.nan,mean, inplace=True)
mean=df_houses['bedrooms'].mean()
df_houses['bedrooms'].replace(np.nan,mean, inplace=True)
df_houses['sqm_living'] = df_houses['sqft_living']*0.09290304
# print(df_houses['sqft_living'].head())

# Fit the model into the hall dataset
features = ['sqm_living']
# print('Fit model and coefficient of determination')
X = df_houses[features]
Y = df_houses['price']
lm = LinearRegression()
lm.fit(X, Y)
# print(lm.score(X))


# Save the model as a pickle in a file
joblib.dump(lm, 'LR_ser.pkl')
joblib.dump(df_houses, "training_data.pkl")

# Load the model from the file
# lm_from_joblib = joblib.load('LR_ser.pkl')

# Use the loaded model to make predictions
# print(lm_from_joblib.predict([[2000]]))
