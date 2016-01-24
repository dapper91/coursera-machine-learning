from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import math


def rmse_scorer(estimator, X, y):
    return math.sqrt(np.mean((estimator.predict(X) - y) ** 2))


def fit_and_score(df_train, df_test, features, output):
    X_train, y_train = df_train[features], df_train[output]
    X_train, y_train = df_train[features], df_train[output]

    X_test, y_test = df_test[features], df_test[output]
    X_test, y_test = df_test[features], df_test[output]

    model = LinearRegression()
    model.fit(X_train, y_train)

    return rmse_scorer(model, X_test, y_test)



df = pd.read_csv('home_data.csv.gz', compression = 'gzip')
df_train = pd.read_csv('home_data_train.csv.gz', compression = 'gzip')
df_test  = pd.read_csv('home_data_test.csv.gz', compression = 'gzip')



print("---------- Question 1 ----------")

max_avg_zip = df.groupby(['zipcode']).mean()['price'].max()
print("max average house price by zip: %d" % max_avg_zip)




print("---------- Question 2 ----------")

filtered_num = df[(2000 < df['sqft_living']) & (df['sqft_living'] < 4000)].shape[0]
total_num = df.shape[0]
print("fraction: %f" % (float(filtered_num) / total_num) )




print("---------- Question 3 ----------")

my_features, output = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode'], 'price'
rmse1 = fit_and_score(df_train, df_test, my_features, output)



advanced_features = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
    'condition',        # condition of house
    'grade',            # measure of quality of construction
    'waterfront',       # waterfront property
    'view',             # type of view
    'sqft_above',       # square feet above ground
    'sqft_basement',    # square feet in basement
    'yr_built',         # the year built
    'yr_renovated',     # the year renovated
    'lat', 'long',      # the lat-long of the parcel
    'sqft_living15',    # average sq.ft. of 15 nearest neighbors
    'sqft_lot15',       # average lot size of 15 nearest neighbors
]
output = 'price'
rmse2 = fit_and_score(df_train, df_test, advanced_features, output)


print("diff: %.0f" % (rmse1 - rmse2))