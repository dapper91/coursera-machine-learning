from sklearn.linear_model import LinearRegression
from pandas import read_csv
import numpy as np


DATA_PATH = '../../data/'


def rss_scorer(estimator, X, y):
    return np.sum((estimator.predict(X) - y) ** 2)

def add_features(df):
    df['bedrooms_squared'] = df['bedrooms'] * df['bedrooms']
    df['bed_bath_rooms']   = df['bedrooms'] * df['bathrooms']
    df['lat_plus_long']    = df['lat'] + df['long']
    df['log_sqft_living']  = np.log(df['sqft_living'])

def fit_and_print(df_train, df_test, features, output):
    X_train, y_train = df_train[features], df_train[output]
    X_test, y_test = df_test[features], df_test[output]

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("COEFFICIENTS:")
    for feature, coef in zip(features, model.coef_):
        print("%-20s%15e" % (feature, coef))

    print("")

    print("RSS:")
    print("train    %e" % rss_scorer(model, X_train, y_train))
    print("test     %e" % rss_scorer(model, X_test, y_test))


df_train = read_csv(DATA_PATH + 'kc_house_train_data.csv.gz', compression = "gzip")
add_features(df_train)

df_test  = read_csv(DATA_PATH + 'kc_house_test_data.csv.gz', compression = "gzip")
add_features(df_test)


print("================== Question 1,2,3 ==================")

features, output = list(set(df_train.axes[1]) - set(['price'])), 'price'

X_train, y_train = df_train[features], df_train[output]
X_test, y_test   = df_test[features],  df_test[output]

print("bedrooms_squared avg:    %10.2f" % np.mean(X_test['bedrooms_squared']))
print("bed_bath_rooms avg:      %10.2f" % np.mean(X_test['bed_bath_rooms']))
print("log_sqft_living avg:     %10.2f" % np.mean(X_test['log_sqft_living']))
print("lat_plus_long avg:       %10.2f" % np.mean(X_test['lat_plus_long']))


print("==================== Question 5 ====================")
fit_and_print(df_train, df_test, ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long'], 'price')


print("==================== Question 6 ====================")
fit_and_print(df_train, df_test, ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms'], 'price')


print("=================== Question 7,8 ===================")
fit_and_print(df_train, df_test, ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms', 'bedrooms_squared', 'log_sqft_living', 'lat_plus_long'], 'price')