from sklearn.linear_model import LinearRegression
from pandas import read_csv
import numpy as np


def get_data(df, features, output):
    return df[features], df[[output]]

def get_rss(model, X, y):
    return np.sum((model.predict(X) - y) ** 2)

def add_features(df):
    df['bedrooms_squared'] = df['bedrooms'] * df['bedrooms']
    df['bed_bath_rooms']   = df['bedrooms'] * df['bathrooms']
    df['lat_plus_long']    = df['lat'] + df['long']
    df['log_sqft_living']  = np.log(df['sqft_living'])

def fit_and_print(df_train, df_test, features, output):    
    X_train, y_train = get_data(df_train, features, output)
    X_test, y_test = get_data(df_test, features, output)

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("coefficients: " + ', '.join([str(i) for i in model.coef_[0]]))
    print("train rss:    %f" % get_rss(model, X_train, y_train))
    print("test rss:     %f" % get_rss(model, X_test, y_test))


df_train = read_csv('kc_house_train_data.csv')
add_features(df_train)

df_test  = read_csv('kc_house_test_data.csv')
add_features(df_test)


print("---------- model 0 ----------")

features, output = list(set(df_train.axes[1]) - set(['price'])), 'price'
X_train, y_train = get_data(df_train, features, output)
X_test, y_test = get_data(df_test, features, output)

print("bedrooms_squared avg:    %.2f" % np.mean(X_test['bedrooms_squared']))
print("bed_bath_rooms avg:      %.2f" % np.mean(X_test['bed_bath_rooms']))
print("log_sqft_living avg:     %.2f" % np.mean(X_test['log_sqft_living']))
print("lat_plus_long avg:       %.2f" % np.mean(X_test['lat_plus_long']))


print("---------- model 1 ----------")
fit_and_print(df_train, df_test, ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long'], 'price')


print("---------- model 2 ----------")
fit_and_print(df_train, df_test, ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms'], 'price')


print("---------- model 3 ----------")
fit_and_print(df_train, df_test, ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long', 'bed_bath_rooms', 'bedrooms_squared', 'log_sqft_living', 'lat_plus_long'], 'price')