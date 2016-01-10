from sklearn.linear_model import LinearRegression
from pandas import read_csv
import numpy as np


def get_data(df_train, features, output):
    return df_train[features], df_train[[output]]
           

def get_rss(model, X, y):
    return np.sum((model.predict(X) - y) ** 2)


df_train = read_csv("kc_house_train_data.csv")
df_test  = read_csv("kc_house_test_data.csv")



print("---------- model 1 ----------")

features, output = ['sqft_living'], 'price'

X_train, y_train = get_data(df_train, features, output)
X_test, y_test   = get_data(df_test, features, output)

model1 = LinearRegression()
model1.fit(X_train, y_train)

print("coef:        %f" % model1.coef_)
print("intercept:   %f" % model1.intercept_)
print("prediction:  %f" % model1.predict(2650))
print("train rss:   %e" % get_rss(model1, X_train, y_train))
print("test rss:    %e" % get_rss(model1, X_test, y_test))


print("---------- model 2 ----------")

features, output = ['bedrooms'], 'price'

X_train, y_train = get_data(df_train, features, output)
X_test, y_test = get_data(df_test, features, output)

model2 = LinearRegression()
model2.fit(X_train, y_train)

print("test rss:    %e" % get_rss(model2, X_test, y_test))