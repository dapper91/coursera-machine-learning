from sklearn.linear_model import LinearRegression
from pandas import read_csv
import numpy as np


DATA_PATH = '../../data/'


def rss_scorer(estimator, X, y):
    return np.sum((estimator.predict(X) - y) ** 2)


df_train = read_csv(DATA_PATH + 'kc_house_train_data.csv.gz', compression = 'gzip')
df_test  = read_csv(DATA_PATH + 'kc_house_test_data.csv.gz', compression = 'gzip')



print("==================== Model 1 ====================")

features, output = ['sqft_living'], 'price'

X_train, y_train = df_train[features], df_train[output]
X_test,  y_test  = df_test[features],  df_test[output]

model1 = LinearRegression()
model1.fit(X_train, y_train)

print("coef:      %15.2f" % model1.coef_)
print("intercept: %15.2f" % model1.intercept_)
print("train rss: %15e" % rss_scorer(model1, X_train, y_train))
print("test rss:  %15e" % rss_scorer(model1, X_test, y_test))

print("")
print("predictions:")
print("2650 sq. feets apartment costs %.2f$" % model1.predict(2650))
print("for 800000$ you can buy %.2f sq. feets apartment" % ((800000 - model1.intercept_)/model1.coef_))



print("==================== Model 2 ====================")

features, output = ['bedrooms'], 'price'

X_train, y_train = df_train[features], df_train[output]
X_test, y_test   = df_test[features],  df_test[output]

model2 = LinearRegression()
model2.fit(X_train, y_train)

print("test rss:   %15e" % rss_scorer(model2, X_test, y_test))