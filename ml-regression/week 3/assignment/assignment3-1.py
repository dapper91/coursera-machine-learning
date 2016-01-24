from sklearn.linear_model import LinearRegression
import pandas
import numpy as np
import matplotlib.pyplot as plt


DATA_PATH = '../../data/'



def polynomial_features(feature, degree):    
    return pandas.DataFrame( {'power_' + str(pow) : feature ** pow for pow in range(1, degree + 1)} )


def fit_and_print(filename, label):
    df = pandas.read_csv(DATA_PATH + filename, compression = "gzip", dtype = {'sqft_living': np.float64}).sort(['sqft_living','price'])
    X, y = polynomial_features(df['sqft_living'], 15), df['price']

    model = LinearRegression()
    model.fit(X, y)

    print("COEFFICIENTS:")
    for degree, coef in enumerate(model.coef_):
        print("power_%02d: %15e" % (degree, coef))
    print("")

    plt.plot(X['power_1'], model.predict(X),'-', label = label)


def fit_and_plot(df, feature, output, degree):
    X, y = polynomial_features(df.ix[:,feature], degree), df[[output]]
    model = LinearRegression()
    model.fit(X, y)

    plt.plot(X['power_1'], model.predict(X),'-', label = '$x^{%d}$' % degree)


def rss_scorer(estimator, X, y):
    return np.sum((estimator.predict(X) - y) ** 2)



data_all = pandas.read_csv(DATA_PATH + 'kc_house_data.csv.gz', compression = "gzip", dtype = {'sqft_living': np.float64}).sort(['sqft_living','price'])



print("---------- Question 0 ----------")

fit_and_plot(data_all, 'sqft_living', 'price', 1)
fit_and_plot(data_all, 'sqft_living', 'price', 2)
fit_and_plot(data_all, 'sqft_living', 'price', 3)
fit_and_plot(data_all, 'sqft_living', 'price', 15)

X, y = data_all['sqft_living'], data_all['price']
plt.plot(X, y,'.')

plt.legend()
plt.show()



print("---------- Question 1,2 ----------")

print("SET_1")
fit_and_print('wk3_kc_house_set_1_data.csv.gz', 'set_1')

print("SET_2")
fit_and_print('wk3_kc_house_set_2_data.csv.gz', 'set_2')

print("SET_3")
fit_and_print('wk3_kc_house_set_3_data.csv.gz', 'set_3')

print("SET_4")
fit_and_print('wk3_kc_house_set_4_data.csv.gz', 'set_4')

X, y = data_all['sqft_living'], data_all['price']
plt.plot(X, y,'.')

plt.legend()
plt.show()



print("---------- Question 3 ----------")

data_train = pandas.read_csv(DATA_PATH + 'wk3_kc_house_train_data.csv.gz', compression = "gzip", dtype = {'sqft_living': np.float64})
data_valid = pandas.read_csv(DATA_PATH + 'wk3_kc_house_valid_data.csv.gz', compression = "gzip", dtype = {'sqft_living': np.float64})
data_test  = pandas.read_csv(DATA_PATH + 'wk3_kc_house_test_data.csv.gz', compression = "gzip",  dtype = {'sqft_living': np.float64})


for degree in range(1, 16):
    X_train, y_train = polynomial_features(data_train.ix[:,'sqft_living'], degree), data_train['price']

    model = LinearRegression()
    model.fit(X_train, y_train)

    X_valid, y_valid = polynomial_features(data_valid.ix[:,'sqft_living'], degree), data_valid['price']
    
    rss = rss_scorer(model, X_valid, y_valid)    
    print("degree: %2d\trss: %e" % (degree, rss))




print("---------- Question 4 ----------")

X_train, y_train = polynomial_features(data_train.ix[:,'sqft_living'], 6), data_train['price']
X_test, y_test = polynomial_features(data_test.ix[:,'sqft_living'], 6), data_test['price']

model = LinearRegression()
model.fit(X_train, y_train)

rss = rss_scorer(model, X_test, y_test)
print("rss: %e" % rss)