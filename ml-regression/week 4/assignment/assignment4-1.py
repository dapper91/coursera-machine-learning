from sklearn.linear_model import Ridge, RidgeCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DATA_PATH = '../../data/'



def polynomial_features(feature, degree):
    return pd.DataFrame( {'power_' + str(pow) : feature ** pow for pow in range(1, degree + 1)} )


def fit_and_print(filename, penalty):
    df = pd.read_csv(DATA_PATH + filename, compression = 'gzip', dtype = {'sqft_living': np.float64}).sort(['sqft_living','price'])

    X, y = polynomial_features(df['sqft_living'], 15), df['price']

    model = Ridge(alpha = penalty, normalize = True)
    model.fit(X, y)

    print("power_1 coef: %d" % model.coef_[0])


def rss_scorer(estimator, X, y):
    return np.sum((estimator.predict(X) - y) ** 2)




print("==================== Question 1 ====================")

df = pd.read_csv(DATA_PATH + 'kc_house_data.csv.gz', compression = 'gzip', dtype = {'sqft_living': np.float64}).sort(['sqft_living','price'])

X, y = polynomial_features(df.ix[:,'sqft_living'], 15), df['price']
l2_penalty = 1.5e-5

model = Ridge(alpha = l2_penalty, normalize = True)
model.fit(X, y)

print("power_1 coef: %d" % model.coef_[0])



print("=================== Question 2,3 ===================")

l2_small_penalty = 1e-9

print("SET_1")
fit_and_print('wk3_kc_house_set_1_data.csv.gz', l2_small_penalty)

print("SET_2")
fit_and_print('wk3_kc_house_set_2_data.csv.gz', l2_small_penalty)

print("SET_3")
fit_and_print('wk3_kc_house_set_3_data.csv.gz', l2_small_penalty)

print("SET_4")
fit_and_print('wk3_kc_house_set_4_data.csv.gz', l2_small_penalty)



print("=================== Question 4,5 ===================")

l2_large_penalty = 1.23e2

print("SET_1")
fit_and_print('wk3_kc_house_set_1_data.csv.gz', l2_large_penalty)

print("SET_2")
fit_and_print('wk3_kc_house_set_2_data.csv.gz', l2_large_penalty)

print("SET_3")
fit_and_print('wk3_kc_house_set_3_data.csv.gz', l2_large_penalty)

print("SET_4")
fit_and_print('wk3_kc_house_set_4_data.csv.gz', l2_large_penalty)



print("==================== Question 6 ====================")

df = pd.read_csv(DATA_PATH + 'wk3_kc_house_train_valid_shuffled.csv.gz', compression = 'gzip', dtype = {'sqft_living': np.float64}).sort(['sqft_living','price'])

X, y = polynomial_features(df['sqft_living'], 15), df['price']

model_cv = RidgeCV(alphas = np.logspace(3, 9, num=13), normalize = True, cv = 10, scoring = rss_scorer)
model_cv.fit(X, y)
best_alpha = model_cv.alpha_

print("best alpha: %e" % best_alpha)




print("==================== Question 7 ====================")

df_train = pd.read_csv(DATA_PATH + 'wk3_kc_house_train_data.csv.gz', compression = 'gzip', dtype = {'sqft_living': np.float64}).sort(['sqft_living','price'])
df_test  = pd.read_csv(DATA_PATH + 'wk3_kc_house_test_data.csv.gz',  compression = 'gzip', dtype = {'sqft_living': np.float64}).sort(['sqft_living','price'])

X_train, y_train = polynomial_features(df_train['sqft_living'], 15), df_train['price']
X_test, y_test   = polynomial_features(df_test['sqft_living'],  15), df_test['price']

model = Ridge(alpha = best_alpha, normalize = True)
model.fit(X_train, y_train)

print("test rss: %e" % rss_scorer(model, X_test, y_test))