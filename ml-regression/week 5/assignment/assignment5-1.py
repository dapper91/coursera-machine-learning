import pandas as pd
import numpy as np
from math import log, sqrt
from sklearn.linear_model import Lasso
from sklearn.cross_validation import PredefinedSplit
from sklearn.grid_search import GridSearchCV


DATA_PATH = '../../data/'


def add_features(df):    
    df['sqft_living_sqrt'] = df['sqft_living'].apply(sqrt)
    df['sqft_lot_sqrt'] = df['sqft_lot'].apply(sqrt)
    df['bedrooms_square'] = df['bedrooms']*df['bedrooms']
    df['floors_square'] = df['floors']*df['floors']


def rss_scorer(estimator, X, y):
    return np.sum((estimator.predict(X) - y) ** 2)


def nonzero_coefs(model):
    return np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_)


dtype_dict = {
    'bathrooms':        float,
    'waterfront':       int,
    'sqft_above':       int,
    'sqft_living15':    float,
    'grade':            int,
    'yr_renovated':     int,
    'price':            float,
    'bedrooms':         float,
    'zipcode':          str,
    'long':             float,
    'sqft_lot15':       float,
    'sqft_living':      float,
    'floors':           float,
    'condition':        int,
    'lat':              float,
    'date':             str,
    'sqft_basement':    int,
    'yr_built':         int,
    'id':               str,
    'sqft_lot':         int,
    'view':             int
}

df = pd.read_csv(DATA_PATH + 'kc_house_data.csv.gz', dtype=dtype_dict, compression = "gzip")
add_features(df)



print("==================== Queston 1 ====================")

features = [
    'bedrooms', 'bedrooms_square',
    'bathrooms',
    'sqft_living', 'sqft_living_sqrt',
    'sqft_lot', 'sqft_lot_sqrt',
    'floors', 'floors_square',
    'waterfront', 'view', 'condition', 'grade',
    'sqft_above',
    'sqft_basement',
    'yr_built', 'yr_renovated'
]

output = 'price'


model = Lasso(alpha = 5e2, normalize = True)
model.fit(df[features], df[output])

print("NONZERO FEATURES:")
for coef, feature in zip(model.coef_, features):
    if coef != 0:
        print("%20s: %8.2f" % (feature, coef))




print("==================== Queston 2 ====================")

df_train = pd.read_csv(DATA_PATH + 'wk3_kc_house_train_data.csv.gz', dtype = dtype_dict, compression = "gzip")
add_features(df_train)

df_valid = pd.read_csv(DATA_PATH + 'wk3_kc_house_valid_data.csv.gz', dtype = dtype_dict, compression = "gzip")
add_features(df_valid)

df_test  = pd.read_csv(DATA_PATH + 'wk3_kc_house_test_data.csv.gz',  dtype = dtype_dict, compression = "gzip")
add_features(df_test)




# df_train_valid = pd.concat([df_train, df_valid])

# l1_penalty = np.logspace(1, 7, num=13)
# parameters = {
#   'alpha': l1_penalty
# }

# ps = PredefinedSplit([-1] * len(df_train) + [1] * len(df_valid))
# model = Lasso(normalize = True)

# classifier = GridSearchCV(model, parameters, scoring = rss_scorer, cv = ps)
# classifier.fit(df_train_valid[features], df_train_valid[output])


best_score, best_alpha = float('Inf'), None
for alpha in np.logspace(1, 7, num = 13):
    model = Lasso(alpha = alpha, normalize = True)
    model.fit(df_train[features], df_train[output])

    score = rss_scorer(model, df_valid[features], df_valid[output])

    if score < best_score:
        best_score = score
        best_alpha = alpha

print("best score: %e" % best_score)
print("best alpha: %f" % best_alpha)




print("==================== Queston 3 ====================")

model = Lasso(alpha = best_alpha, normalize = True)
model.fit(df_train[features], df_train[output])
rss_test = rss_scorer(model, df_test[features], df_test[output])

print("TEST SCORE: %e" % rss_test)
print("NONZERO COEFFICIENTS: %d" % nonzero_coefs(model))



print("==================== Queston 4 ====================")

max_nonzeros = 7

l1_penalty_min, l1_penalty_max = 0, 0

for l1_penalty in np.logspace(1, 4, num = 20):
    model = Lasso(alpha = l1_penalty, normalize = True)
    model.fit(df_train[features], df_train[output])

    if nonzero_coefs(model) > max_nonzeros:
        l1_penalty_min = l1_penalty

    if nonzero_coefs(model) < max_nonzeros:
        l1_penalty_max = l1_penalty
        break
        
print("L1 mix:%f, L2 max:%f" % (l1_penalty_min, l1_penalty_max))




print("==================== Queston 5 ====================")


best_score, best_l1 = float("inf"), None
for l1_penalty in np.linspace(l1_penalty_min, l1_penalty_max, 20):
    model = Lasso(alpha = l1_penalty, normalize = True)
    model.fit(df_train[features], df_train[output])

    score = rss_scorer(model, df_valid[features], df_valid[output])

    if nonzero_coefs(model) == max_nonzeros and score < best_score:
        best_score = score
        best_l1 = l1_penalty


print("best score: %e, best L1: %.2f" % (best_score, best_l1))


print("==================== Queston 6 ====================")

model = Lasso(alpha = best_l1, normalize = True)
model.fit(df_train[features], df_train[output])


print("NONZERO FEATURES:")
for coef, feature in zip(model.coef_, features):
    if coef != 0:
        print("%20s: %8.2f" % (feature, coef))
