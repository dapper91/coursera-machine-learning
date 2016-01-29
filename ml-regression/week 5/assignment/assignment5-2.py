import pandas as pd
import numpy as np
import math
import sys


DATA_PATH = '../../data/'


def get_numpy_data(df, features, output):
    df['constant'] = 1.0
    return df.as_matrix(['constant'] + features), df.as_matrix([output])[:, 0]


def predict_output(feature_matrix, weights):
    return feature_matrix.dot(weights)


def normalize_features(X):
    norms = np.linalg.norm(X, axis = 0)
    return X/norms, norms


def rss_scorer(estimator, weights, X, y):
    return np.sum((estimator(X, weights) - y) ** 2)


def print_features(features, weights):
    print("FEATURES:")
    for weight, feature in zip(weights, features):
        print("%15s: %15.2f" % (feature, weight))


def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):    
    prediction = predict_output(feature_matrix, weights)
    
    ro_i = np.sum(feature_matrix[:,i]*(output - prediction + weights[i]*feature_matrix[:,i]))
    
    if i == 0:
        new_weight_i = ro_i
    elif ro_i < -l1_penalty/2.:
        new_weight_i = ro_i + l1_penalty/2
    elif ro_i > l1_penalty/2.:
        new_weight_i = ro_i - l1_penalty/2
    else:
        new_weight_i = 0.
    
    return new_weight_i


def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    new_weights = np.array(initial_weights)

    while True:
        weights = np.array(new_weights)

        for i in range(feature_matrix.shape[1]):
            new_weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, new_weights, l1_penalty)
        
        if all(abs(new_weights - weights) < tolerance):
            return new_weights   




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

df_all   = pd.read_csv(DATA_PATH + 'kc_house_data.csv.gz', dtype=dtype_dict, compression = "gzip")
df_train = pd.read_csv(DATA_PATH + 'kc_house_train_data.csv.gz', dtype=dtype_dict, compression = "gzip")
df_test  = pd.read_csv(DATA_PATH + 'kc_house_test_data.csv.gz', dtype=dtype_dict, compression = "gzip")



print("====================== Test =======================")

expected = 0.425558846691
actual =  lasso_coordinate_descent_step(1, np.array([[3./math.sqrt(13),1./math.sqrt(10)], 
                                                     [2./math.sqrt(13),3./math.sqrt(10)]]), 
                                        np.array([1., 1.]), np.array([1., 4.]), 0.1)

if expected != round(actual, 12):
    sys.exit("test error!\nexpected: %.12f, actual: %.12f" %(expected, actual))

print("test passed")



print("=================== Queston 1,2 ===================")

features, output = ['sqft_living', 'bedrooms'], 'price'

X_all, y_all = get_numpy_data(df_all, features, output)
X_all, norm_all = normalize_features(X_all)

initial_weights = np.array([1., 4., 1.])

for l1_penalty in [1.4e8, 1.64e8, 1.73e8, 1.9e8, 2.3e8]:
    w_1 = lasso_coordinate_descent_step(1, X_all, y_all, initial_weights, l1_penalty)
    w_2 = lasso_coordinate_descent_step(2, X_all, y_all, initial_weights, l1_penalty)
    print("%e: w_1 = %e, w_2 = %e" % (l1_penalty, w_1, w_2))




print("=================== Queston 3,4 ===================")

features, output = ['sqft_living', 'bedrooms'], 'price'

initial_weights = np.array([0., 0., 0.])
l1_penalty = 1e7
tolerance = 1.0

weights = lasso_cyclical_coordinate_descent(X_all, y_all, initial_weights, l1_penalty, tolerance)

rss = rss_scorer(predict_output, weights, X_all, y_all)

print("rss: %e" % rss)
print_features(['constant'] + features, weights)




print("================== Queston 5,6,7 ==================")

features = [
    'bedrooms', 'bathrooms',
    'sqft_living', 'sqft_lot',
    'floors',
    'waterfront', 'view',
    'condition', 'grade',
    'sqft_above', 'sqft_basement',
    'yr_built', 'yr_renovated']

output = 'price'

X_train, y_train = get_numpy_data(df_train, features, output)
X_train, norms_train = normalize_features(X_train)



print("L1 = 1e7")
initial_weights = np.array([0.] * X_train.shape[1])
l1_penalty = 1e7
tolerance = 1.0
weights1 = lasso_cyclical_coordinate_descent(X_train, y_train, initial_weights, l1_penalty, tolerance)
print_features(['constant'] + features, weights1)


print("L1 = 1e8")
initial_weights = np.array([0.] * X_train.shape[1])
l1_penalty = 1e8
tolerance = 1.0
weights2 = lasso_cyclical_coordinate_descent(X_train, y_train, initial_weights, l1_penalty, tolerance)
print_features(['constant'] + features, weights2)


print("L1 = 1e4")
initial_weights = np.array([0.] * X_train.shape[1])
l1_penalty = 1e4
tolerance = 5e5
weights3 = lasso_cyclical_coordinate_descent(X_train, y_train, initial_weights, l1_penalty, tolerance)
print_features(['constant'] + features, weights3)



print("==================== Queston 8 ====================")

X_test,  y_test  = get_numpy_data(df_test, features, output)
X_test, norms = normalize_features(X_test)

weights1_normalized = weights1 / norms_train
weights2_normalized = weights2 / norms_train
weights3_normalized = weights3 / norms_train

print("L1 = 1e7 rss: %e" % rss_scorer(predict_output, weights1_normalized, X_test, y_test))
print("L1 = 1e8 rss: %e" % rss_scorer(predict_output, weights2_normalized, X_test, y_test))
print("L1 = 1e4 rss: %e" % rss_scorer(predict_output, weights3_normalized, X_test, y_test))