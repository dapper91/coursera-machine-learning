from sklearn.linear_model import LinearRegression
from pandas import read_csv
from math import sqrt
import numpy as np


DATA_PATH = '../../data/'


def get_numpy_data(df, features, output):
    df['constant'] = 1.0
    return df.as_matrix(['constant'] + features), df.as_matrix([output])[:, 0]

def predict_outcome(feature_matrix, weights):
    return feature_matrix.dot(weights)

def feature_derivative(error, feature):
    return -2 * feature.dot(error)

def regression_gradient_descent(feature_matrix, output_vector, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)

    while not converged:
        prediction = predict_outcome(feature_matrix, weights)
        error = output_vector - prediction
        gradient_sum_squares = 0

        for i in range(len(weights)):
            derivative = feature_derivative(error, feature_matrix[:, i])
            gradient_sum_squares += (derivative ** 2)
            weights[i] -= step_size * derivative

        gradient_magnitude = sqrt(gradient_sum_squares)

        if gradient_magnitude < tolerance:
            converged = True

    return weights


def fit_and_print(df_train, df_test, features, output, initial_weights, step_size, tolerance):
    feature_matrix, output_vector = get_numpy_data(df_train, features, output)
    weights = regression_gradient_descent(feature_matrix, output_vector, initial_weights, step_size, tolerance)

    print("WEIGHTS:")
    for feature, weight in zip(['constant'] + features, weights):
        print("%-15s%10.1f" % (feature, weight))
    
    print("")

    feature_matrix, output_vector = get_numpy_data(df_test, features, output)
    print("1st test house prediction: " + '%.0f' % predict_outcome(feature_matrix[0, :], weights))
    print("test rss: " + '%e' % np.sum((predict_outcome(feature_matrix, weights) - output_vector) ** 2))




df_train = read_csv(DATA_PATH + 'kc_house_train_data.csv.gz', compression = "gzip")
df_test  = read_csv(DATA_PATH + 'kc_house_test_data.csv.gz', compression = "gzip")


print("1st test house true price: %.0f" % df_test['price'][0])


print("==================== Model 1 ====================")

features, output = ['sqft_living'], 'price'
initial_weights = np.array([-47000., 1.])
step_size, tolerance = 7e-12, 2.5e7

fit_and_print(df_train, df_test, features, output, initial_weights, step_size, tolerance)


print("==================== Model 2 ====================")

features, output = ['sqft_living', 'sqft_living15'], 'price'
initial_weights = np.array([-100000., 1., 1.])
step_size, tolerance = 4e-12, 1e9

fit_and_print(df_train, df_test, features, output, initial_weights, step_size, tolerance)