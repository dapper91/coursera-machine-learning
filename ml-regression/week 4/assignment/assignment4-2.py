from sklearn.linear_model import Ridge
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DATA_PATH = '../../data/'



def get_numpy_data(df, features, output):
    df['constant'] = 1.0
    return df.as_matrix(['constant'] + features), df.as_matrix([output])[:, 0]

def predict_outcome(feature_matrix, weights):
    return feature_matrix.dot(weights)

def feature_derivative(error, feature):
    return -2 * feature.dot(error)

def ridge_regression_gradient_descent(feature_matrix, output_vector, initial_weights, step_size, l2_penalty, tolerance, max_iterations, feature0_is_constant = True):
    converged = False
    iters = 0
    weights = np.array(initial_weights)

    while not converged:
    	iters += 1
        prediction = predict_outcome(feature_matrix, weights)
        error = output_vector - prediction
        gradient_sum_squares = 0

        for i in range(len(weights)):
            derivative = feature_derivative(error, feature_matrix[:, i])
            
            if i == 0 and feature0_is_constant:            	
            	derivative += 0
            else:
            	derivative += 2 * l2_penalty * weights[i]

            weights[i] = weights[i] - step_size * derivative

        gradient_sum_squares += (derivative ** 2)
        gradient_magnitude = sqrt(gradient_sum_squares)

        if gradient_magnitude < tolerance or iters == max_iterations:
            converged = True

    return weights



def rss_scorer(real, estimated):
    return np.sum((real - estimated) ** 2)



df_train = pd.read_csv(DATA_PATH + 'kc_house_train_data.csv.gz', compression = 'gzip')
df_test  = pd.read_csv(DATA_PATH + 'kc_house_test_data.csv.gz',  compression = 'gzip')



print("---------- Queston 1,2,3 ----------")

features, output = ['sqft_living'], 'price'
feature_matrix, output_vector = get_numpy_data(df_train, features, output)

plt.plot(feature_matrix[:,1], output_vector, 'k.')

step_size = 1e-12
max_iterations = 1000
initial_weights = np.array([0., 0.])


print("low penalty")
l2_penalty = 0.0
weights_0_penalty = ridge_regression_gradient_descent(feature_matrix, output_vector, initial_weights, step_size, l2_penalty, None, max_iterations)
print("weight1: %.1f" % weights_0_penalty[1])

plt.plot(feature_matrix[:,1], predict_outcome(feature_matrix, weights_0_penalty),'b-', label = 'low l2')


print("high penalty")
l2_penalty = 1e11
weights_high_penalty = ridge_regression_gradient_descent(feature_matrix, output_vector, initial_weights, step_size, l2_penalty, None, max_iterations)
print("weight1: %.1f" % weights_high_penalty[1])

plt.plot(feature_matrix[:,1], predict_outcome(feature_matrix, weights_high_penalty),'r-', label = 'high l2')


plt.legend()
plt.show()




print("---------- Queston 4 ----------")

features, output = ['sqft_living'], 'price'
feature_matrix, output_vector = get_numpy_data(df_test, features, output)

print("rss: %e" % rss_scorer(predict_outcome(feature_matrix, weights_0_penalty), output_vector))




print("---------- Queston 5 ----------")

features, output = ['sqft_living', 'sqft_living15'], 'price'
feature_matrix_train, output_vector_train = get_numpy_data(df_train, features, output)
feature_matrix_test , output_vector_test  = get_numpy_data(df_test, features, output)

step_size = 1e-12
max_iterations = 1000
initial_weights = np.array([0., 0., 0.])

print("low penalty")
l2_penalty = 0.0
weights_0_penalty = ridge_regression_gradient_descent(feature_matrix_train, output_vector_train, initial_weights, step_size, l2_penalty, None, max_iterations)
print("weight1: %.1f" % weights_0_penalty[1])




print("---------- Queston 6 ----------")

print("high penalty")
l2_penalty = 1e11
weights_high_penalty = ridge_regression_gradient_descent(feature_matrix_train, output_vector_train, initial_weights, step_size, l2_penalty, None, max_iterations)
print("weight1: %.1f" % weights_high_penalty[1])




print("---------- Queston 7 ----------")

print("rss: %e" % rss_scorer(predict_outcome(feature_matrix_test, weights_high_penalty), output_vector_test))




print("---------- Queston 8 ----------")

print("real price: %e" % df_test['price'][0])
print("low penalty prediction:  %e" % predict_outcome(feature_matrix_test, weights_0_penalty)[0])
print("high penalty prediction: %e" % predict_outcome(feature_matrix_test, weights_high_penalty)[0])