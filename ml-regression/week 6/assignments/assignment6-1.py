import pandas as pd
import numpy as np
import math


DATA_PATH = '../../data/'



def get_numpy_data(df, features, output):
    df['constant'] = 1.0
    return df.as_matrix(['constant'] + features), df.as_matrix([output])[:, 0]

def normalize_features(X):
    norms = np.linalg.norm(X, axis = 0)
    return X/norms, norms

def euclidean_distance(x1, x2):    
    return math.sqrt(np.dot(x1, x1) - 2 * np.dot(x1, x2) + np.dot(x2, x2))

def compute_distances(features_instances, features_query):
    return np.sum((features_instances - features_query)**2, axis = 1)

def k_nearest_neighbors(k, feature_train, features_query):
    return np.argsort(compute_distances(feature_train, features_query))[0:k]

def predict_output(k, features_train, output_train, features_query):
    return [np.mean(output_train[k_nearest_neighbors(k, features_train, feature_query)]) for feature_query in features_query]




dtype_dict = {
    'bathrooms':        float,
    'waterfront':       int,
    'sqft_above':       int,
    'sqft_living15':    float,
    'grade':            int,
    'yr_renovated':     int,
    'price':            float,
    'bedrooms':         float,
    'zipcode':          int,
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

df_all   = pd.read_csv(DATA_PATH + 'kc_house_data_small.csv.gz', dtype = dtype_dict, compression = "gzip")
df_train = pd.read_csv(DATA_PATH + 'kc_house_data_small_train.csv.gz', dtype = dtype_dict, compression = "gzip")
df_valid = pd.read_csv(DATA_PATH + 'kc_house_data_small_validation.csv.gz', dtype = dtype_dict, compression = "gzip")
df_test  = pd.read_csv(DATA_PATH + 'kc_house_data_small_test.csv.gz', dtype = dtype_dict, compression = "gzip")


features = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
    'condition',        # condition of house
    'grade',            # measure of quality of construction
    'waterfront',       # waterfront property
    'view',             # type of view
    'sqft_above',       # square feet above ground
    'sqft_basement',    # square feet in basement
    'yr_built',         # the year built
    'yr_renovated',     # the year renovated
    'lat', 'long',      # the lat-long of the parcel
    'sqft_living15',    # average sq.ft. of 15 nearest neighbors
    'sqft_lot15',       # average lot size of 15 nearest neighbors
]
output = 'price'


X_train, y_train = get_numpy_data(df_train, features, output)
X_test,  y_test  = get_numpy_data(df_test,  features, output)
X_valid, y_valid = get_numpy_data(df_valid, features, output)


X_train, norms = normalize_features(X_train)
X_test = X_test / norms
X_valid = X_valid / norms


print("==================== Queston 1 ====================")

dst = euclidean_distance(X_train[9], X_test[0])
print("euclidean distance: %.3f" % dst)



print("==================== Queston 2 ====================")

print("DISTANCES:")
for i,x in enumerate(X_train[0:10]):
    print("house %d: %f" % (i, euclidean_distance(x, X_test[0])))



print("==================== Queston 3 ====================")

closest_house = k_nearest_neighbors(1, X_train, [X_test[2]])
print("closest house: %d" % closest_house)



print("==================== Queston 4 ====================")

print("1-nn price prediction: %d" % y_train[closest_house])



print("==================== Queston 5 ====================")

closest_houses = k_nearest_neighbors(4, X_train, [X_test[2]])
print("closest houses: " + ", ".join([str(house) for house in closest_houses]))



print("==================== Queston 6 ====================")

closest_houses = predict_output(4, X_train, y_train, [X_test[2]])
print("predicted price: " + ", ".join(["%.0f" % house for house in closest_houses]))



print("==================== Queston 7 ====================")

predictions = predict_output(10, X_train, y_train, X_test[0:10])
print("first 10 houses predictions:")
for i, prediction in enumerate(predictions):
    print("%d: %.0f" % (i, prediction))



print("==================== Queston 8 ====================")

best_rss, best_k = float('Inf'), None
for k in range(1, 15):
    rss = np.sum((predict_output(k, X_train, y_train, X_valid) - y_valid) ** 2)

    if rss < best_rss:
        best_rss = rss
        best_k = k

print("valid rss: %e" % best_rss)
print("best k: %f" % best_k)

rss_test = np.sum((predict_output(best_k, X_train, y_train, X_test) - y_test) ** 2)

print("test rss: %e" % rss_test)