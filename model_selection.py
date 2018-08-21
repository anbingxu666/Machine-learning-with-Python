import numpy as np

def train_test_split(X,y,rate=0.2,seed=None):
    "split data set into train data set and test data set"

    assert X.shape[0]!=0 and y.shape[0]!=0 and X.shape[0]== y.shape[0],"X and y must be valid"
    assert 0.0<rate<1,"rate must be valid"

    if seed:
        np.random.seed(seed)
    random_index = np.random.permutation(len(X))
    test_size = int(rate*len(X))
    x_test = X[random_index[:test_size]]
    y_test = y[random_index[:test_size]]
    x_train = X[random_index[test_size:]]
    y_train = y[random_index[test_size:]]

    return x_train,y_train,x_test,y_test