import numpy as np

def accuracy_calculate(y_true,y_test):
    """计算准确率accuracy"""
    assert y_true.shape[0] == y_test.shape[0],"y_true and y_test must be valid"
    return np.sum(y_true == y_test) / len(y_true)


def mean_square_error(y_true,y_test):
    mse = np.sum(((y_true-y_test)**2)) / len(y_true)
    return mse


def root_mean_square_error(y_true,y_test):
    rmse = np.sqrt(np.sum((y_true-y_test)**2) / len(y_true))
    return rmse


def mean_absolute_error(y_true,y_test):
    mae = np.sum(np.absolute(y_true - y_test)) / len(y_true)
    return mae


def r2_score(y_true,y_predict):
    r_square = 1 - mean_square_error(y_true, y_predict) / np.var(y_true)
    return r_square
