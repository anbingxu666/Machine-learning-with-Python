import numpy as np
def accuracy_calculate(y_true,y_test):
    '''计算准确率accuracy'''
    assert y_true.shape[0]== y_test.shape[0],"y_true and y_test must be valid"
    return np.sum(y_true==y_test) / len(y_true)
