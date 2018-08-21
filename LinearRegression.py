import numpy as np
from metrics import r2_score


class LinearRegression():
    def __init__(self):
        self._theta = None
        self.cofficients_ = None
        self.intercept_ = None

    def fit_normal(self,X_train,y_train):
        """通过训练数据 fit模型参数"""
        X_temp = np.hstack([np.ones((X_train.shape[0],1)),X_train])
        self._theta = np.linalg.inv((X_temp.T).dot(X_temp)).dot(X_temp.T).dot(y_train)
        self.cofficients_ = self._theta[1:]
        self.intercept_ = self._theta[:1]
        return self
    def predict(self,X):
        """给定X 预测y的值"""
        X = np.hstack([np.ones((X.shape[0],1)),X])
        predict_y = X.dot(self._theta)
        return predict_y
    def __repr__(self):
        return "LinearRegression()  with normal_equation"

    def score(self,X_test,y_test):
        """根据给定的测试数据集 计算R Square"""
        res_y = self.predict(X_test)
        return r2_score(y_test,res_y)