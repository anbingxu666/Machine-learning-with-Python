import numpy as np
from metrics import r2_score

class SimpleLinearRegression1:

    def __init__(self):
        self.w_ = None
        self.b_ = None

    def fit(self,x,y):
        """根据训练数据集x_train,y_train训练Simple Linear Regression模型"""
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        #高度向量化  尽量不要使用np.sum()之类的东西
        #两个向量相乘  矩阵乘法 就是相乘并相加
        self.w_ =((x - mean_x).dot(y - mean_y)) / (x - mean_x).dot(x - mean_x)
        self.b_ = mean_y - self.w_ * mean_x
        return self

    def predict(self,X):
        """给定待预测数据集x_predict，返回表示x_predict的结果向量"""
        assert X.ndim == 1, \
            "Simple Linear Regression can only solve single feature training data."
        assert self.w_ is not None and self.b_ is not None, \
            "must fit before predict!"
        predict_y = self.w_ * X + self.b_
        return predict_y

    def score(self,x_test,y_test):
        """根据测试数据返回模型的准确度"""
        y_predict = self.predict(x_test)
        return r2_score(y_test,y_predict)

    def __repr__(self):
        return "SimpleLinearRegression1()"


class SimpleLinearRegression2:

    def __init__(self):
        """初始化Simple Linear Regression 模型"""
        self.w_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """根据训练数据集x_train,y_train训练Simple Linear Regression模型"""
        assert x_train.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0
        for x, y in zip(x_train, y_train):
            num += (x - x_mean) * (y - y_mean)
            d += (x - x_mean) ** 2

        self.w_ = num / d
        self.b_ = y_mean - self.w_ * x_mean

        return self

    def predict(self, x_predict):
        """给定待预测数据集x_predict，返回表示x_predict的结果向量"""
        assert x_predict.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert self.w_ is not None and self.b_ is not None, \
            "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """给定单个待预测数据x，返回x的预测结果值"""
        return self.w_ * x_single + self.b_


    def score(self,x_test,y_test):
        """根据测试数据返回模型的准确度"""
        y_predict = self.predict(x_test)
        return r2_score(y_test,y_predict)



    def __repr__(self):
        return "SimpleLinearRegression1()"
