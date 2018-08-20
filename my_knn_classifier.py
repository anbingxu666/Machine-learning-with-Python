import numpy as np
from collections import Counter
class my_knn_classifier:
    def __init__(self,k):
        '''初始化KNN分类器'''
        assert k>=1,"k must be valid"
        self.k = k
        self._X_train = None
        self._Y_train = None
    def fit(self,training_set_x,training_set_y):
        assert training_set_y is not None and training_set_y is not None,"you must import parameters"
        self._X_train = training_set_x
        self._Y_train = training_set_y
        return self
    def predict_Y(self,X_predict):
        y_predict=[self._predict(x) for x in X_predict]
        return np.array(y_predict)
    def _predict(self,x):
        '''功能：预测单个x的y'''
        x.reshape((1,-1))
        distance = np.sqrt(np.sum(((self._X_train-x)**2),axis=1))
        res_index=np.argsort(distance) # 顺序从小到大的下标
        first_k_item=[self._Y_train[the_k_item] for the_k_item in res_index[:self.k]] #找到距离最大的前k个的y值
        predict_result=Counter(first_k_item).most_common(1)[0][0] #选出出现频率最高的1个y作为结果
        return predict_result
    # def _predict(self, x):
    #     """给定单个待预测数据x，返回x的预测结果值"""
    #     assert x.shape[0] == self._X_train.shape[1], \
    #         "the feature number of x must be equal to X_train"
    #
    #     distances = [np.sqrt(np.sum((x_train - x) ** 2))
    #                  for x_train in self._X_train]
    #     nearest = np.argsort(distances)
    #
    #     topK_y = [self._Y_train[i] for i in nearest[:self.k]]
    #     votes = Counter(topK_y)
    #
    #     return votes.most_common(1)[0][0]

    def score(self,x_test,y_test):
        '''计算准确率并返回'''
        y_predict = self.predict_Y(x_test)
        accuracy = np.sum(y_test==y_predict)/len(y_test)
        return accuracy