from typing import List

# 调用精度评估和朴素贝叶斯模型
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB 

def get_acc(estimator, X, y):   # the accuracy
    y_pred = estimator.predict(X)
    acc = accuracy_score(y, y_pred)
    return acc


def get_additonal_term(estimator, X, y):  # fitness func 的附加项，默认为 0 
    additional_term = 0     
    return additional_term

#   元组tuple和列表list的区别：
#       * 列表定义的时候使用[]，元组使用()
#       * 列表可以改变，但是元组不可变，也就是元组定义好以后我们不能修改里面的元素

def fitness_function(population: List, train_dataset: tuple, 
                     test_dataset: tuple, weight: float) -> None:
    for p in population:
        clf = GaussianNB()
        if sum(p.arr) == 0:
            p.f_value[0] = 0.0
        else:
            X, y = train_dataset
            test_X, test_y = test_dataset
            # print("before X:\n", X)
            X = X[:, p.arr == 1]
            # print("After X :\n", X)
            test_X = test_X[:, p.arr == 1]
            clf.fit(X, y)
            score = get_acc(clf, test_X, test_y) + weight * get_additonal_term(clf, test_X, test_y)
            p.f_value[0] = score
            if p.pbest_score < score:
                p.pbest_score = score
                p.pbest = p.arr[:]
