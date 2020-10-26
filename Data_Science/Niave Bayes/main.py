# import csv
# import math
# import random 

# from sklearn import datasets
# from sklearn import metrics
# from sklearn.naive_bayes import GaussianNB

# dataset=datasets.load_iris()
# model=GaussianNB()
# model.fit(dataset.data,dataset.target)

# expected=dataset.target
# predicted=model.predict(dataset.data)


# print(metrics.classification_report(expected,predicted))
# print(metrics.confusion_matrix(expected,predicted))

arr=[22,33,11,3,121,2,3,4]
arr.sort(reverse=True)
arr.reverse()
print(arr)
