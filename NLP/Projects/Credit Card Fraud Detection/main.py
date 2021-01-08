import numpy as np
from numpy import random
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
rcParams['figure.figsize'] = 8, 6
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]


data = pd.read_csv('D:/study/datasets/CSV_data/creditcard.csv',sep=',')
# print(data.head())

# print(data.info())

# print(data.isnull().values.any())

# print(data.columns)

# print(data["Class"].describe())

# print(data["Class"].value_counts())

# count_classes=pd.value_counts(data['Class'],sort=True)
# count_classes.plot(kind='bar',rot=0)
# plt.title("Transcation Class Distribution")
# plt.xticks(range(2),labels=LABELS)
# plt.xlabel("Class")
# plt.ylabel("Frequency")
# plt.show()

# fraud=data[data["Class"]==1]

# normal=data[data["Class"]==0]

# print(fraud.shape,normal.shape)

# print(fraud.describe())
# print(normal.describe())


# f,(ax1,ax2)=plt.subplots(2,1,sharex=True)
# f.suptitle('Amount Per Transaction y class')
# bins=50
# ax1.hist(fraud.Amount,bins=bins)
# ax1.set_title('Fraud')
# ax2.hist(normal.Amount,bins=bins)
# ax2.set_title('Normal')
# plt.xlabel('Amount (&) ')
# plt.ylabel('Number of Transction')
# plt.xlim((0,2000))
# plt.yscale('log')
# plt.show()

# f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# f.suptitle('Time of transaction vs Amount by class')
# ax1.scatter(fraud.Time, fraud.Amount)
# ax1.set_title('Fraud')
# ax2.scatter(normal.Time, normal.Amount)
# ax2.set_title('Normal')
# plt.xlabel('Time (in Seconds)')
# plt.ylabel('Amount')
# plt.show()

data1=data.sample(frac=0.1,random_state=1)

# print(data1.shape)

fraud=data1[data1["Class"]==1]

normal=data1[data1["Class"]==0]

# print(fraud.shape,normal.shape)

outlier_fration=len(fraud) / float(len(normal))

# print("Outlier {}".format(outlier_fration))

# import seaborn as sns
# #get correlations of each features in dataset
# corrmat = data1.corr()
# top_corr_features = corrmat.index
# plt.figure(figsize=(30,30))
# #plot heat map
# g=sns.heatmap(data[top_cor r_features].corr(),annot=True,cmap="RdYlGn")
# plt.show()

columns=data1.columns.tolist()
print(columns)

columns=[c for c in columns if c not in ["Class"]]
# print(columns)

target="Class"

state=np.random.RandomState(42)

x=data1[columns]
y=data1[target]

outlier=state.uniform(low=0,high=1,size=(x.shape[0],x.shape[1]))


# print(outlier)
# print(x.shape,y.shape)


classifier={
    "Isolation Forest":IsolationForest(n_estimators=100,max_samples=len(x),contamination=outlier_fration,random_state=state,verbose=0),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20,algorithm='auto',leaf_size=30,metric='minkowski',p=2,metric_params=None,contamination=outlier_fration),
    "Support Vector Machine":OneClassSVM(kernel='rbf',degree=3,gamma=0.1,nu=0.05,max_iter=-1)
}


n_outliers = len(fraud)

for i,(clf_name,clf) in enumerate(classifier.items()):
    if (clf_name=="Local Outlier Factor"):
        y_pred=clf.fit_predict(x)
        score_pred=clf.negative_outlier_factor_
    elif (clf_name == "Support Vector Machine"):
        clf.fit(x)
        y_pred=clf.predict(x)
    else:
        clf.fit(x)
        score_params=clf.decision_function(x)
        y_pred=clf.predict(x)

    y_pred[y_pred==1]=0
    y_pred[y_pred==-1]=1
    n_errors=(y_pred !=y).sum()
    print("{}: {}".format(clf_name,n_errors))
    print("Accuracy Score :")
    print(accuracy_score(y,y_pred))
    print("Classification Report :")
    print(classification_report(y,y_pred))




