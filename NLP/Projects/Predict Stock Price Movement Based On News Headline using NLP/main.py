from os import scandir
import pandas as pd

df=pd.read_csv('Data.csv',encoding='ISO-8859-1')


# print(df.head())

train=df[df['Date'] < '20150101']
test=df[df['Date'] > '20150101']


# print(train.head())
data=train.iloc[:,2:27]


data.replace("[^a-zA-Z]"," ",regex=True,inplace=True)


list1=[i for i in range(25)]
new_index=[str(i) for i in list1]
data.columns=new_index
# print(data.columns)

for index in new_index:
    data[index]=data[index].str.lower()

# print(data.head())



headlines=[]

# print(len(data.index))
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))

# print(headlines[0])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

cv=CountVectorizer(ngram_range=(2,2))
trainDataset=cv.fit_transform(headlines)

# print(trainDataset)

randomClassifer=RandomForestClassifier(n_estimators=200,criterion='entropy')
model=randomClassifer.fit(trainDataset,train['Label'])

# import pickle
# TrainFile=open('data','wb')

# pickle.dump(model,TrainFile)
# TrainFile.close()

test_transform=[]
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset=cv.transform(test_transform)
predictions=randomClassifer.predict(test_dataset)

# print(predictions)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)