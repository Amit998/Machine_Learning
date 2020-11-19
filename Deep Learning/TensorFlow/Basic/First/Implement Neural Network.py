import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv('dataset_insurance.csv')

X_train,X_test,y_train,y_test=train_test_split(df[['age','affordibility']],df['bought_insurance'],test_size=0.2,random_state=2)


X_train_scaled=X_train.copy()
X_train_scaled['age']=X_train_scaled['age']/100

X_test_scaled=X_test.copy()

X_test_scaled['age']=X_test_scaled['age']/100














class myNN:
    def __init__(self):
        self.w1=1
        self.w2=1
        self.bias=0
    def sigmoid(self,x):
        import math
        return 1 / (1 + math.exp(-x))
    def log_loss(slef,y_test,y_pred):
        epsilon=1e-15
        y_pred_new=[max(i,epsilon) for i in y_pred]
        y_pred_new=[min(i,1-epsilon) for i in y_pred_new]
        y_pred_new=np.array(y_pred_new)
        return -np.mean(y_test*np.log(y_pred_new)+(1-y_test) * np.log(1-y_pred_new))
    
    

    def fit(self,x,y,epochs,loss_thrashold):
        self.w1,self.w2,self.bias= self.gradient_descent(x['age'],x['affordibility'],y,epochs,loss_thrashold)
    
    def predict(self,test_data):
        weighted_sum=self.w1 * test_data['age']+self.w2 * test_data['affordibility']  + self.bias
        return self.sigmoid_numpy(weighted_sum)

    def sigmoid_numpy(self,X):
        return 1/(1+np.exp(-X))

    def gradient_descent(self,age,affordibility,y_train,epochs,loss_thrashold):
        w1=w2=1
        bias=0
        rate=0.5
        n=len(age)

        for i in range(epochs):
            wighted_sum=w1 * age + w2 * affordibility + bias
            y_pred=self.sigmoid_numpy(wighted_sum)
            loss=self.log_loss(y_train,y_pred)

            wd1=(1/n)*np.dot(np.transpose(age),(y_pred-y_train))
            wd2=(1/n)*np.dot(np.transpose(affordibility),(y_pred-y_train))

            bias_d=np.mean(y_pred-y_train)

            w1=w1 - rate * wd1
            w2 = w2 - rate * wd2
            bias=bias - rate * bias_d
            
            if (i%50==0):
                print(f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')

            if( loss <= loss_thrashold):
                print(f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
                return w1,w2,bias
        return w1,w2,bias
    

customModel=myNN()
customModel.fit(X_train_scaled,y_train,epochs=500,loss_thrashold=0.4631)
print(customModel.predict(X_test_scaled))
print(y_test)