import numpy as np

def gradient_descent(x,y):
    m_curr=b_curr=0
    iterations=10
    learning_rate=0.001

    n=len(x)
    for i in range(iterations):
       
        y_pred=m_curr * x + b_curr
        md=-(2/n) * sum(x * (y-y_pred))
        cost=(1/n) * sum([val**2 for val in (y-y_pred)])
        bd=-(2/n) * sum(y-y_pred)
        m_curr=m_curr - learning_rate * md
        b_curr=b_curr-learning_rate * bd
        print("M {}, B {},Cost {}, iteration {}".format(m_curr,b_curr,cost,i))




x=np.array([1,2,3,5,6,7])
y=np.array([6,7,8,9,10,11])

gradient_descent(x,y)