import numpy as np

def gradient_descent(x,y):
    coef_curr=intersect_curr=0
    iterations=3000
    learning_rate=0.01

    n=len(x)
    for i in range(iterations):
       
        y_pred=coef_curr * x + intersect_curr
        
        coef_d=-(2/n) * sum(x * (y-y_pred))
        intersect_d=-(2/n) * sum(y-y_pred)


        cost=(1/n) * sum([val**2 for val in (y-y_pred)])
        coef_curr=coef_curr - learning_rate * coef_d
        intersect_curr=intersect_curr-learning_rate * intersect_d
        # print("Coef {}, B {},Cost {}, iteration {}".format(coef_curr,intersect_curr,cost,i))
    
    return coef_curr,intersect_curr




x=np.array([1,2,3,5,6,7])
y=np.array([6,7,8,9,10,11])

coef,intesect=gradient_descent(x,y)

print(2 * coef+ intesect)