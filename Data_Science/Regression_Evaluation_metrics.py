import statistics as st
sample=[600,470,170,430,300]
print(st.mean(sample))
print(st.stdev(sample))
print(st.variance(sample))

from sklearn.metrics import explained_variance_score
y_true=[3,-0.5,2,7]
y_pred=[2.5,0.0,2,8]

print(explained_variance_score(y_true,y_pred))


y_true=[[0.5,1],[-0.5,0.5],[-2,2],[7,-6]]
y_pred=[[0,3],[-1,2],[8,-5],[-8,9]]

explained_variance_score(y_true,y_pred,multioutput='uniform_average')


from sklearn.metrics import max_error

y_true=[3,-0.5,2,7]
y_pred=[2.5,0.0,2,8]
print(max_error(y_true,y_pred))


from sklearn.metrics import mean_absolute_error

y_true=[3,-0.5,2,7]
y_pred=[2.5,0.0,2,8]
print(mean_absolute_error(y_true,y_pred))


from sklearn.metrics import mean_squared_error
import math

y_true=[3,-0.5,2,7]
y_pred=[2.5,0.0,2,8]
print((mean_squared_error(y_true,y_pred)))


from sklearn.metrics import mean_squared_error
import math

y_true=[3,-0.5,2,7]
y_pred=[2.5,0.0,2,8]
print(math.sqrt(mean_squared_error(y_true,y_pred)))


from sklearn.metrics import mean_squared_log_error
import math
import numpy as np

y_true=[3,-0.5,2,7]
y_pred=[2.5,0.0,2,8]
print(np.squt(mean_squared_log_error(y_true,y_pred)))
