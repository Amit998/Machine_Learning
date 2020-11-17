import numpy as np




def mean_error(y_pred,y_true):
    # print(y_pred)
    total_error=0
    for y_p,y_t in zip(y_pred,y_true):
        total_error += abs( y_t - y_p )
        
    print("Total error",total_error)

    mea=total_error/len(y_true)

    print("Mean Error",mea)

    return mea


y_pred=np.array([1,1,0,0,1])
y_true=np.array([0.30,0.7,1,0,0.5])

print(mean_error(y_pred,y_true))

print(np.mean(np.abs(y_pred,y_true)))


print(np.log([0.00000000000000001]))

eplion=1e-15


y_pred_new=[max(i,eplion) for i in y_pred ]

print(y_pred_new)

y_pred_new=[min(i,1-eplion) for i in y_pred_new ]

print(y_pred_new)

y_pred_new=np.array(y_pred_new)
np.log(y_pred_new)
print(np.mean(y_true*np.log(y_pred_new)+(1-y_true)*np.log(1-y_pred_new)))