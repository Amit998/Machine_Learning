# import numpy as np


# # f= w * x
# # f = 2 * x


# X=np.array([1,2,3,4],dtype=np.float32)
# Y=np.array([1,2,3,4],dtype=np.float32)
# w= 0.0
# def forward(x):
#     return w * x

# #loss =MSE
# def loss(y,y_pred):
#     return ((y_pred-y) ** 2).mean()
# #gradient

# #MSE= 1/ N (w*x - y)**2
# # dj/dw = 1/N 2x (w*x -y )

# def gradient(x,y,y_pred):
#     return np.dot(2*x,y_pred-y).mean()

# print(f'prediction before training f(5) = {forward(5):.3f}')

# learning_rate=0.01
# n_iter=10

# for epoch in range(n_iter):
#     #forward pass
#     y_pred=forward(X)

#     #loss

#     l=loss(Y,y_pred)

#     #gradient
#     dw=gradient(X,Y,y_pred)

#     #update wights

#     w-=learning_rate * dw

#     if epoch %1 == 0:
#         print(f'epoch {epoch+1} : w = {w:.3f}, loss={l:.8f}')


# print(f'prediction before training f(5) = {forward(5):.3f}')




################# USING TORCH


# import numpy as np
import torch

# f= w * x
# f = 2 * x


X=torch.tensor([1,2,3,4],dtype=torch.float32)
Y=torch.tensor([2,8,10,20],dtype=torch.float32)

w= torch.tensor(0.0,dtype=torch.float32,requires_grad=True)


def forward(x):
    return w * x

#loss =MSE
def loss(y,y_pred):
    return ((y_pred-y) ** 2).mean()
#gradient

#MSE= 1/ N (w*x - y)**2
# dj/dw = 1/N 2x (w*x -y )

# def gradient(x,y,y_pred):
#     return np.dot(2*x,y_pred-y).mean()

print(f'prediction before training f(5) = {forward(5):.3f}')

learning_rate=0.01
n_iter=1000

for epoch in range(n_iter):
    #forward pass
    y_pred=forward(X)

    #loss

    l=loss(Y,y_pred)

    #gradient
    # dw=gradient(X,Y,y_pred)
    l.backward()

    #update wights
    with  torch.no_grad():
        w-=learning_rate * w.grad

    #zero gradient
    w.grad.zero_()
    if epoch %10 == 0:
        print(f'epoch {epoch+1} : w = {w:.3f}, loss={l:.8f}')


print(f'prediction before training f(5) = {forward(5):.3f}')

