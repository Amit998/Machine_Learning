import torch
import numpy as np

# x=torch.empty(1)

# x=torch.ones(2,2,dtype=torch.float16)

x=torch.rand(4,4)
y=torch.rand(2,3)


# z=torch.mul(x,y)
# z=torch.add(x,y)
# z=torch.sub(x,y)
# z=torch.div(x,y)

# print(x[:,:])

# print(x[1,1].item())

# y=x.view(-1,8)
# print(y.size())

# a=torch.ones(5)
# print(a)
# b=a.numpy()
# print(b)


# a=np.ones(5)
# print(a)
# b=torch.from_numpy(a)
# print(b)

if torch.cuda.is_available():
    device=torch.device("cuda")
    x=torch.ones(5,device=device)
    y=torch.ones(5)
    y=y.to(device)
    z=x+y
    z=z.to("cpu")
    x1=torch.ones(5,requires_grad=True)
    print(x1)