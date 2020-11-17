import math


age=[22,25,47,52,46,56,27]
affordabiity=[1,0,1,0,1,1,0]
have_insurance=[0,0,1,0,1,1,0]


w1=0
x1=0
w2=0
x2=0
bias=0
actY=1

for i,j,k in zip(age,affordabiity,have_insurance):
    w1=1
    w2=1
    x1=i
    x2=j
    # print(x1,x2)
    y=w1 * x1 + w2 * x2 + bias
    print(y)

    y_dash=1/(1+math.exp(-y))
      
    error=(math.log(y_dash)  + ((1-actY) * math.log(1-y_dash)))

    print(error)