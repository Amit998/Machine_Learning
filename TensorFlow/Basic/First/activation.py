import math
def sigmoid(x):
    return 1 / (1+ math.exp(-x))
print(sigmoid(-56))
print(sigmoid(18))
print(sigmoid(200))
print(sigmoid(100))
print(sigmoid(1))
print(sigmoid(100))



def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x)+ math.exp(-x))

print(tanh(0))


def relu(x):
    return max(0,x)

print(relu(9))

def leaky_relu(x):
    return max(0.1 * x,x)

print(leaky_relu(-100))