import math
def sigmoid(x):
    return 1 / (1+ math.exp(-x))

print(sigmoid(-1))
print(sigmoid(-2))
print(sigmoid(-3))
print(sigmoid(1))
print(sigmoid(2))
print(sigmoid(3))
print(sigmoid(-10))
print(sigmoid(10))



# def tanh(x):
#     return (math.exp(x) - math.exp(-x)) / (math.exp(x)+ math.exp(-x))

# print(tanh(0))


# def relu(x):
#     return max(0,x)

# print(relu(9))

# def leaky_relu(x):
#     return max(0.1 * x,x)

# print(leaky_relu(-100))