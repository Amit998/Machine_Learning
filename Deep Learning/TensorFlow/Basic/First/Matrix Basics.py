import numpy as np

revenue=np.array([[180,200,220],[24,36,40],[12,18,20]])
expenses=np.array([[100,100,120],[14,26,10],[112,48,25]])
profit=revenue-expenses
print(profit)

price_per_unit=np.array([1000,400,1200])
units=np.array([[30,40,50],[5,10,15],[2,5,7]])

# total_price=price_per_unit *  units

print(np.dot(price_per_unit,units))

# print(total_price)