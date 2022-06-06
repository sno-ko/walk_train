import numpy as np

a = 1
b = 5
c = 3
sum = a + b + c


a_ = a / sum
b_ = b / sum
c_ = c / sum
list = [a_, b_, c_]
print(a_, b_, c_)

print(np.random.choice([0, 1, 2], 2, replace=False, p=list))
