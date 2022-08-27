import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])

plt.plot(x_train,y_train)
plt.show()
cost_val = []
def compute_cost(x, y, w, b):
    # number of training examples
    m = x.shape[0]
    print(m)
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
        print(cost_sum)
        cost_val.append((1 / (2 * m)) * cost_sum)
    total_cost = (1 / (2 * m)) * cost_sum
    print("total cost",total_cost)
    return total_cost

w = [200,100,250,150,210,150]
#w = np.sort(w)
b = [1,2,3,7,0.1,2,50]
compute_cost(x_train,y_train,250,10)

cost_list = []
#computing cost for different  parameters
for i in range(len(w)):
    cost_list.append(compute_cost(x_train,y_train,w[i],b[i]))
n = list(range(x_train.shape[0]))
plt.scatter(w,cost_list)