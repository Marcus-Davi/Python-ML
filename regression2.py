import matplotlib.pyplot as plt
import sys
from random import random
import numpy as np


def sum_all(vec):
    soma = 0
    for item in vec:
        soma += item
    return soma

def square(x):
    return x*x

def model(x,beta):
    return beta[0] + x*beta[1]


data = open("Nanook_Vels.csv").readlines()
del data[0] # tira header

ref = []
m_d = []
m_e = []

for it in data:
    line_split = it.split(",")
    ref.append(float(line_split[0]))
    m_d.append(float(line_split[1]))
    m_e.append(float(line_split[2]))

#plt.plot(ref,m_d)
#plt.show(block=False)

# Gradient Descent
# J = 1/(2N) sum (ref - model)^2
# Model = theta0 + theta1*x
N = len(data)

maxIt = 100
#beta = [10*random(),10*random()]
beta = np.array([random(), random()])
#usando numpy
err = np.zeros((N,1))
grad = np.array([0,0])
l_rate = np.array([0.1,0.0001])
#print(err)
for it in range(maxIt-1):
    
    for k in range(N):
        err[k] = model(ref[k],beta) - m_d[k]
        grad[0] += err[k]
        grad[1] += err[k]*ref[k]
    grad = grad/N
    print(l_rate)
    print(grad)
    print(l_rate*grad)
    beta = beta - l_rate*grad #elementwise

    cost = err.sum()
    print(cost)
print(beta)
#print(err)





