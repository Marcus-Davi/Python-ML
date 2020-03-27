import matplotlib.pyplot as plt
import sys
from random import random

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
l_rate0 = 0.1
l_rate1 = 0.0001

maxIt = 100
beta = [10*random(),10*random()]
#print(beta)

err = [0]*N
#print(err)

err_vec = []

for it in range(maxIt-1):

    #compute all errors
    grad0 = 0
    grad1 = 0
    for k in range(N):
        err[k] = model(ref[k],beta) - m_d[k]
        grad0 = grad0 + err[k] #derivative of cost
        grad1 = grad1 + err[k]*ref[k]
    grad0 = grad0/N
    grad1 = grad1/N

    beta[0] = beta[0] - l_rate0*grad0
    beta[1] = beta[1] - l_rate1*grad1

    err2 = list(map(square,err))
#    print(err2)
    cost = sum_all(err2)
#    print(cost)
    err_vec.append(cost)

model_out = [model(x,beta) for x in ref] #list compreenhsion

#print(err)
print(beta) 
plt.plot(err_vec)
plt.show(block=False)
plt.xlabel("Iterations")
plt.ylabel("Quadratic Error")

plt.figure()
plt.plot(ref,m_d, label="Measurements")
plt.plot(ref,model_out, label="Model Fit")
plt.legend()
plt.grid()
plt.show()



