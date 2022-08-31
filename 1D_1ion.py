import math
import numpy as np
import matplotlib.pyplot as plt
# a=np.zeros((2,2))
# print(a)
# b=np. ones((1,2))
# print(b)
# d=b[0,1]
# print(d)
# a=np. array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
# b=a[:2,1:3]
# print(b)
# b=a[1,:]
# print(b)
tau=pow(10,-5)
T=1
x0=5.0
v0=100.0
k=4*math.pi**2*4
N=int(T/tau)
x=np. zeros((N))
v=np. zeros((N))
t=np. zeros((N))
v[0]=v0
# print(x)
for i in range(0,N-1):
    t[i+1]=t[i]+tau
    v[i+1]=v[i]-k*x[i]*tau
    x[i+1]=x[i]+(v[i]+v[i+1])*tau/2
# print(x)

# a=np.array([[1,2],[3,4]])
# b=np.array([[1,2],[3,4]])
# print(a.dot(b))

plt.plot(t,x)
# plt.plot(t,v)
plt.show()

plt.plot(t,v)
plt.show()
