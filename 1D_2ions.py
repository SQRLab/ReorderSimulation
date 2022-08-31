import math
import numpy as np
import matplotlib.pyplot as plt

tau=pow(10,-4)
T=10
vx0=100
vy0=500
x0=400
y0=100
k=100
N=int(T/tau)
x=np. zeros(N)
y=np. zeros(N)
vx=np.zeros(N)
vy=np.zeros(N)
t=np.zeros(N)

x[0]=x0
y[0]=y0
vx[0]=vx0
vy[0]=vy0

for i in range(0,N-1):
    t[i+1]=t[i]+tau
    vx[i+1]=vx[i]-k*x[i]*tau
    vy[i+1]=vy[i]-k*y[i]*tau
    x[i+1]=x[i]+(vx[i+1]+vx[i])*tau/2
    y[i+1]=y[i]+(vy[i+1]+vy[i])*tau/2
plt. plot(t,x)
# plt. show()

plt. plot(t,y)
plt. show()

plt. plot(x,y)
plt. show()


