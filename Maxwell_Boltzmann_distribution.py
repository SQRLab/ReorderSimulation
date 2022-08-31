import math
import numpy as np
import matplotlib.pyplot as plt

mp=1.67*pow(10,-27)
m1=40*mp
m2=2*mp
k=1.38*pow(10,-23)
epsilon=8.854*pow(10,-12)
miu=1.257*pow(10,-6)
T=300

n=pow(10,6)

nv=np.zeros(n)
vx=np.zeros(n)
for i in range(0,n-1):
    vi=np.random.normal(0,math.sqrt(k*T/m2),3)
    nv[i]=math.sqrt(vi[0]**2+vi[1]**2+vi[2]**2)
    vx[i]=vi[1]

nvm=np.zeros(100)
nvx=np.zeros(100)
t=np.zeros(100)

for i in range(0,100):
    t[i]=100*(i+1)
print(t)

for i in range(0,n):
    for j in range(0,100):
        if nv[i]<j*100:
            nvm[j]=nvm[j]+1
            break

for i in range(0,n):
    for j in range(0,100):
        if abs(vx[i])<(j+1)*100:
            nvx[j]=nvx[j]+1
            break

print('Caculation finished')

plt.plot(t,nvm)
plt.plot(t,nvx)
plt.show()

