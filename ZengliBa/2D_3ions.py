# 2D_3ions
import math
import numpy as np
import matplotlib.pyplot as plt

mp=1.67e-27
m=40*mp
Q=1.6e-19
epsilon=8.854e-12
kq=Q**2/(4*math.pi*epsilon)
w1=2*math.pi*5e6
w2=2*math.pi*0.2e6
k1=m*w1**2
k2=m*w2**2

tau=pow(10,-9)
N=pow(10,5)
T=tau*N
l=pow(5*Q*Q/(16*m*w2**2*math.pi*epsilon),1.0/3)
# U1=m*w1*w1*l**2+Q*Q/(4*math.pi*epsilon*l)*5/2

r10=[0,l]
r20=[0,-l]
r30=[0,0]
v10=[20,-80]
v20=[0,0]
v30=[0,0]

r=np.zeros((N,3,2))
v=np.zeros((N,3,2))
t=np.zeros(N)
d12=np.zeros(N)
d23=np.zeros(N)
d31=np.zeros(N)
a=np.zeros((3,2))

r[0,0,:]=r10
r[0,1,:]=r20
r[0,2,:]=r30
v[0,0,:]=v10
v[0,1,:]=v20
v[0,2,:]=v30

for i in range(0,N-1):
    t[i+1]=t[i]+tau
    d12[i]=math.sqrt((r[i,0,0]-r[i,1,0])**2+(r[i,0,1]-r[i,1,1])**2)
    d23[i]=math.sqrt((r[i,1,0]-r[i,2,0])**2+(r[i,1,1]-r[i,2,1])**2)
    d31[i]=math.sqrt((r[i,0,0]-r[i,2,0])**2+(r[i,0,1]-r[i,2,1])**2)
    a[0,0]=-k1/m*r[i,0,0]+kq/m*((r[i,0,0]-r[i,1,0])/(d12[i])**3+(r[i,0,0]-r[i,2,0])/(d31[i])**3)
    a[0,1]=-k2/m*r[i,0,1]+kq/m*((r[i,0,1]-r[i,1,1])/(d12[i])**3+(r[i,0,1]-r[i,2,1])/(d31[i])**3)
    a[1,0]=-k1/m*r[i,1,0]+kq/m*((r[i,1,0]-r[i,0,0])/(d12[i])**3+(r[i,1,0]-r[i,2,0])/(d23[i])**3)
    a[1,1]=-k2/m*r[i,1,1]+kq/m*((r[i,1,1]-r[i,0,1])/(d12[i])**3+(r[i,1,1]-r[i,2,1])/(d23[i])**3)
    a[2,0]=-k1/m*r[i,2,0]+kq/m*((r[i,2,0]-r[i,1,0])/(d23[i])**3+(r[i,2,0]-r[i,0,0])/(d31[i])**3)
    a[2,1]=-k2/m*r[i,2,1]+kq/m*((r[i,2,1]-r[i,1,1])/(d23[i])**3+(r[i,2,1]-r[i,0,1])/(d31[i])**3)
    #a[1,0]=-k1/m*r[i,1,0]+kq/m*(r[i,1,0]-r[i,0,0])/(d12[i])**3
    #a[1,1]=-k2/m*r[i,1,1]+kq/m*(r[i,1,1]-r[i,0,1])/(d12[i])**3
    v[i+1,:,:]=v[i,:,:]+tau*a
    r[i+1,:,:]=r[i,:,:]+(v[i+1,:,:]+v[i,:,:])*tau/2


#plt.plot(t,v[:,0,0])
#plt.plot(t,v[:,0,1])
#plt.show()

plt.plot(t,r[:,0,1])
plt.plot(t,r[:,1,1])
plt.plot(t,r[:,2,1])
plt.show()

#plt.plot(t,r[:,0,0])
#plt.plot(t,r[:,1,0])
#plt.plot(t,r[:,2,0])
#plt.show()

plt.plot(r[:,0,0],r[:,0,1])
#plt.plot(r[:,1,0],r[:,1,1])
plt.show()


