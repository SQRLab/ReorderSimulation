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
l1=pow(Q*Q/(2*m*w1*w1*math.pi*epsilon),1.0/3)
U1=m*w1*w1*l1*l1/4+Q*Q/(4*math.pi*epsilon*l1)
l2=pow(Q*Q/(2*m*w2*w2*math.pi*epsilon),1.0/3)
U2=m*w2*w2*l2*l2/4+Q*Q/(4*math.pi*epsilon*l2)
U=U1-U2
vt=pow(2*U/m,0.5)
r10=[0,l2/2]
r20=[0,-l2/2]
v10=[0,10]
v20=[0,0]
r=np.zeros((N,2,2))
v=np.zeros((N,2,2))
t=np.zeros(N)
d=np.zeros(N)
E=np.zeros(N)
a=np.zeros((2,2))

r[0,0,:]=r10
r[0,1,:]=r20
v[0,0,:]=v10
v[0,1,:]=v20
d[0]=math.sqrt((r[0,0,0]-r[0,1,0])**2+(r[0,0,1]-r[0,1,1])**2)
E[0]=m/2*(v[0,0,0]**2+v[0,0,1]**2+v[0,1,0]**2+v[0,1,1]**2)+k1/2*(r[0,0,0]**2+r[0,1,0]**2)+k2/2*(r[0,0,1]**2+r[0,1,1]**2)+kq*(1/d[0])


for i in range(0,N-1):
    t[i+1]=t[i]+tau
    d2=(r[i,0,0]-r[i,1,0])**2+(r[i,0,1]-r[i,1,1])**2
    d[i]=math.sqrt(d2)
    a[0,0]=-k1/m*r[i,0,0]+kq/m*(r[i,0,0]-r[i,1,0])/(d[i])**3
    a[0,1]=-k2/m*r[i,0,1]+kq/m*(r[i,0,1]-r[i,1,1])/(d[i])**3
    a[1,0]=-k1/m*r[i,1,0]+kq/m*(r[i,1,0]-r[i,0,0])/(d[i])**3
    a[1,1]=-k2/m*r[i,1,1]+kq/m*(r[i,1,1]-r[i,0,1])/(d[i])**3
    v[i+1,:,:]=v[i,:,:]+tau*a
    r[i+1,:,:]=r[i,:,:]+(v[i+1,:,:]+v[i,:,:])*tau/2
    E[i+1]=m/2*(v[i+1,0,0]**2+v[i+1,0,1]**2+v[i+1,1,0]**2+v[i+1,1,1]**2)+k1/2*(r[i+1,0,0]**2+r[i+1,1,0]**2)+k2/2*(r[i+1,0,1]**2+r[i+1,1,1]**2)+kq*(1/d[i])

#plt.plot(t,v[:,0,0])
#plt.plot(t,v[:,0,1])
#plt.show()

#plt.plot(t,r[:,0,0])
#plt.plot(t,r[:,1,1])
#plt.show()

#plt.plot(r[:,0,0],r[:,0,1])
#plt.plot(r[:,1,0],r[:,1,1])
plt.plot(t,r[:,0,1])
plt.plot(t,r[:,1,1])
plt.show()

#plt.plot(t,d)
#plt.show()

#print("%.30 f"%r[0,0,1])

plt.plot(t,E)
plt.show()
