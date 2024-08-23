#from tools.distribution import *
from Collision2DClean import *
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
import math
import random

def Boltz(m,T,vmin=0,vmax=5000,bins=100):
    amu = 1.66*10**-27
    m = m*amu
    k = 1.386e-23 # boltzmann constant
    boltz = np.zeros(bins) # initialize vector
    dv = (vmax - vmin)/bins # define bin spacing in speed
    a = (k*T/m)**(1/2) # normalization constant for distribution function

    
    for i in range(bins):
        vhere = vmin + i*dv # define speed of bin
        vlast = vhere-dv
        boltz[i] = (special.erf(vhere/(a*np.sqrt(2))) - np.sqrt(2/np.pi)*(vhere/a)*np.exp(-vhere**2/(2*a**2)) ) - (special.erf(vlast/(a*np.sqrt(2))) - np.sqrt(2/np.pi)*(vlast/a)*np.exp(-vlast**2/(2*a**2)) ) # here we use the cumulative distribution function and subtract the one-step down value from the this step value for the probability density in this slice
    
    return boltz/np.sum(boltz)

def sign(x):
    if x > 0: return 1
    else: return -1
    return 1

# Define useful constants
amu = 1.67e-27 ; eps0 = 8.854e-12 ; qe = 1.6e-19 # SI units 

# Define physical params
m = 40. *amu; q = 1. *qe; wr = 2*np.pi*3e6# SI units

# Define sim params
Nr = 20001 ; Nz = 20001 # number of cells related to trapping axis needs to be odd to allow null line
Nrmid = (Nr-1)/2 ; Nzmid = (Nz-1)/2 # middle cell which exists at r=0 (z=0) in physical units
Dr = Nr*1.5e-9 ; Dz = Nz*4.5e-9 # physical width in m of the sim
dr = Dr/float(Nr) ; dz = Dz/float(Nz) # width of a cell in z in m
Ezf = np.zeros((Nr, Nz)) ; Erf = np.zeros((Nr, Nz)) # array of electric fields s

# Here we'll make the DC and RF (pseudo)potentials
RF = makeRF0(m,q,wr,Nr,Nz,Nrmid,dr)
DC = makeDC(m,q,wz,Nz,Nr,Nzmid,dz)
nullFields = np.zeros((Nr,Nz))
print("constants set and modules imported") ; print("Simulation Size = ",Dr,"m in r ", Dz,"m in z")

# Here we describe how the collisional particles work
# they are a list of lists where the first index is the particle identifier
# the second index gives r,z,vr,vz,q,m,a (dipole moment)
aH2 = 8e-31 # dipole moment of H2 in SI units
mH2 = 2.0*amu # mass of H2 in kg


dtSmall = 1e-12 ;dtCollision = 1e-16; dtLarge = 1e-10 # length of a time step in s

sigmaV = 100e-6 # fall-off of potential outside trapping region
dv = 20.0 # bin size for particle speed in determining if collision occurs
vmax = 5000 # maximum particle speed we allow
Nrmid = (Nr-1)/2 ; Nzmid = (Nz-1)/2
l = 1 ; dsepz = 2*(620+1)*dz # starting separation between ions in virtual units
vbumpr = 0.00e0 ; vbumpz = -0.0e0 # starting velocity in r and z of the lth ion in the chain
offsetz = 0.0e-7 ; offsetr = 0.0e-8 # starting distance from eq. in r and z of the lth ion in the chain

lower = 2*np.pi*1e6
upper = 2*np.pi*1.5e6
wz = np.linspace(lower,lower,1)

Ni = np.linspace(2,2,1) # number of trapped ions to loop over. must give whole integers
Ni = np.round(Ni).astype(int) 

Ni = 2
Nc = 1
T = 300
vMin = 50
vMax = 7000
numBins = 1000
shots = 1000

boltzDist = Boltz(2,T,vMin,vMax,numBins)
v = np.linspace(vMin,vMax,numBins)
velocity = random.choices(v,weights=boltzDist)[0]

angles = np.linspace(-np.pi/2,np.pi/2,100)
angle_choice = random.choice(angles)

offsets = np.linspace(-2e-9,2e-9,200)
offset_choice = random.choice(offsets)

ion_collided = random.randint(0,Ni-1)

if velocity < 200:
    Nt = 700000
elif velocity < 1500:
    Nt = 400000
else:
    Nt = 250000

vf = makeVf(Ni,1.0*q,m,l,wz,offsetr,offsetz,vbumpr,vbumpz)

max_hypotenuse = 1.5e-5 #limit the distance of the initial position of the collisional particle. this should be choosen based on the size of the simulation space
r = -np.cos(angle_choice)*max_hypotenuse
z = vf[ion_collided,1] + np.sin(angle_choice)*max_hypotenuse + offset_choice
vz=-1*velocity*np.sin(angle_choice) ; vr = np.abs(velocity*np.cos(angle_choice))

print("ion collided with = ",ion_collided+1); print("additional offset = ",offset_choice)
print("angle = ",angle_choice); print("velocity = ",velocity)

rs,zs,vrs,vzs,rcolls,zcolls,vrcolls,vzcolls,reorder = mcCollision(vf,r,z,vr,vz,q,mH2,aH2,Nt,dtSmall,RF,DC,Nr,Nz,dr,dz,dtLarge,dtCollision) 

#plots the results of the simulation. only use for single runs
plotPieceWise(Nc,Ni,rcolls,rs,zcolls,zs,0,100000,"H2 Collides with Trapped Ca+","Radial Position(m)","Axial Position (m)",-2e-5,2e-5,-4e-5,4e-5)

#plots a zoomed in image of the collision on a single ion
#plotPieceWise(Nc,Ni,rcolls,rs,zcolls,zs,1000,Nt,"H2 Collides with Trapped Ca+","Radial Position(m)","Axial Position (m)",-.25e-5,.25e-5,.25e-5,.5e-5)