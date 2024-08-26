"""
    This script is used to run a monte carlo simulation of a collision between a trapped ion and a collisional particle. 
    Currently of interest is to see how the cell size and timestep size impacts the results of the simulation.
    The parameters to change are:
    Nr/Nz: The number of cells in the radial and axial directions
    Dr/Dz: The physical size of the simulation space in the radial and axial directions
    dtSmall/dtCollision/dtLarge: The time step sizes for the simulation
"""
from Collision2DClean import *
from scipy import special
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time


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

#wz can be varied if desired
lower = 2*np.pi*1e6
# upper = 2*np.pi*1.5e6
# num_freq = 5
wz = lower #np.linspace(lower,upper,num_freq)

# Here we'll make the DC and RF (pseudo)potentials
RF = makeRF0(m,q,wr,Nr,Nz,Nrmid,dr)
DC = makeDC(m,q,wz,Nz,Nr,Nzmid,dz)
nullFields = np.zeros((Nr,Nz))
print("constants set and modules imported") ; print("Simulation Size = ",Dr,"m in r ", Dz,"m in z")

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


"""
    Here we define the simulation parameters that will be chosen randomly. These are,
    Ni: how many ions in the crystal
    Nc: how many collisional particles. currently only supports Nc = 1
    vMin,vMax: The range of velocities that the collisional particle can have
    T: Temperature of the system. This is used to determine the probability of the collisional particle being at a given velocity in the range
    collisionalMass: Mass of the collisional particle in amu
    numBins: how many points for the velocity
    angles: range of angles for impact. currently only supports -np.pi/2 to np.pi/2
    offsets: how much offset from a direct collision the collisional particle will have
    max hypotenuse: defines how far away the collisional particle starts from the ion. This should be chosen based on the size of the simulation space
"""

Ni = 2
Nc = 1
T = 300
collisionalMass = 2
vMin = 50
vMax = 7000
numBins = 1000
boltzDist = Boltz(collisionalMass,T,vMin,vMax,numBins)
v = np.linspace(vMin,vMax,numBins)
angles = np.linspace(-np.pi/2,np.pi/2,100)
offsets = np.linspace(-2e-9,2e-9,200)
max_hypotenuse = 1.5e-5


shots = 100
start_time = time.perf_counter()
file_name = str(Ni)+"ionSimulation.txt"
f = open("2ionSimulation.txt", "w")
f.write("axial trapping frequency (MHz) \t velocity(m/s) \t ion collided with \t angle(rads) \t collision offset(m) \t reorder? (1 is reorder 2 is ejection) \n")

for i in range(shots):
    vf = makeVf(Ni,1.0*q,m,l,wz,offsetr,offsetz,vbumpr,vbumpz) #ignore offsetr/z,vbumpr/z for now. If you don't call this in the loop it causes a memory error for some reason
    #randomly chooses parameters for the run
    velocity = random.choices(v,weights=boltzDist)[0]
    angle_choice = random.choice(angles)
    offset_choice = random.choice(offsets)
    ion_collided = random.randint(0,Ni-1)

    #time allocated per run is based on the speed of the collisional particle
    if velocity < 200:
        Nt = 700000
    elif velocity < 1500:
        Nt = 400000
    else:
        Nt = 250000

    r = -np.cos(angle_choice)*max_hypotenuse
    z = vf[ion_collided,1] + np.sin(angle_choice)*max_hypotenuse + offset_choice
    vz=-1*velocity*np.sin(angle_choice) ; vr = np.abs(velocity*np.cos(angle_choice))

    reorder = mcCollision(vf,r,z,vr,vz,q,mH2,aH2,Nt,dtSmall,RF,DC,Nr,Nz,dr,dz,dtLarge,dtCollision)

    output = str(wz) +"\t" + str(velocity)+"\t"+str(ion_collided+1)+"\t"+str(angle_choice)+"\t"+str(offset_choice)+"\t"+str(reorder)+"\n"
    f.write(output)

finish_time = time.perf_counter()
timeTaken = finish_time - start_time

f.close()
print("Completed Succesfully! It took " + str(timeTaken) + " seconds!")