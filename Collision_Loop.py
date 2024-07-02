from Collision_Code_python import *
import numpy as np
import matplotlib.pyplot as plt
import math

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
nullFields = np.zeros((Nr,Nz))
print("constants set and modules imported") ; print("Simulation Size = ",Dr,"m in r ", Dz,"m in z")

#Laser Cooling parameters
c = 2.998e8 #speed of light in m/s
nu0 = 2*np.pi*c/397e-9 # atomic transition frequency in Hz for S1/2->P1/2 transition
vl = np.array([1,1]) #laser is both radial and axial
I0 = 5e-5/(np.pi*(25e-5)**2) #laser intensity in W/m^2 using 50 um with a 50um diameter beamwaist
dnul = 4e6# laser frequency width
A = 1.35e8/(2*np.pi) # einstein A coefficient for transition from the Nist table at 397.37nm
nul = 1.0*nu0 - A/2#Laser center frequency detuned from atomic transition
Ti = 1e-9 # transition lifetime
# Here we describe how the collisional particles work
# they are a list of lists where the first index is the particle identifier
# the second index gives r,z,vr,vz,q,m,a (dipole moment)
aH2 = 8e-31 # dipole moment of H2 in SI units
mH2 = 2.0*amu # mass of H2 in kg

Nt = 1000000# number of time steps to take
dtSmall = 1e-12 ;dtCollision = 1e-16; dtLarge = 1e-10 # length of a time step in s
dCutOut = 2*5000.0*dtLarge ; dCutIn = 0 #such that the fastest particle we track can only get half-way in before switching regimes

P = 3e-8 ; T = 300 # pressure and temperature in SI units
sigmaV = 100e-6 # fall-off of potential outside trapping region
dv = 20.0 # bin size for particle speed in determining if collision occurs
vmax = 5000 # maximum particle speed we allow
Nrmid = (Nr-1)/2 ; Nzmid = (Nz-1)/2 ; first = 10 ; last=Nt
l = 1 ; dsepz = 2*(620+1)*dz # starting separation between ions in virtual units
vbumpr = 0.00e0 ; vbumpz = -0.0e0 # starting velocity in r and z of the lth ion in the chain
offsetz = 0.0e-7 ; offsetr = 0.0e-8 # starting distance from eq. in r and z of the lth ion in the chain
lower = 2*np.pi*1e6
upper = 2*np.pi*1.5e6
wz = np.linspace(lower,lower,1)
Ni = np.linspace(2,2,1) # number of trapped ions to loop over. must give whole integers
Ni = np.round(Ni).astype(int) 
offset_array = np.linspace(0,2e-9,10)
v = np.linspace(0,250,10)
#minimum_dist = .1e-9
 #over-rode from 2 to allow collisions to occur faster
print("Simulation started with an lower wz of ", lower, " and an upper wz of ", upper, "\n")

f = open("output.txt", "w")
f.write("axial trapping frequency (MHz) \t velocity(m/s) \t #ions \t ion collided with \t collision offset(m) \t reorder? (1 is reorder 2 is ejection) \n")

#for axial in wz:
continue_calc = True
# Set up other constants
for axial in wz:
    DC = makeDC(m,q,axial,Nz,Nr,Nzmid,dz)
    Vfield = makeVField(m,q,wr,axial,Nr,Nz,Nrmid,Nzmid,dr,dz) # we add a matrix of the potential energy through our trap
    for speed in v:
        reorder_array = np.zeros(Ni)
        for ions in Ni:
            for offset in offset_array:
                for i in range(ions):
                    if continue_calc:
                        if speed < 200:
                            Nt = 700000
                        elif speed < 1500:
                            Nt = 400000
                        else:
                            Nt = 250000
                        t =np.pi/2; vz=speed*np.cos(t) ; vr = speed*np.sin(t)
                        r = vtopPos(2,Nrmid,dr)
                        vf = makeVf(ions,1.0*q,m,l,axial,offsetr,offsetz,vbumpr,vbumpz)
                        z = vf[i,1] + offset
                        withCollision = True
                        cooling = False
                        if withCollision == True:
                            reorder = runFasterCollision(vf,r,z,vr,vz,q,mH2,aH2,Nt,dtSmall,RF,DC,Nr,Nz,dr,dz,vl,I0,nul,dnul,nu0,A,Ti,dtLarge,dCutOut,dCutIn,dtCollision,cooling)    
                        else:
                            rs,zs,vrs,vzs,vf= ionNoCollission(vf,r,z,vr,vz,q,mH2,aH2,Nt,RF,DC,Nr,Nz,dr,dz,vl,I0,nul,dnul,nu0,A,Ti,dtLarge,cooling)
                        #write data to output file
                        output = str(axial) +"\t" + str(speed)+"\t"+str(ions)+"\t"+str(i+1)+"\t"+str(offset)+"\t"+str(reorder)+"\n"
                        f.write(output)

                        reorder_array[i] = reorder
                    
#                 if np.sum(reorder_array) == 0:
#                         continue_calc = False
                
#             continue_calc = True
f.close()
print("Completed Succesfully!")