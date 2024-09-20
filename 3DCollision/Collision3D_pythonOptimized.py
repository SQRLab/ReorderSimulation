import sys
sys.path.append('../tools/')
from IonChainTools import *
import matplotlib.pyplot as plt
import numpy as np
import math
import random
#from sklearn.preprocessing import normalize
import numba

"""
    Need to think about how to update the code to include the following:
    - Rewrite the code to include the new ion chain class
    - Reimplement laser cooling
    - fill out this section...
"""

class IonChain:
    def __init__(self, Ni, Nc, q, m, l, wr, wz, offsetr, offsetz, vbumpr, vbumpz):
        self.Ni = Ni
        self.Nc = Nc
        self.q = q
        self.m = m
        self.l = l
        self.wr = wr
        self.offsetr = offsetr
        self.offsetz = offsetz
        self.vbumpr = vbumpr
        self.vbumpz = vbumpz
        self.vf = makeVf(Ni,q,m,l,wr,offsetr,offsetz,vbumpr,vbumpz)
        self.vc = np.zeros((Nc,9))
        

# converts from physical to virtual units in position
@numba.njit
def ptovPos(pos,Nmid,dcell):  
    return (pos/dcell + float(Nmid)) # fractional position in vertual units

# converts from virtual to physical units in position (natually placing the point at the center of a cell)
@numba.njit
def vtopPos(pos,Nmid,dcell): 
    return float((pos-Nmid))*dcell # returns the center of the cell in physical units

# returns AC fields at each grid cell based on the amplitude at each cell, starting phase, current time, and frequency
@numba.njit
def ACFields(ErAC0,EzAC0,phaseAC,f,t): 
    return ErAC0*np.sin(phaseAC+f*t*2*np.pi),EzAC0*np.sin(phaseAC+f*t*2*np.pi)

# dummy function to generate the RF field in our grid assuming that it is a harmonic potential about (0,0) and focuses in x
@numba.njit
def makeRF0(m,q,w,Nr,Nz,Nrmid,Nzmid,dr): 
    # We take in the mass, frequency for that mass, cell numbers, midpoint, and physical width of a cell and output the RF electric fields (constant in z) as a function of radial cell
    C = m*(w**2)/q ; RFx = np.ones((Nr,Nr,Nz)); RFy = np.ones((Nr,Nr,Nz))
    for jCell in range(Nr):
        RFx[jCell,:,:] = RFx[jCell,:,:]*C*(Nrmid-jCell)*dr# electric field in pseudo-potential and harmonic potential approximation
    for iCell in range(Nr):
        RFy[:,iCell,:] = RFy[:,iCell,:]*C*(Nrmid-iCell)*dr
    return RFx, RFy

# dummy function to generate the DC fields assuming that it is a harmonic potential about (0,0) and focuses in z
    # We take in the mass, frequency for that mass, cell numbers, midpoint, and physical width of a cell and output the DC electric fields (constant in r) as a function of longitudinal cell
@numba.njit
def makeDC(m,q,w,Nz,Nr,Nzmid,dz): 
    C = m*(w**2)/q ; DC = np.ones((Nr,Nr,Nz))
    for jCell in range(Nr):
        for iCell in range(Nr):
            for kCell in range(Nz):
                DC[jCell,iCell,kCell] = DC[jCell,iCell,kCell]*C*(Nzmid-kCell)*dz # electric field for DC in harmonic potential approximation 
    return DC

# dummy function to generate the V field
@numba.njit
def makeVField(m,q,wr,wz,Nr,Nz,Nrmid,Nzmid,dr,dz):
    # we assign voltage at each point given our trapping frequencies
    Cr = -0.5*m*(wr**2)/q ; Cz = -0.5*m*(wz**2)/q ; Vf = np.zeros((Nr,Nr,Nz))
    for jCell in range(Nr):
        for iCell in range(Nr):
            for kCell in range(Nz):
                Vf[jCell,iCell,kCell] = Cr*((Nrmid-jCell)*dr)**2 + Cr*((Nrmid-iCell)*dr)**2 + Cz*((Nzmid-kCell)*dz)**2 # makes a harmonic potential in each axis, adds them
    return Vf

# this makes an initial array for the ions, the first index is the ion, the second index is r,z,vr,vz,q,m,polarizability [treated as zero]
def makeVf(Ni,q,m,wr,l=0,offsetx=0,offsety=0,offsetz=0,vbumpx=0,vbumpy=0,vbumpz=0):
    '''
    [x-position,y-position,axial(z)-position,x-velocity,y-velocity,z-velocity,charge,mass,polarizability]
    '''
    vf = np.zeros((Ni,9))
    pos = calcPositions(Ni); lscale = lengthScale(wr); scaledPos = pos*lscale
    for i in range(Ni):
        vf[i,:] = [0.0e-6,0.0e-6,-scaledPos[i],0,0,0,q,m,0.0]
    vf[l,0] += offsetx ; vf[l,1] += offsety; vf[l,2] += offsetz
    vf[l,3] += vbumpx ; vf[l,4] += vbumpy; vf[l,5] += vbumpz
    return vf

# takes in velocities and mass and returns energy
def particleE(vr,vz,m):
    return (1/2)*m*(vr**2 + vz**2)

#takes in velocity and mass and returns momentum
def particleP(v,m):
    return m*v

# takes in vectors of velocities and masses and returns total energy and momenta
def totalEandP(vrs,vzs,ms):
    En = 0.0; pr = 0.0; pz = 0.0
    for i in range(len(vrs)):
        En+=particleE(vrs[i],vzs[i],ms) ; pr+=particleP(vrs[i],ms) ; pz+=particleP(vzs[i],ms)
    return [En,pr,pz]

# we're only checking ion-dipole distances, checks if anything is closer together than dCut
def farApart(vf,vc,dCut):
    dCut2 = dCut**2
    Ni = len(vf[:,0]); Nc = len(vc[:,0])
    for i in range(Ni):
        for j in range(Nc):
            d2 = ((vf[i,0]-vc[j,0]))**2 + ((vf[i,1]-vc[j,1]))**2 + ((vf[i,2]-vc[j,2]))**2
            if d2 < dCut2:
                return False
    
    return True


def plotSim(Nc,Ni,xcolls,ycolls,zcolls,xions,yions,zions,xlimit,ylimit,zlimit,first,last,title1):
    """
    Makes one continuous plot
    Nc,Ni are the numbers of collisional particles and ions to plot
    first,last are the bounds 
    colls,ions are the arrays of what we want to plot for, like position of each ion over time (we assume time is linear) 
    dt is the time step and the step of the thing we want to plot for
    """
    # Now we plot their positions over time
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    for i in range(Nc):
        ax.scatter(xcolls[i,first:last],ycolls[i,first:last],zcolls[i,first:last])
    for i in range(Ni):
        ax.scatter(xions[i,first:last],yions[i,first:last],zions[i,first:last])
    ax.set_zlim(-zlimit, zlimit)  # Set the limits of the z-axis
    plt.xlim(-xlimit,xlimit); plt.ylim(-ylimit,ylimit)
    plt.xlabel("x-coordinate (radial)"); plt.ylabel("y-coordinate (radial)")
    plt.title(title1)
    plt.show()    

def plotPosition(Ni,dt,ions,first,last,title1,xlabel1,ylabel1):
    """
    Makes one continuous plot
    Nc,Ni are the numbers of collisional particles and ions to plot
    first,last are the bounds 
    colls,ions are the arrays of what we want to plot for, like position of each ion over time (we assume time is linear) 
    dt is the time step and the step of the thing we want to plot for
    """
    # Now we plot their positions over time
    timesteps = np.linspace(first,last,last)
    for i in range(Ni):
        plt.plot(dt*timesteps,ions[i,first:last])
    plt.xlabel(xlabel1) ; plt.ylabel(ylabel1) ; plt.title(title1)
    #plt.ylim(-3e-5,3e-5)
    plt.show() 

def plotPieceWise(Nc,Ni,colls1,ions1,colls2,ions2,first,last,title1,xlabel1,ylabel1,xlow,xhigh,ylow,yhigh):
    """
    Makes a scatter plot of any two things against each other
    Nc,Ni are the numbers of collisional particles and ions to plot
    first,last give the slice of the attributes we wish to plot against each other
    colls,ions are the vectors of what we want to plot for 1 denotes the x-axis variable, 2 for the y-axis variable
    title1 is a title, xlabel1 and ylabel1 are axes labels, x/y low/high are axes bounds
    """
    for i in range(Nc):
        plt.scatter(colls1[i,first:last],colls2[i,first:last])
    for i in range(Ni):
        plt.scatter(ions1[i,first:last],ions2[i,first:last])
    plt.xlabel(xlabel1) ; plt.ylabel(ylabel1) ; plt.title(title1)
    plt.xlim(xlow,xhigh) ; plt.ylim(ylow,yhigh)
    plt.show()

    
def subPlotThings(N1,N2,dt,thing1,thing2,first,last,title1,xlabel1,ylabel1,ylabel2):
    """
    Makes two plots that share an x-axis of time
    N1,N2 are the numbers of vectors of things to plot in each array thing1 and 2 (number of particles)
    first,last are the bounds 
    thing1,thing2 are the vectors of what we want to plot for 
    dt is the time step assuming linear time
    first,last give the slice of thing1 and 2
    title1,x/ylabel1/2 give a title and axes labels
    """
    for i in range(N1):
        plt.plot(dt*range(first,last),thing1[i,first:last])
    for i in range(N2):
        plt.plot(dt*range(first,last),thing2[i,first:last])
    plt.xlabel(xlabel1) ; plt.ylabel(ylabel2) ; plt.title(title1)
    plt.show()

# Define Primary Functions
@numba.njit
def minDists(vf,vc):
    #sets initial distances to compare to within the loops. if the distance is less then these then the value will be reassigned
    rid2 = 1e6 ; rii2 = 1e6 ; vid2 = 1e6 ; vii2 = 1e6
    Ni = len(vf[:,0]) ; Nc = len(vc[:,0])
    for i in range(Ni):
        for j in range(i+1,Ni): # check each pair of ions for distance and speed
            x = vf[i,0]-vf[j,0] ; y = vf[i,1]-vf[j,1]; z=vf[i,2]-vf[j,2] ; vx = vf[i,3]-vf[j,3] ; vy = vf[i,4]-vf[j,4] ; vz = vf[i,5]-vf[j,5]
            dist2 = x**2 + y**2 + z**2
            v2 = vx**2 + vy**2 + vz**2
            if dist2<rii2:
                vii2 = v2 ; rii2 = dist2                
        for j in range(Nc): # check each ion-dipole pair for distance and speed
            x = vf[i,0]-vc[j,0] ; y = vf[i,1]-vc[j,1] ; z=vf[i,2]-vc[j,2] ; vx = vf[i,3]-vc[j,3] ; vy = vf[i,4]-vc[j,4] ; vz = vf[i,5]-vc[j,5]
            dist2 = x**2 + y**2 + z**2
            v2 = vx**2 + vy**2 + vz**2
            if dist2<rid2:
                vid2 = v2 ; rid2 = dist2     
    #returns sqrt due to distance formula. this saves application of the sqrt operation
    return np.sqrt(rid2),np.sqrt(rii2),np.sqrt(vid2),np.sqrt(vii2)

@numba.njit
def collisionMode(rii,rid,a,e=0.1):
    return (a*rii**2)/(rid**5)>e

@numba.njit
def confirmValid(vf,dr,dz,Nr,Nz,Nrmid,Nzmid):
    """
    initial check to make sure that the collisional particle is not going to break the sim
    """
    for i in range(len(vf[:,0])):
        xCell = ptovPos(vf[i,0],Nrmid,dr) ; yCell = ptovPos(vf[i,1],Nrmid,dr) ; zCell = ptovPos(vf[i,2],Nzmid,dz) 
        if xCell>Nr-2 or xCell<1:
            vf[i,:] = vf[i,:]*0.0 ; vf[i,0] = 2.0 ; vf[i,1] = 2.0 ; vf[i,2] = 2.0 ; vf[i,3] = 0.0 ; vf[i,4] = 0.0 ; vf[i,5] = 0.0 ; vf[i,7] = 1e6
        elif yCell>Nr-2 or yCell<1:
            vf[i,:] = vf[i,:]*0.0 ; vf[i,0] = 2.0 ; vf[i,1] = 2.0 ; vf[i,2] = 2.0 ; vf[i,3] = 0.0 ; vf[i,4] = 0.0 ; vf[i,5] = 0.0 ; vf[i,7] = 1e6
        elif zCell>Nz-2 or zCell<1:
            vf[i,:] = vf[i,:]*0.0 ; vf[i,0] = 2.0 ; vf[i,1] = 2.0 ; vf[i,2] = 2.0 ; vf[i,3] = 0.0 ; vf[i,4] = 0.0 ; vf[i,5] = 0.0 ; vf[i,7] = 1e6
    return vf

@numba.njit
def updatePoss(vf,dr,dz,dt,Nr,Nz,Nrmid,Nzmid):
    """ This moves our vf particles as their velocity suggests for one time step
    vf is the vector of ion parameters, the first index is the ion, the second index is r,z,vr,vz,q,m,polarizability
    dr,dz are the cell dimensions of our trapping field grid
    dt is the timestep 
    Nr,Nz are the grid sizes and Nrmid,Nzmid are the midpoints
    """
    for i in range(len(vf[:,0])):
        vf[i,0] += vf[i,3]*dt ; vf[i,1] += vf[i,4]*dt ; vf[i,2] += vf[i,5]*dt # correct if physical since one time step is passing
        xCell = ptovPos(vf[i,0],Nrmid,dr) ; yCell = ptovPos(vf[i,1],Nrmid,dr) ; zCell = ptovPos(vf[i,2],Nzmid,dz) 
        """
        will update the parameters of the particle if it is outside of the simulation region, works as an ejection
        """
        vf = confirmValid(vf,dr,dz,Nr,Nz,Nrmid,Nzmid)

    return vf


# apply the forces from the local electric field over one time step
@numba.njit
def updateVels(vf,Exf,Eyf,Ezf,dt,Nrmid,Nzmid): 
    for i in range(len(vf[:,1])): # apply E-field of the cell
        Fx = vf[i,6]*Exf[i] ; Fy = vf[i,6]*Eyf[i] ; Fz = vf[i,6]*Ezf[i] # solve force 
        # then we would need to convert back to virtual units 
        vf[i,3] += Fx*dt/(vf[i,7]) ; vf[i,4] += Fy*dt/(vf[i,7]) ; vf[i,5] += Fz*dt/(vf[i,7]) # update velocity with F*t/m
    return vf

@numba.njit
def solveFields(vf,Fx,Fy,Fz,Nrmid,Nzmid,Ni,dr,dz):
    """ this solves for the electric fields at  each ion from each ion (and the trap)
    vf is the vector of ion parameters, the first index is the ion, the second index is x,y,z,vx,vy,vz,q,m,polarizability
    ErDC is the array of electric fields from the DC electrodes in r
    EzDC is the array of electric fields from the DC electrodes in z
    ErAC is the array of electric fields from the AC electrodes in r
    EzAC is the array of electric fields from the AC electrodes in z
    Nrmid, Nzmid are the midpoints of the grid
    Ni is the number of ions
    dr and dz are the cell sizes of the grid
    Erf2 and Ezf2 are the electric fields at each ion. These names could probably be improved
    """
    #sets constants
    eps0 = 8.854e-12 ; C1 = 4*np.pi*eps0 #SI units 
    Exf2 = np.zeros(Ni) ; Eyf2 = np.zeros(Ni) ; Ezf2 = np.zeros(Ni)
    for i in range(len(vf[:,0])): # note that this no longer takes the electric field from all particles, so chillax
        """
        BUG
        kCell is going to 600ish for some reason. need to fix this so that I can use variable Nr/Nz sizes since 3D requires a much smaller grid
        BUG
        """
        jCell = ptovPos(vf[i,0],Nrmid,dr) ; iCell = ptovPos(vf[i,1],Nrmid,dr) ; kCell = ptovPos(vf[i,2],Nzmid,dz) #
        jCell = int(round(jCell)) ; iCell = int(round(iCell)) ; kCell = int(round(kCell)) # local cell index in r and z
        Exf2[i] += Fx[jCell,iCell,kCell]; Ezf2[i] += Fz[jCell,iCell,kCell]; Eyf2[i] += Fy[jCell,iCell,kCell] # add trap fields
        for j in range(len(vf[:,0])):
            if j!=i: #here we solve for the fields from each other ion on this ion
                xdist = (vf[j,0]-vf[i,0]) ; ydist = (vf[j,1]-vf[i,1]) ; zdist = (vf[j,2]-vf[i,2]) # get each distance
                sqDist = (xdist)**2 + (ydist)**2 + (zdist)**2 #square distance from particle to cell
                projX = xdist/sqDist**(1/2) ; projY = ydist/sqDist**(1/2) ; projZ = zdist/sqDist**(1/2) # cos theta to project E field to z basis, sin to r basis               
                Exf2[i] += -projX*vf[j,6]/(C1*sqDist) ; Eyf2[i] += -projY*vf[j,6]/(C1*sqDist) ; Ezf2[i] += -projZ*vf[j,6]/(C1*sqDist) # add fields in r and z 
    return Exf2,Eyf2,Ezf2

@numba.njit
def collisionParticlesFields(vf,vc,Ni,RFx,RFy,DCz,dr,dz,dt,Nrmid,Nzmid):
    """
    vf is the ion parameters, the first index is the ion, the second index is r,z,vr,vz,q,m,polarizability
    vc is the collisional particle parameters with the same scheme as vf
    Ni is the number of ions
    Erfi is the radial electric fields on ions
    Ezfi is the axial electric fields on ions
    Erfc is the radial electric fields on collisional particles at both pseudo-particle points
    Ezfc is the axial electric fields on collisional particles at both pseudo-particle points
    dr is the physical size of a cell in r, dz is the physical size of a cell in z
    ErDC,EzDC,ErAC,EzAC are the electric fields from the background
    note that we treat dipoles as if they instantly align with the dominant electric field
    """
    eps0 = 8.854e-12
    Nc = len(vc[:,0])
    # we begin by instantiating the electric field lists (fields are in physical units)
    Exfi = np.zeros(Ni); Eyfi = np.zeros(Ni); Ezfi = np.zeros(Ni)
    Exfc = np.zeros((Nc,2)); Eyfc = np.zeros((Nc,2)); Ezfc = np.zeros((Nc,2)) # [i,1] is middle field, 2 is high index - low index divided by the size of a cell (ie, the local slope of the E-field)
    sqDist = np.zeros((Nc,Ni)); projX = np.zeros((Nc,Ni)); projY = np.zeros((Nc,Ni)); projZ = np.zeros((Nc,Ni))
    C1 = 4*np.pi*eps0 # commonly used set of constants put together
    # we solve the electric fields on the collisional particles
    for i in range(Nc): # for each collisional particle that exists        
        # In order to allow for electric field gradients of the background field, here we need to implement a linear E-field gradient between neighboring cells
        jCell = ptovPos(vc[i,0],Nrmid,dr) ; iCell = ptovPos(vc[i,1],Nrmid,dr) ; kCell = ptovPos(vc[i,2],Nzmid,dz)
        jCell = int(round(jCell)) ; iCell = int(round(iCell)) ; kCell = int(round(kCell)) # local cell index in r and z
        # we initialize the interpolated field for each 
        Exfc[i,0] += (RFx[jCell,iCell,kCell]) ; Eyfc[i,0] += (RFy[jCell,iCell,kCell]) ;Ezfc[i,0] += (DCz[jCell,iCell,kCell])
        Exfc[i,1] += (RFx[jCell+1,iCell,kCell])-(RFx[jCell-1,iCell,kCell])/dr ; Eyfc[i,1] += (RFy[jCell,iCell+1,kCell] )-(RFy[jCell,iCell-1,kCell] )/dr ; Ezfc[i,1] += (DCz[jCell,iCell,kCell+1]-DCz[jCell,iCell,kCell-1])/dz       
        for j in range(Ni): # solve the electric field exerted by each ion
            xdist = (vf[j,0]-vc[i,0]) ; ydist = (vf[j,1]-vc[i,1]) ; zdist = (vf[j,2]-vc[i,2])
            sqDist[i,j] = (xdist)**2 + (ydist)**2 + (zdist)**2 #distance from particle to cell
            projX[i,j] = xdist/sqDist[i,j]**(1/2) ; projY[i,j] = ydist/sqDist[i,j]**(1/2) ; projZ[i,j] = zdist/sqDist[i,j]**(1/2) #cos theta to project E field to z basis and sin to r basis
            Exfc[i,0] += -projX[i,j]*vf[j,6]/(C1*sqDist[i,j]) ;  Eyfc[i,0] += -projY[i,j]*vf[j,6]/(C1*sqDist[i,j]) ; Ezfc[i,0] += -projZ[i,j]*vf[j,6]/(C1*sqDist[i,j]) # add fields in r and z   
            # I just need to add the gradient field from these now and the colliding particle should rebound
            Exfc[i,1] += 2*projX[i,j]*vf[j,6]/(C1*sqDist[i,j]**(3/2)) ; Eyfc[i,1] += 2*projY[i,j]*vf[j,6]/(C1*sqDist[i,j]**(3/2)) ; Ezfc[i,1] += 2*projZ[i,j]*vf[j,6]/(C1*sqDist[i,j]**(3/2)) # add fields in r and z                 
    pX = np.zeros(Nc); pY = np.zeros(Nc); pZ = np.zeros(Nc); pTot = np.zeros(Nc)
    for i in range(Nc):    # a dipole is induced in the direction of the electric field vector with the positive pseudoparticle in the positive field direction
        if vc[i,6]!=0.0: # if there is a dipole moment that can be obtained
            pX[i] = -2*np.pi*eps0*vc[i,8]*Exfc[i,0] # dipole in r in SI units note this factor of 2 pi epsilon0 which corrects the units of m^-3 on alpha and Volts/m on E to give a dipole moment in Coulomb*meters
            pY[i] = -2*np.pi*eps0*vc[i,8]*Eyfc[i,0]
            pZ[i] = -2*np.pi*eps0*vc[i,8]*Ezfc[i,0] # dipole in z in SI units ###FIX THIS###
            pTot[i] = (pX[i]**2+pY[i]**2+pZ[i]**2)**(1/2) # total dipole length in physical units
            # we can now induce the force on the dipole
            Fx = abs(pX[i])*Exfc[i,1] ; Fy = abs(pY[i])*Eyfc[i,1] ; Fz = abs(pZ[i])*Ezfc[i,1]
            #then we would need to convert back to virtual units once we apply the forces
            vc[i,3] += Fx*dt/(vc[i,7]) ; vc[i,4] += Fy*dt/(vc[i,7]) ; vc[i,5] += Fz*dt/(vc[i,7]) # update velocity with F*t/m               
    # we then solve for the fields the collisional particles exert on the ions from the dipole (and quadrapole potentially) as well as the charge if the particle has one
    for i in range(Ni): # for each ion in the trap
        for j in range(Nc): # apply the field from each collisional particle
            # the dipole field is (3*(dipole moment dotted with vector from particle to ion)(in vector from particle to ion) - (dipole moment))/(3*pi*eps0*distance^3)
            # at close proximity, it should be treated as monopoles with charge qe separated by (dipole moment)/qe distance along the dipole moment vector
            # for now we treat the electric field it exerts as a pure dipole
            if vc[j,6]!=0.0: # if there is a potential dipole moment
                Rhatx = projX[j,i] ; Rhaty = projY[j,i] ; Rhatz = projZ[j,i]
                dist = sqDist[j,i]**(1/2)
                Exfi[i] += -abs(pX[j])*(2*Rhatx)/(C1*dist**3) ; Eyfi[i] += -abs(pY[j])*(2*Rhaty)/(C1*dist**3) ; Ezfi[i] += -abs(pZ[j])*(2*Rhatz)/(C1*dist**3) # add dipole fields
    return vc,Exfi,Eyfi,Ezfi,Exfc,Eyfc,Ezfc # the ion electric fields are just from the collisional particles, the collisional electric fields are from all sources
# note that I haven't applied electric fields from collisional particles onto each other

@numba.njit
def runFasterCollision(vf,xc,yc,zc,vxc,vyc,vzc,qc,mc,ac,Nt,dtSmall,dtCollision,RFx,RFy,DC,Nr,Nz,dr,dz,dtLarge,eii = 0.01,eid = 0.01, withCollision = True):
    """    This sim runs one collisional particle through the trap until it exits, an ion exits, or ions cross
    vf is the ion array, the first index is the ion, the second index is r,z,vr,vz,q,m,polarizability
    rc,zc,vrc,vzc,qc,mc,ac are the initial parameters of the collisional particle
    Nt,Nr,Nz are the number of time steps and cells in each dimension
    dt,dr,dz,dvr,dvz are the time step, cell size, and speed for 1cell/1timestep
    Er,Ez are the background electric fields both radially and axially
    vl,I0,nul,dnul,nu0,A,Ti are laser cooling parameters
    dtLarge is the timestep when no particles are closer than dCutOut, dtMid when none closer than dCutIn, dtSmall is the timestep when at least one is inside dCutIn
    """
    reorder = 0
    # Start by defining some constants
    Nrmid = (Nr-1)/2; Nzmid = (Nz-1)/2
    Ni = len(vf[:,1]); Nc = 1
    vc= np.zeros((Nc,9)); vc[0,:] = [xc,yc,zc,vxc,vyc,vzc,qc,mc,ac] # initialize collisional particle
    zs = np.zeros((Ni,Nt)); zmeans = np.zeros(Nt); xs = np.zeros((Ni,Nt)); xmeans = np.zeros(Nt) ; ys = np.zeros((Ni,Nt)); ymeans = np.zeros(Nt)
    vxs = np.zeros((Ni,Nt)); vys = np.zeros((Ni,Nt)); vzs = np.zeros((Ni,Nt)); aveSpeeds = np.zeros((2,Nt))
    xcolls = np.zeros((Nc,Nt)); ycolls = np.zeros((Nc,Nt)); zcolls = np.zeros((Nc,Nt)); vxcolls = np.zeros((Nc,Nt)); vycolls = np.zeros((Nc,Nt)); vzcolls = np.zeros((Nc,Nt)) 
    # we assume that the collisional particle was initialized far enough away to be in the large distance scale zone
    dtLast = dtLarge; dtNow = dtSmall
    dvr = dr/dtNow; dvz = dz/dtNow
    
    crossTest = 0 # means our ions have not yet crossed paths
    nullFields = np.zeros((Nr,Nr,Nz))
    vc = confirmValid(vc,dr,dz,Nr,Nz,Nrmid,Nzmid)
    for i in range(Nt):       
        # Now we apply the time step calculation
        rid,rii,vid,vii = minDists(vf,vc)
        collision = collisionMode(rii,rid,vc[0,8],0.3)
        if collision:
             #println("Collision time-scale")
            dtNow = rid*eid/(5*vid) # if a collision is occuring, we set the time-step to limit the change of ion-dipole force between time-steps to be eid as a fraction of the starting value
        else:
            dtNow = dtSmall # otherwise we ensure eii fractional change in ion-ion forces or dtLarge as the maximum possible time step
        if dtNow < dtCollision:
            dtNow = dtCollision
        #RF = RF_start*np.cos(omega*i)
        Exfi,Eyfi,Ezfi = solveFields(vf,RFx,RFy,DC,Nrmid,Nzmid,Ni,dr,dz) # solve fields from ions on ions
        if withCollision==True:
            if vc[0,7]<1e6: #if the collisional particle exists
                dtNow = dtSmall
                vc,Exfic,Eyfic,Ezfic,Exfc,Eyfc,Ezfc = collisionParticlesFields(vf,vc,Ni,RFx,RFy,DC,dr,dz,dtNow,Nrmid,Nzmid) # solve fields from ions on collision particles and vice versa, also updates collisional particle velocity
                vc = updatePoss(vc,dr,dz,dtNow,Nr,Nz,Nrmid,Nzmid) # updates collisional particle positions
                Exfi += Exfic; Eyfi += Eyfic; Ezfi += Ezfic # add the collisional fields  
            else:
                dtNow = dtLarge
        else:
            pass
            
        vf = updateVels(vf,Exfi,Eyfi,Ezfi,dtNow,Nrmid,Nzmid) # fields apply force to ions
        vf = updatePoss(vf,dr,dz,dtNow,Nr,Nz,Nrmid,Nzmid) # velocities carry ions
        #print(vf[0])
        if np.any(np.isnan(vf)): # end sim if ion values break
            print("NaN detected in ion parameters before laserCool!!") ; print("vf = ",vf) ; print("i = ",i)
            vf[:,0] = 0.0; vf[:,1] = 0.0; vf[:,2] = 0.0; vf[:,3] = 0.0; vf[:,4] = 0.0; vf[:,5] = 0.0; vf[:,6] = 1e1; vf[:,7] = 0.0; vf[:,8] = 0.0;
            vc[:,0] = 0.0; vc[:,1] = 0.0; vf[:,2] = 0.0; vc[:,3] = 0.0; vc[:,4] = 0.0; vc[:,5] = 0.0; vc[:,6] = 1e1; vc[:,7] = 0.0; vf[:,8] = 0.0;
            break

            
        xs[:,i]=vf[:,0]; ys[:,i]=vf[:,1]; xcolls[:,i]=vc[:,0]; ycolls[:,i]=vc[:,1]; zs[:,i]=vf[:,2]; zcolls[:,i]=vc[:,2]
        vxs[:,i]=vf[:,3]; vys[:,i]=vf[:,4]; vzs[:,i]=vf[:,5]; vxcolls[:,i]=vc[:,3]; vycolls[:,i]=vc[:,4]; vzcolls[:,i]=vc[:,5]
        
        if np.any(np.isnan(vf)): # end sim if ion values break
            print("NaN detected in ion parameters after laserCool!!") ; print("vf = ",vf) ; print("i = ",i)
            vf[:,0] = 0.0; vf[:,1] = 0.0; vf[:,2] = 0.0; vf[:,3] = 0.0; vf[:,4] = 0.0; vf[:,5] = 0.0; vf[:,6] = 1e1; vf[:,7] = 0.0; vf[:,8] = 0.0;
            vc[:,0] = 0.0; vc[:,1] = 0.0; vf[:,2] = 0.0; vc[:,3] = 0.0; vc[:,4] = 0.0; vc[:,5] = 0.0; vc[:,6] = 1e1; vc[:,7] = 0.0; vf[:,8] = 0.0;
            break
        if np.sum(vf[:,7])>1e5: # end sim if ion leaves
            print("Ion Ejected") ; print("vf = ",vf) ; print("i = ",i)
            reorder += 2
            break
        """
        Checks for reorder. needs to be updated for 3D
        """
        for j in range(1,Ni):
            if zs[j,i]>zs[j-1,i] and crossTest<1: # end sim if neighboring ions have reordered
                reorder += 1
                crossTest+=1
                #print("Crossing at timestep ",i," of ion ",j) 
                Nt = i+1000 # do 5000 more time steps after a crossing is detected
                #break
        #if crossTest>0:
        #    break
    return xs,ys,zs,vxs,vys,vzs,xcolls,ycolls,zcolls,vxcolls,vycolls,vzcolls,reorder
    #return reorder
