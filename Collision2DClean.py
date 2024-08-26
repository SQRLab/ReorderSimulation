#2D ion collision simulation code in python. currently there is an issue with low speed collisions that I do not believe was present in earlier
#iterations. Currently trying to debug this.

from tools.IonChainTools import * # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import math
#from sklearn.preprocessing import normalize
import numba

@numba.njit
def ptovPos(pos,Nmid,dcell):  # converts from physical to virtual units in position
    return (pos/dcell + float(Nmid)) # fractional position in virtual units

@numba.njit
def vtopPos(pos,Nmid,dcell): # converts from virtual to physical units in position (natually placing the point at the center of a cell)
    return float((pos-Nmid))*dcell # returns the center of the cell in physical units

@numba.njit
def ACFields(ErAC0,EzAC0,phaseAC,f,t): # returns AC fields at each grid cell based on the amplitude at each cell, starting phase, current time, and frequency
    return ErAC0*np.sin(phaseAC+f*t*2*np.pi),EzAC0*np.sin(phaseAC+f*t*2*np.pi)

@numba.njit
def makeRF0(m,q,w,Nr,Nz,Nrmid,dr): # dummy function to generate the RF field in our grid assuming that it is a harmonic potential about (0,0) and focuses in x
    # We take in the mass, frequency for that mass, cell numbers, midpoint, and physical width of a cell and output the RF electric fields (constant in z) as a function of radial cell
    C = -m*(w**2)/q ; RF = np.ones((Nr,Nz))
    for jCell in range(Nr):
        RF[jCell,:] = -RF[jCell,:]*C*(Nrmid-jCell)*dr # electric field in pseudo-potential and harmonic potential approximation
    return RF

@numba.njit
def makeDC(m,q,w,Nz,Nr,Nzmid,dz): # dummy function to generate the DC fields assuming that it is a harmonic potential about (0,0) and focuses in z
    # We take in the mass, frequency for that mass, cell numbers, midpoint, and physical width of a cell and output the DC electric fields (constant in r) as a function of longitudinal cell
    C = -m*(w**2)/q ; DC = np.ones((Nr,Nz))
    for kCell in range(Nz):
        DC[:,kCell] = -DC[:,kCell]*C*(Nzmid-kCell)*dz # electric field for DC in harmonic potential approximation 
    return DC

@numba.njit
def makeVField(m,q,wr,wz,Nr,Nz,Nrmid,Nzmid,dr,dz):
    # we assign voltage at each point given our trapping frequencies
    Cr = -0.5*m*(wr**2)/q ; Cz = -0.5*m*(wz**2)/q ; Vf = np.ones((Nr,Nz))
    for jCell in range(Nr):
        for kCell in range(Nz):
            Vf[jCell,kCell] = Cr*((Nrmid-jCell)*dr)**2 + Cz*((Nzmid-kCell)*dz)**2 # makes a harmonic potential in each axis, adds them
    return Vf

def makeVf(Ni,q,m,l,wr,offsetr,offsetz,vbumpr,vbumpz):
    # this makes an initial array for the ions, the first index is the ion, the second index is r,z,vr,vz,q,m,polarizability [treated as zero]
    vf = np.zeros((Ni,7));
    pos = calcPositions(Ni); lscale = lengthScale(wr); scaledPos = pos*lscale;
    for i in range(Ni):
        vf[i,:] = [0.0e-6,-scaledPos[i],0,0,q,m,0.0]
    vf[l,0] += offsetr ; vf[l,1] += offsetz
    vf[l,2] += vbumpr ; vf[l,3] += vbumpz
    return vf

@numba.njit
def particleE(vr,vz,m):
    # takes in velocities and mass and returns energy
    return (1/2)*m*(vr**2 + vz**2)

@numba.njit
def particleP(v,m):
    # takes in velocity and mass and returns momentum
    return m*v

@numba.njit
def totalEandP(vrs,vzs,ms):
    # takes in vectors of velocities and masses and returns total energy and momenta
    En = 0.0; pr = 0.0; pz = 0.0
    for i in range(len(vrs)):
        En+=particleE(vrs[i],vzs[i],ms) ; pr+=particleP(vrs[i],ms) ; pz+=particleP(vzs[i],ms)
    return [En,pr,pz]

@numba.njit
def farApart(vf,vc,dCut):
    # we're only checking ion-dipole distances, checks if anything is closer together than dCut
    dCut2 = dCut**2
    Ni = len(vf[:,0]); Nc = len(vc[:,0])
    for i in range(Ni):
        for j in range(Nc):
            d2 = ((vf[i,0]-vc[j,0]))**2 + ((vf[i,1]-vc[j,1]))**2
            if d2 < dCut2:
                return false
    
    return True


def plotThing(Nc,Ni,dt,colls,ions,first,last,title1,xlabel1,ylabel1):
    """
    Makes one continuous plot
    Nc,Ni are the numbers of collisional particles and ions to plot
    first,last are the bounds 
    colls,ions are the arrays of what we want to plot for, like position of each ion over time (we assume time is linear) 
    dt is the time step and the step of the thing we want to plot for
    """
    # Now we plot their positions over time
    timesteps = np.linspace(first,last,last)
    for i in range(Nc):
        plt.plot(dt*timesteps,colls[i,first:last])
    for i in range(Ni):
        plt.plot(dt*timesteps,ions[i,first:last])
    plt.xlabel(xlabel1) ; plt.ylabel(ylabel1) ; plt.title(title1)
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
    rid2 = 1e6 ; rii2 = 1e6 ; vid2 = 1e6 ; vii2 = 1e6
    Ni = len(vf[:,0]) ; Nc = len(vc[:,0])
    for i in range(Ni):
        for j in range(i+1,Ni): # check each pair of ions for distance and speed
            r = vf[i,0]-vf[j,0] ; z=vf[i,1]-vf[j,1] ; vr = vf[i,2]-vf[j,2] ; vz = vf[i,3]-vf[j,3]
            dist2 = r**2 + z**2
            v2 = vr**2 + vz**2
            if dist2<rii2:
                vii2 = v2 ; rii2 = dist2                
        for j in range(Nc): # check each ion-dipole pair for distance and speed
            r = vf[i,0]-vc[j,0] ; z=vf[i,1]-vc[j,1] ; vr = vf[i,2]-vc[j,2] ; vz = vf[i,3]-vc[j,3]
            dist2 = r**2 + z**2
            v2 = vr**2 + vz**2
            if dist2<rid2:
                vid2 = v2 ; rid2 = dist2                
    return np.sqrt(rid2),np.sqrt(rii2),np.sqrt(vid2),np.sqrt(vii2)

@numba.njit
def collisionMode(rii,rid,a,e=.3):
    return (a*rii**2)/(rid**5)>e

@numba.njit
def updatePoss(vf,dr,dz,dt,Nr,Nz,Nrmid,Nzmid):
    """ This moves our vf particles as their velocity suggests for one time step
    vf is the vector of ion parameters, the first index is the ion, the second index is r,z,vr,vz,q,m,polarizability
    dr,dz are the cell dimensions of our trapping field grid
    dt is the timestep 
    Nr,Nz are the grid sizes and Nrmid,Nzmid are the midpoints
    """
    for i in range(len(vf[:,0])):
        vf[i,0] += vf[i,2]*dt ; vf[i,1] += vf[i,3]*dt # correct if physical since one time step is passing
        rCell = ptovPos(vf[i,0],Nrmid,dr) ; zCell = ptovPos(vf[i,1],Nzmid,dz) 
        if rCell>Nr-2 or rCell<1:
            #println("Position out of simulation in r ",vf[i,1]," Particle # ",i," vf = ",vf[i,:]) ; println("dt = ",dt)
            vf[i,:] = vf[i,:]*0.0 ; vf[i,0] = 2.0 ; vf[i,1] = 2.0 ; vf[i,2] = 0.0 ; vf[i,3] = 0.0 ; vf[i,5] = 1e6
        elif zCell>Nz-2 or zCell<1:
            #println("Position out of simulation in z ",vf[i,2]," Particle # ",i," vf = ",vf[i,:]) ; println("dt = ",dt)
            vf[i,:] = vf[i,:]*0.0 ; vf[i,0] = 2.0 ; vf[i,1] = 2.0; vf[i,2] = 0.0 ; vf[i,3] = 0.0 ; vf[i,5] = 1e6
    return vf

@numba.njit
def updateVels(vf,Erf,Ezf,dt,Nrmid,Nzmid): # apply the forces from the local electric field over one time step
    for i in range(len(vf[:,1])): # apply E-field of the cell
        Fr = vf[i,4]*Erf[i] ; Fz = vf[i,4]*Ezf[i] # solve force 
        # then we would need to convert back to virtual units 
        vf[i,2] += Fr*dt/(vf[i,5]) ; vf[i,3] += Fz*dt/(vf[i,5]) # update velocity with F*t/m
    return vf

@numba.njit
def solveFields(vf,ErDC,EzDC,ErAC,EzAC,Nrmid,Nzmid,Ni,dr,dz):
    """ this solves for the electric fields at  each ion from each ion (and the trap)
    vf is the vector of ion parameters, the first index is the ion, the second index is r,z,vr,vz,q,m,polarizability
    ErDC is the array of electric fields from the DC electrodes in r
    EzDC is the array of electric fields from the DC electrodes in z
    ErAC is the array of electric fields from the AC electrodes in r
    EzAC is the array of electric fields from the AC electrodes in z
    Nrmid, Nzmid are the midpoints of the grid
    Ni is the number of ions
    dr and dz are the cell sizes of the grid
    Erf2 and Ezf2 are the electric fields at each ion. These names could probably be improved
    """
    eps0 = 8.854e-12 ; C1 = 4*np.pi*eps0 #SI units 
    Erf2 = np.zeros(Ni) ; Ezf2 = np.zeros(Ni)
    for i in range(len(vf[:,0])): # note that this no longer takes the electric field from all particles, so chillax 
        jCell = ptovPos(vf[i,0],Nrmid,dr) ; kCell = ptovPos(vf[i,1],Nzmid,dz) # 
        jCell = int(round(jCell)) ; kCell = int(round(kCell)) # local cell index in r and z
        Erf2[i] += ErDC[jCell,kCell] + ErAC[jCell,kCell] ; Ezf2[i] += EzDC[jCell,kCell] + EzAC[jCell,kCell] # add trap fields
        for j in range(len(vf[:,0])):
            if j!=i: #here we solve for the fields from each other ion on this ion
                rdist = (vf[j,0]-vf[i,0]) ; zdist = (vf[j,1]-vf[i,1]) # get each distance
                sqDist = (rdist)**2 + (zdist)**2 #square distance from particle to cell
                projR = rdist/sqDist**(1/2) ; projZ = zdist/sqDist**(1/2) # cos theta to project E field to z basis, sin to r basis               
                Erf2[i] += -projR*vf[j,4]/(C1*sqDist) ; Ezf2[i] += -projZ*vf[j,4]/(C1*sqDist) # add fields in r and z 
    return Erf2,Ezf2

def laserInt(m,I0,nul,sigl,A21,nua,fwhma,bins=2000,width=10,vels=10001,velMax=10):
    """
    This function outputs the interaction strength (resulting in an interaction rate and energy per interaction) between a gaussian laser and a lorentzian atomic transition as an array of frequencies and absorption rates.
    This is equivalent to integrating for each bin over the relevant range between the normalized lineshapes of each component multiplied together.
    This calculates absorption rate as ((c^2*fwhma*A21*I0/(16*sqrt(2)*pi^(5/2)*h*sigl*nu0^3))*Integral(exp(-(nu[1,i]-nul)^2/(2*sigl^2))*((nu[1,i] - nua)^2 + (1.0/4.0)*fwhma^2)^-1,{nu,0,Inf})
   
    nul is the central laser frequency in Hz
    sigl is the gaussian linewidth of the laser in Hz
    nua is the central atomic frequency in Hz
    fwhma is the full width at half maximum of the lorentzian linewidth of the atomic transition in Hz
    vels is the number of velocities to consider for the ion-laser interaction
    bins is the number of frequencies to consider in the given range for interaction strength
    width is the number of frequency widths to consider when binning the interaction strength vector
    A21 is the einstein A coefficient of the atomic transition
    I0 is the laser intensity integrated over frequency [W/m^2]    
    nu is a 2D array where [1,:] is the velocities along the laser direction and [2,:] are the absorption rates for those velocities and [3,:] are the energy differences between absorbed and emitted photon
    """
    h = 6.626e-34 # planck constant [J*s]
    c = 2.998e8 # speed of light [m/s]
    nu = np.zeros((3,bins)) ; nuj = np.zeros((3,vels))
    # println("In laserInt: I0 = ",I0) ; println("nul = ",nul) ; println("sigl = ",sigl)
    # println("A21 = ",A21) ; println("nua = ",nua) ; println("fwhma = ",fwhma)    
    if sigl < fwhma/2.0: # set center frequency and scan width to encompass whole possible lineshape (smaller lineshape dominates)
        d=width*sigl ; nu0=nul
    else:
        d=width*fwhma/2.0 ; nu0=nua
    delta = 2.0*d/bins # step size between bins
    for j in range(vels):
        vel = -velMax + (j-1)*2*velMax/(vels-1) # set velocity for jth bin
        nulj = nul*(1-vel/c)/(1+vel/c) # laser frequency for jth velocity
        nuj[0,j] = vel;
        for i in range(bins): # do the integral
            nuInt = nu0 - d + (i-0.5)*delta # sets frequency to the middle of the bin scanning from nu0 - d to nu0 + d
            gl = exp(-(nuInt-nulj)**2/(2*sigl**2)) # laser lineshape minus constants
            ga = ((nuInt - nua)**2 + (1.0/4.0)*fwhma**2)**(-1) # atomic lineshape minus constants
            nu[1,i] = gl*ga*delta # width of the rectangle with height of the center of the bin for lineshape and width of the bin
        nu[1,:] = nu[1,:]*(c**2*fwhma*A21*I0/(16*np.sqrt(2)*np.pi**(5/2)*h*sigl*nu0**3)) # scale the interaction rate by the various constants
        nuj[1,j] = np.sum(nu[1,:]) #set interaction rate to sum of interaction rates
        if nuj[1,j]>1.0*A21:
            nuj[2,j] = (m/2)*( (h*nulj/(m*c))**2 + (h*nua/(m*c))**2 -2*(nuj[0,j])*(h*nua/(m*c)) + 2*(nuj[0,j])*(h*nulj/(m*c)) -2*(h*nua/(m*c))*(h*nulj/(m*c)) )  # if stimulated emission regime, recoils from absorption in laser direction then recoils from emission against laser direction
        else:
            nuj[2,j] = (m/2)*( (h*nulj/(m*c))**2 + (h*nua/(m*c))**2 + 2*(h*nulj/(m*c))*(nuj[0,j]) ) # if spontaneous emission regime, cools by 
    plt.plot(nuj[0,:],nuj[1,:])
    return nuj

def laserCoolSmoothNew(vf,vl,nu,Ni,dt): # this takes the particle population and applies a global laser cooling field coupled to the radial and axial velocities that purely reduces the velocity like a true damping force
    """
    vf is the vector of ion parameters, the first index is the ion, the second index is r,z,vr,vz,q,m,polarizability
    vl is the laser propagation vector (vr,vz) normalized to 1
    Ni is number of ions, dt is time step
    nu is the array where nu[0,:] are the velocities in the laser direction that interact with the atom and nu[1,;] are the absorption rates of         those frequencies and [2,:] are the energy differences between absorbed and emitted photon
    """

    Nnu = len(nu[0,:])
    #vl = normalize([vl])
    #v_rms = 0.5737442651757222 #rms velocity limit of 40Ca+ for the S1/2 to P1/2 transition in m/s
    for i in range(Ni):
        #magnitude = sqrt(vf[i,3]^2+vf[i,4]^2)
        #if
        #println("ion = ",i)
        if vf[i,5]<1.0: #if the particle exists it should have a realistic mass
            vil = vf[i,2]*vl[0][0]+vf[i,3]*vl[0][1] #velocity of ion in laser propagation direction
            rate = 0.0 ; dE = 0.0
            first = np.searchsorted(nu[0,:],vil)
            #println("first: ", first)
            if first > len(nu[0,:]):
#                println("Error, velocity above scope of laserInt") ; println("index found = ",first) ; println(" velocity = ",vil)
                first = len(nu[0,:])-1 # this assumes the cooling rates at the bounds are sufficiently close to zero
            elif first < 1:
#                println("Error, velocity below scope of laserInt") ; println("index found = ",first) ; println(" velocity = ",vil)   
                first = 0 # this assumes the cooling rates at the bounds are sufficiently close to zero
            rate = nu[1,first] ; dE = nu[2,first]
            #photon_emission_direction = normalize([rand(-1:.01:1),rand(-1:.01:1)]) #random photon emission unit vector
            dv = (2*rate*dt*abs(dE)/vf[i,5])**(1/2) #velocity change determined by the change in kinetic energy
            #println("rate: ", rate, " dE: ", dE, " dt: ", dt, " dv: ", dv)
            #println(dv)
            if dE>0: #the absorbed photon was higher in energy then the emitted photon
                vf[i,2] = vf[i,2]+abs(vl[0][0])*dv*math.copysign(1,vf[i,2]) # radial velocity increase from absorption
                vf[i,3] = vf[i,4]+abs(vl[0][1])*dv*math.copysign(1,vf[i,3]) # axial velocity increase from absorption
            if dE<0:#the absorbed photon was lower in energy then the emitted photon
                vf[i,2] = vf[i,3]-abs(vl[0][0])*dv*math.copysign(1,vf[i,2]) # radial velocity reduction from absorption
                vf[i,3] = vf[i,4]-abs(vl[0][1])*dv*math.copysign(1,vf[i,3]) # axial velocity reduction from absorption
    return vf # we return the updated populations              

@numba.njit
def collisionParticlesFields(vf,vc,Ni,ErDC,EzDC,ErAC,EzAC,dr,dz,dt,Nrmid,Nzmid): # this applies the fields of all existing collisional particles and changes the velocity of those collisional particles
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
    Erfi = np.zeros(Ni); Ezfi = np.zeros(Ni)
    Erfc = np.zeros((Nc,2)); Ezfc = np.zeros((Nc,2)) # [i,1] is middle field, 2 is high index - low index divided by the size of a cell (ie, the local slope of the E-field)
    sqDist = np.zeros((Nc,Ni)); projR = np.zeros((Nc,Ni)); projZ = np.zeros((Nc,Ni))
    C1 = 4*np.pi*eps0 # commonly used set of constants put together
    # we solve the electric fields on the collisional particles
    for i in range(Nc): # for each collisional particle that exists        
        # In order to allow for electric field gradients of the background field, here we need to implement a linear E-field gradient between neighboring cells
        jCell = ptovPos(vc[i,0],Nrmid,dr) ; kCell = ptovPos(vc[i,1],Nzmid,dz)
        jCell = int(round(jCell)) ; kCell = int(round(kCell)) # local cell index in r and z
        # we initialize the interpolated field for each 
        Erfc[i,0] += (ErDC[jCell,kCell] + ErAC[jCell,kCell]) ; Ezfc[i,0] += (EzDC[jCell,kCell] + EzAC[jCell,kCell])
        Erfc[i,1] += ((ErDC[jCell+1,kCell] + ErAC[jCell+1,kCell])-(ErDC[jCell-1,kCell] + ErAC[jCell-1,kCell]))/dr ; Ezfc[i,1] += ((EzDC[jCell,kCell+1] + EzAC[jCell,kCell+1])-(EzDC[jCell,kCell-1] + EzAC[jCell,kCell-1]))/dz       
        for j in range(Ni): # solve the electric field exerted by each ion
            rdist = (vf[j,0]-vc[i,0]) ; zdist = (vf[j,1]-vc[i,1])
            sqDist[i,j] = (rdist)**2 + (zdist)**2 #distance from particle to cell
            projR[i,j] = rdist/sqDist[i,j]**(1/2) ; projZ[i,j] = zdist/sqDist[i,j]**(1/2) #cos theta to project E field to z basis and sin to r basis
            Erfc[i,0] += -projR[i,j]*vf[j,4]/(C1*sqDist[i,j]) ; Ezfc[i,0] += -projZ[i,j]*vf[j,4]/(C1*sqDist[i,j]) # add fields in r and z   
            # I just need to add the gradient field from these now and the colliding particle should rebound
            Erfc[i,1] += 2*projR[i,j]*vf[j,4]/(C1*sqDist[i,j]**(3/2)) ; Ezfc[i,1] += 2*projZ[i,j]*vf[j,4]/(C1*sqDist[i,j]**(3/2)) # add fields in r and z                 
    pR = np.zeros(Nc); pZ = np.zeros(Nc); pTot = np.zeros(Nc)
    for i in range(Nc):    # a dipole is induced in the direction of the electric field vector with the positive pseudoparticle in the positive field direction
        if vc[i,6]!=0.0: # if there is a dipole moment that can be obtained
            pR[i] = -2*np.pi*eps0*vc[i,6]*Erfc[i,0] # dipole in r in SI units note this factor of 2 pi epsilon0 which corrects the units of m^-3 on alpha and Volts/m on E to give a dipole moment in Coulomb*meters
            pZ[i] = -2*np.pi*eps0*vc[i,6]*Ezfc[i,0] # dipole in z in SI units ###FIX THIS###
            pTot[i] = (pR[i]**2+pZ[i]**2)**(1/2) # total dipole length in physical units
            # we can now induce the force on the dipole
            Fr = abs(pR[i])*Erfc[i,1] ; Fz = abs(pZ[i])*Ezfc[i,1]
            #then we would need to convert back to virtual units once we apply the forces
            vc[i,2] += Fr*dt/(vc[i,5]) ; vc[i,3] += Fz*dt/(vc[i,5]) # update velocity with F*t/m                 
    # we then solve for the fields the collisional particles exert on the ions from the dipole (and quadrapole potentially) as well as the charge if the particle has one
    for i in range(Ni): # for each ion in the trap
        for j in range(Nc): # apply the field from each collisional particle
            # the dipole field is (3*(dipole moment dotted with vector from particle to ion)(in vector from particle to ion) - (dipole moment))/(3*pi*eps0*distance^3)
            # at close proximity, it should be treated as monopoles with charge qe separated by (dipole moment)/qe distance along the dipole moment vector
            # for now we treat the electric field it exerts as a pure dipole
            if vc[j,6]!=0.0: # if there is a potential dipole moment
                Rhatr = projR[j,i] ; Rhatz = projZ[j,i]
                dist = sqDist[j,i]**(1/2)
                Erfi[i] += -abs(pR[j])*(2*Rhatr)/(C1*dist**3) ; Ezfi[i] += -abs(pZ[j])*(2*Rhatz)/(C1*dist**3) # add dipole fields
    return vc,Erfi,Ezfi,Erfc,Ezfc # the ion electric fields are just from the collisional particles, the collisional electric fields are from all sources
# note that I haven't applied electric fields from collisional particles onto each other


@numba.njit
def mcCollision(vf,rc,zc,vrc,vzc,qc,mc,ac,Nt,dtSmall,RF,DC,Nr,Nz,dr,dz,dtLarge,dtCollision,eii = 0.01,eid = 0.01):
    """    This sim runs one collisional particle through the trap until it exits, an ion exits, or ions cross
    vf is the ion array, the first index is the ion, the second index is r,z,vr,vz,q,m,polarizability
    rc,zc,vrc,vzc,qc,mc,ac are the initial parameters of the collisional particle
    Nt,Nr,Nz are the number of time steps and cells in each dimension
    dt,dr,dz,dvr,dvz are the time step, cell size, and speed for 1cell/1timestep
    Er,Ez are the background electric fields both radially and axially
    dtLarge is the timestep when no collisional particle is in the simulation, 
    dtCollision is the minimum timestep that can be used when a collision is actively occuring, dtSmall is all other cases. 
    """
    reorder = 0
    # Start by defining some constants
    Nrmid = (Nr-1)/2; Nzmid = (Nz-1)/2
    Ni = len(vf[:,1]); Nc = 1
    vc= np.zeros((Nc,7)); vc[0,:] = [rc,zc,vrc,vzc,qc,mc,ac] # initialize collisional particle 
    zs = np.zeros((Ni,Nt)); zmeans = np.zeros(Nt); rs = np.zeros((Ni,Nt)); rmeans = np.zeros(Nt)
    vrs = np.zeros((Ni,Nt)); vzs = np.zeros((Ni,Nt)); aveSpeeds = np.zeros((2,Nt))
    rcolls = np.zeros((Nc,Nt)); zcolls = np.zeros((Nc,Nt)); vrcolls = np.zeros((Nc,Nt)); vzcolls = np.zeros((Nc,Nt)) 
    # we assume that the collisional particle was initialized far enough away to be in the large distance scale zone
    dtNow = dtSmall
    nullFields = np.zeros((Nr,Nz))
    for i in range(Nt):       
        # Now we apply the time step calculation
        rid,rii,vid,vii = minDists(vf,vc)
        collision = collisionMode(rii,rid,vc[0,6],0.1)
        if collision:
             #println("Collision time-scale")
            dtNow = rid*eid/(5*vid) # if a collision is occuring, we set the time-step to limit the change of ion-dipole force between time-steps to be eid as a fraction of the starting value
        else:
            dtNow = dtSmall # otherwise we ensure eii fractional change in ion-ion forces or dtLarge as the maximum possible time step
        if dtNow < dtCollision:
            dtNow = dtCollision

        Erfi,Ezfi = solveFields(vf,nullFields,DC,RF,nullFields,Nrmid,Nzmid,Ni,dr,dz) # solve fields from ions on ions

        if vc[0,5]<1e6: #if the collisional particle exists
            #dtNow = dtSmall 
            vc,Erfic,Ezfic,Erfc,Ezfc = collisionParticlesFields(vf,vc,Ni,nullFields,DC,RF,nullFields,dr,dz,dtNow,Nrmid,Nzmid) # solve fields from ions on collision particles and vice versa, also updates collisional particle velocity
            vc = updatePoss(vc,dr,dz,dtNow,Nr,Nz,Nrmid,Nzmid) # updates collisional particle positions
            Erfi += Erfic; Ezfi += Ezfic # add the collisional fields  
        else:
             dtNow = dtLarge
        vf = updateVels(vf,Erfi,Ezfi,dtNow,Nrmid,Nzmid) # fields apply force to ions
        vf = updatePoss(vf,dr,dz,dtNow,Nr,Nz,Nrmid,Nzmid) # velocities carry ions
        if np.any(np.isnan(vf)): # end sim if ion values break
            print("NaN detected in ion parameters before laserCool!!") ; print("vf = ",vf) ; print("i = ",i)
            vf[:,0] = 0.0; vf[:,1] = 0.0; vf[:,2] = 0.0; vf[:,3] = 0.0; vf[:,4] = 0.0; vf[:,5] = 1e1; vf[:,6] = 0.0
            vc[:,0] = 0.0; vc[:,1] = 0.0; vc[:,2] = 0.0; vc[:,3] = 0.0; vc[:,4] = 0.0; vc[:,5] = 1e1; vc[:,6] = 0.0
            break
        rs[:,i]=vf[:,0]; rcolls[:,i]=vc[:,0]; zs[:,i]=vf[:,1]; zcolls[:,i]=vc[:,1]
        vrs[:,i]=vf[:,2]; vzs[:,i]=vf[:,3]; vrcolls[:,i]=vc[:,2]; vzcolls[:,i]=vc[:,3]
        if np.any(np.isnan(vf)): # end sim if ion values break
            print("NaN detected in ion parameters after laserCool!!") ; print("vf = ",vf) ; print("i = ",i)
            vf[:,0] = 0.0; vf[:,1] = 0.0; vf[:,2] = 0.0; vf[:,3] = 0.0; vf[:,4] = 0.0; vf[:,5] = 1e1; vf[:,6] = 0.0
            vc[:,0] = 0.0; vc[:,1] = 0.0; vc[:,2] = 0.0; vc[:,3] = 0.0; vc[:,4] = 0.0; vc[:,5] = 1e1; vc[:,6] = 0.0
            break
        if np.sum(vf[:,5])>1e5: # end sim if ion leaves
            #print("Ion Ejected") ; print("vf = ",vf) ; print("i = ",i)
            reorder += 2
            break
        for j in range(1,Ni):
            if zs[j,i]>zs[j-1,i]: # end sim if neighboring ions have reordered
                reorder += 1
                #print("Crossing at timestep ",i," of ion ",j) 
                Nt = i+1000 # do 5000 more time steps after a crossing is detected
                #break
        #if crossTest>0:
        #    break
    #return rs,zs,vrs,vzs,rcolls,zcolls,vrcolls,vzcolls, reorder
    return reorder