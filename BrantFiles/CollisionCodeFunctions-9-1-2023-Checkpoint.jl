# Define Helper Functions
function ptovPos(pos,Nmid,dcell)  # converts from physical to virtual units in position
    return (pos/dcell + float(Nmid)) # fractional position in vertual units
end

function vtopPos(pos,Nmid,dcell) # converts from virtual to physical units in position (natually placing the point at the center of a cell)
    return float((pos-Nmid))*dcell # returns the center of the cell in physical units
end

function solveFields(vf,ErDC,EzDC,ErAC,EzAC,Nrmid,Nzmid,Ni,dr,dz) 
    """ takes in particle positionsand spits out electric fields at all ions
    vf is the vector of ion parameters
    ErDC is the array of electric fields from the DC electrodes in r
    EzDC is the array of electric fields from the DC electrodes in z
    ErAC is the array of electric fields from the AC electrodes in r
    EzAC is the array of electric fields from the AC electrodes in z
    Nrmid, Nzmid, are the midpoints of the grid
    dr and dz are the cell dimensions
    can improve this to replace the grid with E-fields by cell with a vector of E-fields by ion
    """
    eps0 = 8.854e-12 ; C1 = 4*pi*eps0 #SI units 
    Erf2 = np.zeros(Ni); Ezf2 = np.zeros(Ni)
    for i=1:length(vf[:,1]) # note that this no longer takes the electric field from all particles, so chillax 
        jCell = ptovPos(vf[i,1],Nrmid,dr)
        kCell = ptovPos(vf[i,2],Nzmid,dz)
        jCell = convert(UInt32,round(jCell)) # local cell index in r
        kCell = convert(UInt32,round(kCell)) # local cell index in z
        Erf2[i] += ErDC[jCell,kCell] + ErAC[jCell,kCell]
        Ezf2[i] += EzDC[jCell,kCell] + EzAC[jCell,kCell]
        for j=1:length(vf[:,1])
            if j!=i #here we solve for the E-fields in this cell from the particles outside this cell
                rdist = (vf[j,1]-vf[i,1])
                zdist = (vf[j,2]-vf[i,2])
                sqDist = (rdist)^2 + (zdist)^2 #distance from particle to cell
                projR = rdist/sqDist^(1/2) #sin theta to project E field to r basis
                projZ = zdist/sqDist^(1/2) #cos theta to project E field to z basis
                Erf2[i] += -projR*vf[j,5]/(C1*sqDist) # add fields in r
                Ezf2[i] += -projZ*vf[j,5]/(C1*sqDist) # add fields in z 
            end
        end        
    end
    return Erf2,Ezf2
end

function ACFields(ErAC0,EzAC0,phaseAC,f,t) # returns AC fields at each grid cell based on the amplitude at each cell, starting phase, current time, and frequency
    return ErAC0.*sin.(phaseAC+f*t*2*pi),EzAC0.*sin.(phaseAC+f*t*2*pi)
end

function updateVels(vf,Erf,Ezf,dt,Nrmid,Nzmid) # apply the forces from the local electric field over one time step
    for i=1:length(vf[:,1]) # apply E-field of the cell
        Fr = vf[i,5]*Erf[i] ; Fz = vf[i,5]*Ezf[i] # solve force 
        # then we would need to convert back to virtual units 
        vf[i,3] += Fr*dt/(vf[i,6]) ; vf[i,4] += Fz*dt/(vf[i,6]) # update velocity with F*t/m
    end
    return vf
end

function updatePoss(vf,dr,dz,dt,Nr,Nz,Nrmid,Nzmid) # move the particles one time step
    for i=1:length(vf[:,1]) 
        vf[i,1] += vf[i,3]*dt; vf[i,2] += vf[i,4]*dt # correct if physical since one time step is passing
        rCell = ptovPos(vf[i,1],Nrmid,dr); zCell = ptovPos(vf[i,2],Nzmid,dz) 
        if rCell>Nr-1 || rCell<2
            println("Position out of simulation in r ",vf[i,1]," Particle # ",i," vf = ",vf[i,:])
            println("dt = ",dt)
            vf[i,:] = vf[i,:].*0.0; vf[i,1] = 2.0; vf[i,2] = 2.0; vf[i,3] = 0.0; vf[i,4] = 0.0; vf[i,6] = 1e6
            
        elseif zCell>Nz-1 || zCell<2
            println("Position out of simulation in z ",vf[i,2]," Particle # ",i," vf = ",vf[i,:])
            println("dt = ",dt)
            vf[i,:] = vf[i,:].*0.0; vf[i,1] = 2.0; vf[i,2] = 2.0; vf[i,3] = 0.0; vf[i,4] = 0.0; vf[i,6] = 1e6
        end
    end
    return vf
end

function makeRF0(m,q,w,Nr,Nz,Nrmid,dr) # dummy function to generate the RF field in our grid assuming that it is a harmonic potential about (0,0) and focuses in x
    # We take in the mass, frequency for that mass, cell numbers, midpoint, and physical width of a cell and output the RF electric fields (constant in z) as a function of radial cell
    C = -m*(w^2)/q ; RF = np.ones((Nr,Nz))
    for jCell=1:Nr
        RF[jCell,:] = -RF[jCell,:].*C*(Nrmid-jCell)*dr # electric field in pseudo-potential and harmonic potential approximation
    end
    return RF
end

function makeDC(m,q,w,Nz,Nr,Nzmid,dz) # dummy function to generate the DC fields assuming that it is a harmonic potential about (0,0) and focuses in z
    # We take in the mass, frequency for that mass, cell numbers, midpoint, and physical width of a cell and output the DC electric fields (constant in r) as a function of longitudinal cell
    C = -m*(w^2)/q ; DC = np.ones((Nr,Nz))
    for kCell=1:Nz
        DC[:,kCell] = -DC[:,kCell].*C*(Nzmid-kCell)*dz # electric field for DC in harmonic potential approximation 
    end
    return DC
end

function makeVField(m,q,wr,wz,Nr,Nz,Nrmid,Nzmid,dr,dz)
    # we assign voltage at each point given our trapping frequencies
    Cr = -0.5*m*(wr^2)/q ; Cz = -0.5*m*(wz^2)/q ; Vf = np.ones((Nr,Nz))
    for jCell=1:Nr
        for kCell=1:Nz
            Vf[jCell,kCell] = Cr*((Nrmid-jCell)*dr)^2 + Cz*((Nzmid-kCell)*dz)^2 # makes a harmonic potential in each axis, adds them
        end
    end
    return Vf
end

function makeVf(Ni,q,m,l,dsepz,offsetr,offsetz,vbumpr,vbumpz)
    # this makes an initial array for the ions, the first index is the ion, the second index is r,z,vr,vz,q,m,polarizability [treated as zero]
    vf = np.zeros((Ni,7)); Nimid = (Ni+1)/2.0
    for i=1:Ni
        vf[i,:] = [0.0e-6,0.0e-6+(Nimid-i)*dsepz,0,0,q,m,0.0]
    end
    vf[l,1] += offsetr ; vf[l,2] += offsetz
    vf[l,3] += vbumpr ; vf[l,4] += vbumpz
    return vf
end

function particleE(vr,vz,m)
    # takes in velocities and mass and returns energy
    return (1/2)m*(vr^2 + vz^2)
end

function particleP(v,m)
    # takes in velocity and mass and returns momentum
    return m*v
end

function totalEandP(vrs,vzs,ms)
    # takes in vectors of velocities and masses and returns total energy and momenta
    En = 0.0; pr = 0.0; pz = 0.0
    for i=1:length(vrs)
        En+=particleE(vrs[i],vzs[i],ms) ; pr+=particleP(vrs[i],ms) ; pz+=particleP(vzs[i],ms)
    end
    return [En,pr,pz]
end

function farApart(vf,vc,dCut) 
    # we're only checking ion-dipole distances, checks if anything is closer together than dCut
    dCut2 = dCut^2
    Ni = length(vf[:,1]); Nc = length(vc[:,1])
    for i=1:Ni
        for j=1:Nc
            d2 = ((vf[i,1]-vc[j,1]))^2 + ((vf[i,2]-vc[j,2]))^2
            if d2 < dCut2
                return false
            end
        end
    end

    
    return true
end

function plotThing(Nc,Ni,dt,colls,ions,first,last,title1,xlabel1,ylabel1)
    """
    Makes one continuous plot
    Nc,Ni are the numbers of collisional particles and ions to plot
    first,last are the bounds 
    colls,ions are the arrays of what we want to plot for, like position of each ion over time (we assume time is linear) 
    dt is the time step and the step of the thing we want to plot for
    """
    # Now we plot their positions over time
    for i=1:Nc
        plot(dt.*range(first,last),colls[i,first:last])
    end
    for i=1:Ni
        plot(dt.*range(first,last),ions[i,first:last])
    end
    xlabel(xlabel1) ; ylabel(ylabel1) ; title(title1)
    show()    
end

function plotPieceWise(Nc,Ni,colls1,ions1,colls2,ions2,first,last,title1,xlabel1,ylabel1,xlow,xhigh,ylow,yhigh)
    """
    Makes a scatter plot of any two things against each other
    Nc,Ni are the numbers of collisional particles and ions to plot
    first,last give the slice of the attributes we wish to plot against each other
    colls,ions are the vectors of what we want to plot for 1 denotes the x-axis variable, 2 for the y-axis variable
    title1 is a title, xlabel1 and ylabel1 are axes labels, x/y low/high are axes bounds
    """
    for i=1:Nc
        scatter(colls1[i,first:last],colls2[i,first:last])
    end
    for i=1:Ni
        scatter(ions1[i,first:last],ions2[i,first:last])
    end
    xlabel(xlabel1) ; ylabel(ylabel1) ; title(title1)
    ax = gca()
    ax[:set_xlim]([xlow,xhigh]) ; ax[:set_ylim]([ylow,yhigh])
    show()
end

function subPlotThings(N1,N2,dt,thing1,thing2,first,last,title1,xlabel1,ylabel1,ylabel2)
    """
    Makes two plots that share an x-axis of time
    N1,N2 are the numbers of vectors of things to plot in each array thing1 and 2 (number of particles)
    first,last are the bounds 
    thing1,thing2 are the vectors of what we want to plot for 
    dt is the time step assuming linear time
    first,last give the slice of thing1 and 2
    title1,x/ylabel1/2 give a title and axes labels
    """
    subplot(211)
    for i=1:N1
        plot(dt.*range(first,last),thing1[i,first:last])
    end
    ylabel(ylabel1)
    subplot(212)
    for i=1:N2
        plot(dt.*range(first,last),thing2[i,first:last])
    end
    xlabel(xlabel1) ; ylabel(ylabel2) ; suptitle(title1)
    show()
end 

# Define Primary Functions
function updatePoss(vf,dr,dz,dt,Nr,Nz,Nrmid,Nzmid) 
    """ This moves our vf particles as their velocity suggests for one time step
    vf is the vector of ion parameters, the first index is the ion, the second index is r,z,vr,vz,q,m,polarizability
    dr,dz are the cell dimensions of our trapping field grid
    dt is the timestep 
    Nr,Nz are the grid sizes and Nrmid,Nzmid are the midpoints
    """
    for i=1:length(vf[:,1]) 
        vf[i,1] += vf[i,3]*dt ; vf[i,2] += vf[i,4]*dt # correct if physical since one time step is passing
        rCell = ptovPos(vf[i,1],Nrmid,dr) ; zCell = ptovPos(vf[i,2],Nzmid,dz) 
        if rCell>Nr-1 || rCell<2
            println("Position out of simulation in r ",vf[i,1]," Particle # ",i," vf = ",vf[i,:]) ; println("dt = ",dt)
            vf[i,:] = vf[i,:].*0.0 ; vf[i,1] = 2.0 ; vf[i,2] = 2.0 ; vf[i,3] = 0.0 ; vf[i,4] = 0.0 ; vf[i,6] = 1e6
        elseif zCell>Nz-1 || zCell<2
            println("Position out of simulation in z ",vf[i,2]," Particle # ",i," vf = ",vf[i,:]) ; println("dt = ",dt)
            vf[i,:] = vf[i,:].*0.0 ; vf[i,1] = 2.0 ; vf[i,2] = 2.0; vf[i,3] = 0.0 ; vf[i,4] = 0.0 ; vf[i,6] = 1e6
        end
    end
    return vf
end

function solveFields(vf,ErDC,EzDC,ErAC,EzAC,Nrmid,Nzmid,Ni,dr,dz) 
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
    eps0 = 8.854e-12 ; C1 = 4*pi*eps0 #SI units 
    Erf2 = np.zeros(Ni) ; Ezf2 = np.zeros(Ni)
    for i=1:length(vf[:,1]) # note that this no longer takes the electric field from all particles, so chillax 
        jCell = ptovPos(vf[i,1],Nrmid,dr) ; kCell = ptovPos(vf[i,2],Nzmid,dz) # 
        jCell = convert(UInt32,round(jCell)) ; kCell = convert(UInt32,round(kCell)) # local cell index in r and z
        Erf2[i] += ErDC[jCell,kCell] + ErAC[jCell,kCell] ; Ezf2[i] += EzDC[jCell,kCell] + EzAC[jCell,kCell] # add trap fields
        for j=1:length(vf[:,1])
            if j!=i #here we solve for the fields from each other ion on this ion
                rdist = (vf[j,1]-vf[i,1]) ; zdist = (vf[j,2]-vf[i,2]) # get each distance
                sqDist = (rdist)^2 + (zdist)^2 #square distance from particle to cell
                projR = rdist/sqDist^(1/2) ; projZ = zdist/sqDist^(1/2) # cos theta to project E field to z basis, sin to r basis               
                Erf2[i] += -projR*vf[j,5]/(C1*sqDist) ; Ezf2[i] += -projZ*vf[j,5]/(C1*sqDist) # add fields in r and z 
            end
        end        
    end
    return Erf2,Ezf2
end

function laserInt(m,I0,nul,sigl,A21,nua,fwhma,bins=2000,range=10,vels=10001,velMax=10)
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
    range is the number of frequency widths to consider when binning the interaction strength vector
    A21 is the einstein A coefficient of the atomic transition
    I0 is the laser intensity integrated over frequency [W/m^2]    
    nu is a 2D array where [1,:] is the velocities along the laser direction and [2,:] are the absorption rates for those velocities and [3,:] are the energy differences between absorbed and emitted photon
    """
    h = 6.626e-34 # planck constant [J*s]
    c = 2.998e8 # speed of light [m/s]
    nu = np.zeros((3,bins)) ; nuj = np.zeros((3,vels))
    println("In laserInt: I0 = ",I0) ; println("nul = ",nul) ; println("sigl = ",sigl)
    println("A21 = ",A21) ; println("nua = ",nua) ; println("fwhma = ",fwhma)    
    if sigl < fwhma/2.0 # set center frequency and scan width to encompass whole possible lineshape (smaller lineshape dominates)
        d=range*sigl ; nu0=nul
    else
        d=range*fwhma/2.0 ; nu0=nua
    end
    delta = 2.0*d/bins # step size between bins
    for j=1:vels
        vel = -velMax + (j-1)*2*velMax/(vels-1) # set velocity for jth bin
        nulj = nul*(1-vel/c)/(1+vel/c) # laser frequency for jth velocity
        nuj[1,j] = vel;
        for i=1:bins # do the integral
            nuInt = nu0 - d + (i-0.5)*delta # sets frequency to the middle of the bin scanning from nu0 - d to nu0 + d
            gl = exp(-(nuInt-nulj)^2/(2*sigl^2)) # laser lineshape minus constants
            ga = ((nuInt - nua)^2 + (1.0/4.0)*fwhma^2)^-1 # atomic lineshape minus constants
            nu[2,i] = gl*ga*delta # width of the rectangle with height of the center of the bin for lineshape and width of the bin
        end
        nu[2,:] = nu[2,:].*(c^2*fwhma*A21*I0/(16*sqrt(2)*pi^(5/2)*h*sigl*nu0^3)) # scale the interaction rate by the various constants
        nuj[2,j] = np.sum(nu[2,:]) #set interaction rate to sum of interaction rates
        if nuj[2,j]>1.0*A21
            nuj[3,j] = (m/2)*( (h*nulj/(m*c))^2 + (h*nua/(m*c))^2 -2*(nuj[1,j])*(h*nua/(m*c)) + 2*(nuj[1,j])*(h*nulj/(m*c)) -2*(h*nua/(m*c))*(h*nulj/(m*c)) )  # if stimulated emission regime, recoils from absorption in laser direction then recoils from emission against laser direction
        else
            nuj[3,j] = (m/2)*( (h*nulj/(m*c))^2 + (h*nua/(m*c))^2 + 2*(h*nulj/(m*c))*(nuj[1,j]) ) # if spontaneous emission regime, cools by 
        end
    end
    plot(nuj[1,:],nuj[2,:])
    return nuj
end

function laserCoolSmoothNew(vf,vl,nu,Ni,dt) # this takes the particle population and applies a global laser cooling field coupled to the radial and axial velocities that purely reduces the velocity like a true damping force
    """
    vf is the vector of ion parameters, the first index is the ion, the second index is r,z,vr,vz,q,m,polarizability
    vl is the laser propagation vector (vr,vz) normalized to 1
    Ni is number of ions, dt is time step
    nu is the array where nu[1,:] are the velocities in the laser direction that interact with the atom and nu[2,;] are the absorption rates of those frequencies
    """
    Nnu = length(nu[1,:])
    for i=1:Ni
        println("ion = ",i)
        if vf[i,6]<1.0 #if the particle exists it should have a realistic mass
            vil = vf[i,3]*vl[1]+vf[i,4]*vl[2] #velocity of ion in laser propagation direction
            rate = 0.0 ; dE = 0.0
            first = searchsortedfirst(nu[1,:],vil)
            if first > length(nu[1,:])
                println("Error, velocity outside scope of laserInt")
                println(" velocity = ",vil)
            end
            rate = nu[2,first] ; dE = -nu[3,first]
            dv = (2*rate*dt*abs(dE)/vf[i,6])^(1/2)
            if dE>0
                println("Cool dv = ",dv," vl = ",vl," change = ",vl[1]*dv)
                vf[i,3] = vf[i,3]+vl[1]*dv*sign(vf[i,3]) # radial velocity reduction from absorption
                vf[i,4] = vf[i,4]+vl[2]*dv*sign(vf[i,4]) # axial velocity reduction from absorption
            end
            if dE<0
                println("Heat dv = ",dv," vl = ",vl," change = ",-vl[1]*dv)
                vf[i,3] = vf[i,3]- vl[1]*dv*sign(vf[i,3]) # radial velocity reduction from absorption
                vf[i,4] = vf[i,4]- vl[2]*dv*sign(vf[i,4]) # axial velocity reduction from absorption
            end
        end
    end
    return vf # we return the updated populations
end

function collisionParticlesFields(vf,vc,Ni,ErDC,EzDC,ErAC,EzAC,dr,dz,dt,Nrmid,Nzmid) # this applies the fields of all existing collisional particles and changes the velocity of those collisional particles
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
    Nc = length(vc[:,1])
    # we begin by instantiating the electric field lists (fields are in physical units)
    Erfi = np.zeros(Ni); Ezfi = np.zeros(Ni)
    Erfc = np.zeros((Nc,2)); Ezfc = np.zeros((Nc,2)) # [i,1] is middle field, 2 is high index - low index divided by the size of a cell (ie, the local slope of the E-field)
    sqDist = np.zeros((Nc,Ni)); projR = np.zeros((Nc,Ni)); projZ = np.zeros((Nc,Ni))
    C1 = 4*pi*eps0 # commonly used set of constants put together
    # we solve the electric fields on the collisional particles
    for i=1:Nc # for each collisional particle that exists        
        # In order to allow for electric field gradients of the background field, here we need to implement a linear E-field gradient between neighboring cells
        jCell = ptovPos(vf[i,1],Nrmid,dr) ; kCell = ptovPos(vf[i,2],Nzmid,dz)
        jCell = convert(UInt32,round(jCell)) ; kCell = convert(UInt32,round(kCell)) # local cell index in r and z
        # we initialize the interpolated field for each 
        Erfc[i,1] += (ErDC[jCell,kCell] + ErAC[jCell,kCell]) ; Ezfc[i,1] += (EzDC[jCell,kCell] + EzAC[jCell,kCell])
        Erfc[i,2] += ((ErDC[jCell+1,kCell] + ErAC[jCell+1,kCell])-(ErDC[jCell-1,kCell] + ErAC[jCell-1,kCell]))/dr ; Ezfc[i,2] += ((EzDC[jCell,kCell+1] + EzAC[jCell,kCell+1])-(EzDC[jCell,kCell-1] + EzAC[jCell,kCell-1]))/dz       
        for j=1:Ni # solve the electric field exerted by each ion
            rdist = (vf[j,1]-vc[i,1]) ; zdist = (vf[j,2]-vc[i,2])
            sqDist[i,j] = (rdist)^2 + (zdist)^2 #distance from particle to cell
            projR[i,j] = rdist/sqDist[i,j]^(1/2) ; projZ[i,j] = zdist/sqDist[i,j]^(1/2) #cos theta to project E field to z basis and sin to r basis
            Erfc[i,1] += -projR[i,j]*vf[j,5]/(C1*sqDist[i,j]) ; Ezfc[i,1] += -projZ[i,j]*vf[j,5]/(C1*sqDist[i,j]) # add fields in r and z   
            # I just need to add the gradient field from these now and the colliding particle should rebound
            Erfc[i,2] += 2*projR[i,j]*vf[j,5]/(C1*sqDist[i,j]^(3/2)) ; Ezfc[i,2] += 2*projZ[i,j]*vf[j,5]/(C1*sqDist[i,j]^(3/2)) # add fields in r and z             
        end
    end      
    pR = np.zeros(Nc); pZ = np.zeros(Nc); pTot = np.zeros(Nc)
    for i=1:Nc    # a dipole is induced in the direction of the electric field vector with the positive pseudoparticle in the positive field direction
        if vc[i,7]!=0.0 # if there is a dipole moment that can be obtained
            pR[i] = -2*pi*eps0*vc[i,7]*Erfc[i] # dipole in r in SI units note this factor of 2 pi epsilon0 which corrects the units of m^-3 on alpha and Volts/m on E to give a dipole moment in Coulomb*meters
            pZ[i] = -2*pi*eps0*vc[i,7]*Ezfc[i] # dipole in z in SI units
            pTot[i] = (pR[i]^2+pZ[i]^2)^(1/2) # total dipole length in physical units
            # we can now induce the force on the dipole
            Fr = abs(pR[i])*Erfc[i,2] ; Fz = abs(pZ[i])*Ezfc[i,2]
            #then we would need to convert back to virtual units once we apply the forces
            vc[i,3] += Fr*dt/(vc[i,6]) ; vc[i,4] += Fz*dt/(vc[i,6]) # update velocity with F*t/m                 
        end
    end
    # we then solve for the fields the collisional particles exert on the ions from the dipole (and quadrapole potentially) as well as the charge if the particle has one
    for i=1:Ni # for each ion in the trap
        for j=1:Nc # apply the field from each collisional particle
            # the dipole field is (3*(dipole moment dotted with vector from particle to ion)(in vector from particle to ion) - (dipole moment))/(3*pi*eps0*distance^3)
            # at close proximity, it should be treated as monopoles with charge qe separated by (dipole moment)/qe distance along the dipole moment vector
            # for now we treat the electric field it exerts as a pure dipole
            if vc[j,7]!=0.0 # if there is a potential dipole moment
                Rhatr = projR[j,i] ; Rhatz = projZ[j,i]
                dist = sqDist[j,i]^(1/2)
                Erfi[i] += -abs(pR[j])*(2*Rhatr)/(C1*dist^3) ; Ezfi[i] += -abs(pZ[j])*(2*Rhatz)/(C1*dist^3) # add dipole fields
            end
        end
    end
    return vc,Erfi,Ezfi,Erfc,Ezfc # the ion electric fields are just from the collisional particles, the collisional electric fields are from all sources
end # note that I haven't applied electric fields from collisional particles onto each other

function runFasterCollision(vf,rc,zc,vrc,vzc,qc,mc,ac,Nt,dtSmall,Er,Ez,Nr,Nz,dr,dz,vl,I0,nul,dnul,nu0,A,Ti,dtLarge,dCutOut,dtMid,dCutIn,laserCool=true)
    """    This sim runs one collisional particle through the trap until it exits, an ion exits, or ions cross
    vf is the ion array, the first index is the ion, the second index is r,z,vr,vz,q,m,polarizability
    rc,zc,vrc,vzc,qc,mc,ac are the initial parameters of the collisional particle
    Nt,Nr,Nz are the number of time steps and cells in each dimension
    dt,dr,dz,dvr,dvz are the time step, cell size, and speed for 1cell/1timestep
    Er,Ez are the background electric fields both radially and axially
    vl,I0,nul,dnul,nu0,A,Ti are laser cooling parameters
    dtLarge is the timestep when no particles are closer than dCutOut, dtMid when none closer than dCutIn, dtSmall is the timestep when at least one is inside dCutIn
    """
    # Start by defining some constants
    Nrmid = (Nr-1)/2; Nzmid = (Nz-1)/2
    Ni = length(vf[:,1]); Nc = 1
    vc= np.zeros((1,7)); vc[1,:] = [rc,zc,vrc,vzc,qc,mc,ac] # initialize collisional particle 
    zs = np.zeros((Ni,Nt)); zmeans = np.zeros(Nt); rs = np.zeros((Ni,Nt)); rmeans = np.zeros(Nt)
    vrs = np.zeros((Ni,Nt)); vzs = np.zeros((Ni,Nt)); aveSpeeds = np.zeros((2,Nt))
    rcolls = np.zeros((Nc,Nt)); zcolls = np.zeros((Nc,Nt)); vrcolls = np.zeros((Nc,Nt)); vzcolls = np.zeros((Nc,Nt)) 
    # we assume that the collisional particle was initialized far enough away to be in the large distance scale zone
    dtLast = dtLarge; dtNow = dtLarge
    dvr = dr/dtNow; dvz = dz/dtNow
    nu = laserInt(vf[1,6],I0,nul,dnul,A,nu0,A/(2*pi)) # we get those cool laser cooling rates these frequencies better be normal and not angular
    for i=1:Nt       
        # Now we apply the time step for the various 
        Erfi,Ezfi = solveFields(vf,nullFields,Ez,Er,nullFields,Nrmid,Nzmid,Ni,dr,dz) # solve fields from ions on ions
        if vc[1,6]<1e6 #if the collisional particle exists
            vc,Erfic,Ezfic,Erfc,Ezfc = collisionParticlesFields(vf,vc,Ni,nullFields,DC,RF,nullFields,dr,dz,dtNow,Nrmid,Nzmid) # solve fields from ions on collision particles and vice versa, also updates collisional particle velocity
            vc = updatePoss(vc,dr,dz,dtNow,Nr,Nz,Nrmid,Nzmid) # updates collisional particle positions
            Erfi += Erfic; Ezfi += Ezfic # add the collisional fields           
        end
        vf = updateVels(vf,Erfi,Ezfi,dtNow,Nrmid,Nzmid) # fields apply force to ions
        vf = updatePoss(vf,dr,dz,dtNow,Nr,Nz,Nrmid,Nzmid) # velocities carry ions
        if any(isnan, vf) # end sim if ion values break
            println("NaN detected in ion parameters before laserCool!!") ; println("vf = ",vf) ; println("i = ",i)
            vf[:,1] .= 0.0; vf[:,2] .= 0.0; vf[:,3] .= 0.0; vf[:,4] .= 0.0; vf[:,5] .= 0.0; vf[:,6] .= 1e1; vf[:,7] .= 0.0
            vc[:,1] .= 0.0; vc[:,2] .= 0.0; vc[:,3] .= 0.0; vc[:,4] .= 0.0; vc[:,5] .= 0.0; vc[:,6] .= 1e1; vc[:,7] .= 0.0
            break
        end
        if laserCool
            println("Laser cooling! i = ",i) ; println("vf speeds beffore cooling = ",vf[1,3:4])
            vf = laserCoolSmoothNew(vf,vl,nu,Ni,dtNow) # apply smoothed laser cooling (fractional photon emission and average momentum and energy change approximations)
            println("vf speeds after cooling = ",vf[1,3:4])
        end
        rs[:,i]=vf[:,1]; rcolls[:,i]=vc[:,1]; zs[:,i]=vf[:,2]; zcolls[:,i]=vc[:,2]
        vrs[:,i]=vf[:,3]; vzs[:,i]=vf[:,4]; vrcolls[:,i]=vc[:,3]; vzcolls[:,i]=vc[:,4]
        if any(isnan, vf) # end sim if ion values break
            println("NaN detected in ion parameters after laserCool!!") ; println("vf = ",vf) ; println("i = ",i)
            vf[:,1] .= 0.0; vf[:,2] .= 0.0; vf[:,3] .= 0.0; vf[:,4] .= 0.0; vf[:,5] .= 0.0; vf[:,6] .= 1e1; vf[:,7] .= 0.0
            vc[:,1] .= 0.0; vc[:,2] .= 0.0; vc[:,3] .= 0.0; vc[:,4] .= 0.0; vc[:,5] .= 0.0; vc[:,6] .= 1e1; vc[:,7] .= 0.0
            break
        end
        if np.sum(vf[:,6])>1e5 # end sim if ion leaves
            println("Ion Ejected") ; println("vf = ",vf) ; println("i = ",i)
            break
        end
        crossTest = 0
        for j=2:Ni
            if zs[j,i]>zs[j-1,i] # end sim if neighboring ions have reordered
                crossTest+=1
                print("Crossing at ",i," of ion ",j) 
                break
            end
        end
        if crossTest>0
            break
        end
    end   
    return rs,zs,vrs,vzs,rcolls,zcolls,vrcolls,vzcolls
end

# Now we begin converting functions to assume vf is in purely physical units
function runMCCollision(Nc,vf,qc,mc,ac,Nt,dt,Er,Ez,Nr,Nz,dr,dz,dvr,dvz,vl,I0,nul,dnul,nu0,A,Ti,P,T,sigmaV,dv,vmax)
    """
    Nc is the number of collisional ions that can exist at once
    vf is the ion array
    qc,mc,ac are the initial parameters of the collisional particle
    Nt,Nr,Nz are the number of time steps and cells in each dimension
    dt,dr,dz,dvr,dvz are conversions from virtual to physical units
    Er,Ez are the background electric fields both radially and axially
    vl,I0,nul,dnul,nu0,A,Ti are laser cooling parameters
    P,T,sigmaV,dv,vmax are all parameters for the thermal distribution of collisional particles and the integration step size and maximum velocity
    This sim runs one collisional particle through the trap until it exits 
    """  
    Nrmid = (Nr-1)/2 ; Nzmid = (Nz-1)/2
    Ni = length(vf[:,1])
    vc= np.zeros((Nc,7))
    vc[:,6] .= 1e6 ; vc[:,1] .= 2 ; vc[:,2] .= 2 ; vc[:,5] .= qc ; vc[:,7] .= ac # initialize collisional particle 
    zs = np.zeros((Ni,Nt)) ; zmeans = np.zeros(Nt); rs = np.zeros((Ni,Nt)) ; rmeans = np.zeros(Nt)
    vrs = np.zeros((Ni,Nt)) ; vzs = np.zeros((Ni,Nt)) ; aveSpeeds = np.zeros((2,Nt))
    rcolls = np.zeros((Nc,Nt)) ; zcolls = np.zeros((Nc,Nt)) ; vrcolls = np.zeros((Nc,Nt)) ; vzcolls = np.zeros((Nc,Nt))    
    for i=1:Nt
        vc = monteCarloCollisionParticles(vc,P,T,Vfield,sigmaV,dt,dv,vmax,Nr,Nz,dr,dz)
        Erfi,Ezfi = solveFields(vf,nullFields,DC,RF,nullFields,Nrmid,Nzmid,Ni,dr,dz) # solve fields from ions on ions
        if sum(vc[:,6])<1e6*Nc
            vc,Erfic,Ezfic,Erfc,Ezfc = collisionParticlesFields(vf,vc,Ni,nullFields,DC,RF,nullFields,dr,dz,dvr,dvz,dt,Nrmid,Nzmid) # solve fields from ions on collision particles and vice versa and update collision particle velocities
            vc = updatePoss(vc,dt,Nr,Nz) # move collision particles
            Erfi += Erfic ; Ezfi += Ezfic # add the collisional fields
        end
        vf = updateVels(vf,Erfi,Ezfi,dt,Nrmid,Nzmid,dr,dz,dvr,dvz) # apply forces on ions
        vf = updatePoss(vf,dt,Nr,Nz) # move ions
        if any(isnan, vf) # catches nan errors in ion parameters
            println("NaN detected in ion parameters before laserCool!!") ; println("vf = ",vf) ; println("i = ",i)
            testr = convert(UInt32,round(vf[1,1])) ; testz = convert(UInt32,round(vf[1,2]))
            println("Efields = ",Erfi," , ",Ezfi)
            vf = vf.*0.0
            vf[:,1] .= 2 ; vf[:,2] .= 2 ; vf[:,6] .= 1e6
            break
        end
        vf = laserCoolSmooth(vf,vl,I0,nul,dnul,nu0,A,Ti,Ni,dt,dvr,dvz) # apply laser cooling
        rs[:,i]=vf[:,1] ; rcolls[:,i]=vc[:,1] ; zs[:,i]=vf[:,2] ; zcolls[:,i]=vc[:,2]
        vrs[:,i]=vf[:,3] ; vzs[:,i]=vf[:,4] ; vrcolls[:,i]=vc[:,3] ; vzcolls[:,i]=vc[:,4]
        if any(isnan, vf) # catches nan errors in ion parameters
            println("NaN detected in ion parameters after laserCool!!") ; println("vf = ",vf) ; println("i = ",i)
            testr = convert(UInt32,round(vf[1,1])) ; testz = convert(UInt32,round(vf[1,2]))
            println("Efields = ",Erfi," , ",Ezfi)
            vf[:,1] .= 2 ; vf[:,2] .= 2 ; vf[:,6] .= 1e6
            break
        end
        for j=1:Ni
            aveSpeeds[1,i] += vf[j,3]^2 ; aveSpeeds[2,i] += vf[j,4]^2
        end
    end
    return rs,zs,vrs,vzs,rcolls,zcolls,vrcolls,vzcolls
end

function monteCarloCollisionParticles(vc,P,T,Vf,sigmaV,dt,dv,vmax,Nr,Nz,dr,dz)
    """
    This will initialize collisional particles following a modifice Boltzmann distribution at the outer edges
    of the simulation frame with velocity dependent cross-section, and the reduced energy from climbing to the top 
    of the trapping potential well
    We assume the particle approach to go up a gaussian potential with sigmaV for its sigma
    this sigma is increased if we approach at a glancing angle (not added yet)
    This means we don't account for deflection incurred by irregularly shaped potentials
    We also don't account for directional outgassing from surfaces
    vc is the collisional particle vector, uninitialized particles have zeros for all their non-position parameters
    P is the pressure in Pa
    T is the temperature in Kelvin
    Vf is the array of potentials in each cell of the sim
    sigmaV is how steep the slope of the voltage outside of the peak is (approximating the shape as half a gaussian)
    dt is the size of the time step in s (used to determine if there is a collision)
    dv is the bin size of the speed dependent functions in this routine
    vmax is the highest velocity to check for
    Nr and Nz are the number of cells in each dimension
    dr and dz are the physical size of those cells in m
    currently we assume all collisional particles are molecular hydrogen
    """
    Nrmid = (Nr-1)/2; Nzmid = (Nz-1)/2
    m = 2.0*1.66e-27; a = 8e-31  
    k = 1.38e-23 # boltzmann constant in SI units
    f0 = P/(k*T) # collision frequency without cross section, velocity, or distribution
    Dr = Nr*dr ; Dz = Nz*dz # physical sim size in r and z
    Nv = convert(UInt32,round(vmax/dv)) # number of velocity bins
    # next we map out the collision frequency as a function of velocity 
    sigmaTB(vr,vz)=Dr*Dz*(Dr*vz/(Dr*vz + Dz*vr)) # cross section for the top/bottom as a function of velocities vr/vz can be replaced by sin(phi)/cos(phi)
    sigmaLR(vr,vz)=(Dr^2)*(pi/4)*(Dz*vr/(Dr*vz + Dz*vr)) # cross section for the left/right as a function of velocities
    g(v,delta,m) = (m/(k*T))*(v-delta)*exp(-m*(v-delta)^2/(2*k*T)) # this sets the shape of the speed distribution (normalized to 1) 
    # then we find out if a particle should be added in this timestep
    exists = rand(Float64,Nv) # roll for particle
    phi = rand(Float64)*2*pi # Pick a particle angle
    vr = sin(phi) ; vz = cos(phi) # set relative r and z velocities
    vrpos = abs(vr) ; vzpos = abs(vz) # explicitly positive values for cross-sections or ratios
    r = ((rand(Float64)*(Nz-3) +2) - Nrmid)*dr ; z = ((rand(Float64)*(Nz-3) +2)-Nzmid)*dz # roll for starting position in grid units
    deltaTB = -pi^(1/4)*Vf[2,z]*(a/(2*m*sigmaV/vrpos))^(1/2) ; deltaLR = -pi^(1/4)*Vf[r,2]*(a/(2*m*sigmaV/vzpos))^(1/2) # speed reduction due to potential shape and angle of approach
    minI = max(1,convert(UInt32,round(min(deltaTB,deltaLR)/dv))) # don't bother checking for a collision under our minimum speed set by the speed reduction
    if minI>=Nv-1
        println("No collisional particles could make it over the escape point this time")
        return vc
    end
    coll = 0 ; speed = 0.0 ; ftot = 0.0 # for no particle
    for i=minI:Nv
        speed = dv*i
        fTB = f0*sigmaTB(vrpos,vzpos)*speed*g(speed,deltaTB,m)*dt*dv #probability of a collision in this stime step on top/bottom in this velocity bin
        fLR = f0*sigmaLR(vrpos,vzpos)*speed*g(speed,deltaLR,m)*dt*dv #probability of a collision in this stime step on left/right in this velocity bin
        ftot+=(fTB+fLR)   
        if exists[i]<fTB #if we get a particle to top/bottom
            if vr>0.0
                r = (2-Nrmid)*dr ; coll = 1 # 1 for bottom collisional particle
            else
                r = (Nr-1 - Nrmid)*dr ; coll = 2 # 2 for top collisional particle
            end
        elseif exists[i]<(fLR+fTB) # if we get a particle to left/right
            if vz>0.0
                z = (2-Nzmid)*dz ; coll=3 # 3 for left side
            else
                z = (Nz-1-Nzmid)*dz ; coll=4 # 4 for right side
            end
        end
        
    end
    if ftot>1.0
        println("Monte Carlo Error: Multiple collisions this cycle")
    end
    if coll>0
        for i=1:length(vc[:,1]) #for each particle
            if vc[i,6]>1.0 #if that particle is free it's assigned a large mass customarily
                vc[i,:]= [r,z,vr*speed,vz*speed,0.0,m,a]
                return vc
            end
        end
        println("No free collisional particle slots, vc = ",vc) # indicates we have so many collisions at once that we need a larger vc array
    end
    return vc
end
