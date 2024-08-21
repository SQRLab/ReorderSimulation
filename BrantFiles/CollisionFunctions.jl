function EBarrier2(wr,wz,m)
    # wr and wz are the angular frequencies of the trap along the rf and dc directions respectively
    # m is the mass of the trapped ion in atomic mass units
    # define constants
    mp=1.67e-27
    mTrap=m*mp #Calcium ion
    epsilon=8.854e-12
    Q=1.6e-19 #Coulombs

    # Solve for the equilibria distance between two ions axially
    # These solutions are derived by expressing the potential energy from the trap and electrostatic interactions between ions and solving explicitly for the separation at the minimums to that potential. This becomes impossible to do in a pure analytical fashion for longer ion chains.
    a=(Q^2/(2*mTrap*wz^2*pi*epsilon))^(1/3)
    U1=mTrap*wz^2*a^2/4+Q^2/(4*pi*epsilon*a)
    # Same but radially
    b=(Q^2/(2*mTrap*wr^2*pi*epsilon))^(1/3)
    U2=mTrap*wr^2*b^2/4+Q^2/(4*pi*epsilon*b)
    # Difference in energy gives the required input from the collision
    U=U2-U1
    #println("a, ",a,", b, ",b,", U, ",U )
    if(U<0)
     println("Barrier less than zero!, a ",a,", b ",b,", U ",U)
    end
    return U
end

function MBDist3D(T,n,low,high,m,verbose)
    #Returns a maxwell boltzmann distribution for the given temp and 3 degrees of movement and n steps between low and high speed values 
    # T is in kelvin and low and high are m/s m is in amu
    mp=1.67e-27
    m2 = m*mp
    k=1.38e-23
    step = (high-low)/n
    p = zeros(Float64,n)
    v = zeros(Float64,n)
    if verbose
        println("MB step ",step)
    end
    for i=1:n
        v[i] = low+step*i
        p[i] = step*4*pi*v[i]^2*(m2/(2*pi*k*T))^(3/2).*exp(-m2*v[i]^2/(2*k*T))
    end
    if verbose
        println("MB Sum",sum(p))
    end
    return v,p
end

function PlotEnergyBarrier(wr,wz,U)
    #wr is a vector of floats eg. [1.0,2.0,3.0]
    #wz is a vector of floats eg. [1.0,2.0,3.0]
    #U is an array of doubles eg. Vector[[1.0,2.0,3.0],[2.0,2.0,3.0],[3.0,2.0,3.0]]
    trace = surface(x=wr,y=wz,z=U)
    layout = Layout(title="Energy Barrier (Joules)", autosize=false, width=800,
                    height=800, margin=attr(l=65, r=50, b=65, t=90),
                    scene = attr(
                    xaxis_title="Radial Angular Frequency (Radians)",
                    yaxis_title="Axial Angular Frequency (Radians)",
                    zaxis_title="Energy Barrier (Joules)"))
    plot(trace, layout)
end

function CollETransfer(mtrap,mcoll,n)
    # m1 is the mass of the trapped ion in amu
    # m2 is the mass of the colliding ion in amu
    # n is the length of the return vector which is made up of equal slices 
    # of Beta between -Pi/2 and Pi/2 this should be 
    # weighted such that large angles are more common? 
    # I haven't implemented this part as the 'radius' of the ion is ill defined and I need to choose an 
    # approximation for closest approach, this probably depends on species as it would be proportional to charge or dipole moment
    # The scaling would then be d*Sin(Beta) and d would depend on initial speed with faster particles less likely
    # to interact at larger distances
    mp=1.67e-27
    m1 = mtrap*mp
    m2 = mcoll*mp
    Bi = -pi/2.0
    Bf = pi/2.0
    Beta = zeros(Float64,n)
    eFrac = zeros(Float64,n)
    for i=1:n
        Beta[i] = Bi+i*pi/n-0.5*pi/n
        eFrac[i] = cos(Beta[i])^2*(4*m1*m2)/(m1+m2)^2
    end
    return Beta,eFrac
end

#fractional likelihood of collision at given angle
function CollAngleProb(a,E,beta)
    # a is the polarizability of the colliding molecule
    # E is a vector of the kinetic energy of the colliding molecule
    # beta is the vector of angles of interest
    # the output probability is an array with velocity/energy on the first axis and collision angle on the second such that pout[i] gives the probability of each collision angle for the energy E[i]

    # Define constants
    Q = 1.6e-19
    eps0 = 8.854e-12
    
    # This calculates the probability of a collision at the given angle
    C4 = (a*Q^2)/(8*pi*eps0) # dipole moment in J*m^4
    pout = []
    sigma = []
    for i=1:length(E)
        bc = (4*C4/E[i])^(1/4) # critical impact parameter or distance of closest approach in a hard sphere model
        p =  bc.*abs.(sin.(beta)) # the probability scales with the size of the circle of same collision angle which is proportional to radius
        p = p./sum(p) # we normalize the probabilities so that they sum to one  
        #println("CollAngleProb Debug ",i)
        #println("CollAngleProb Debug ",pout)
        push!(pout,p)
        push!(sigma,pi*bc^2)
    end
    return sigma,pout
end

#Analytical
function AnalyticReorder2(v,E,P,T,U,pVel,eFrac,pCollAng,sigmaColl)
    # E is a vector of energies of colliding atoms
    # P is the gas pressure of those atoms
    # U is the potential energy barrier of the trap
    # pVel is the probability density of those energies
    # eFrac is the fractional energy transfer across different angles
    # pCollAng is the probability of having those same angles for each energy such that pCollAng[1] is the probability of colliding at each angle for the lowest energy of interest
    # tReorder is the expected time for reorder to occur
    
    k = 1.38*10^-23 #boltzmann constant
    
    pReorder = 0
    fReorder = 0
    n = P/(k*T) #volumetric number density of the idealized gas
    fColl = n.*sigmaColl.*v #collision frequency as a function of velocity
    for i=1:length(E)
        for j=1:length(eFrac)
            
            if E[i]*eFrac[j]>U
                fReorder+=pCollAng[i][j]*pVel[i]*fColl[j]
                pReorder+=pCollAng[i][j]*pVel[i]
            end
        end
    end
    
    #Debug
    #println("AnalyticReorder2 reorder probability (should be <1.0)",pReorder)
    #println("AnalyticReorder2 fColl = ",fColl)
    println("Expected number of collisions until re-order = ",mean(fColl)/fReorder)
    
    return 1.0/fReorder
end

function randomIndexFromCDF(cdf)
    #this takes a cumulative density function and outputs a random index from that function
    n = length(cdf)
    randi = rand()
    #debug
    #println("randomIndexFromCDF randi ",randi)
    for i=1:n
       if randi<=cdf[i]
            return i
       end
    end    
    return n
end

#Monte Carlo
function MonteCarloReorder2(v,E,P,T,U,pVel,eFrac,pCollAng,sigmaColl,nRuns)
    # E is a vector of energies of colliding atoms
    # P is the gas pressure of those atoms
    # U is the potential energy barrier of the trap
    # pVel is the probability density of those energies
    # eFrac is the fractional energy transfer across different angles
    # pCollAng is the probability of having those same angles for each energy such that pCollAng[1] is the probability of colliding at each angle for the lowest energy of interest
    # n is the number of reorders to run
    # tReorder is a vector of measured times to reorder
    
    k = 1.38*10^-23 #boltzmann constant

    n = P/(k*T) #volumetric number density of the idealized gas
    fColls = n.*sigmaColl.*v #collision frequency as a function of velocity
    fColl = mean(fColls)
    #Debug
    #println("fColl = ",fColl)
    
    # here we create cumulative density functions for energy and angle 
    pVelCDF = []
    pCollAngCDF = []
    for i=1:length(pVel)
        push!(pVelCDF,sum(pVel[begin:i]))
    end
    
    for i=1:length(pVel)
        push!(pCollAngCDF,[])
        for j=1:length(pCollAng[begin])
            push!(pCollAngCDF[i],sum(pCollAng[i][begin:j]))
        end
    end

    #Debug
    #println("pCollAngCDF = ",pCollAngCDF)
    
    tReorders = zeros(Int64(nRuns))
    for i=1:nRuns #iterate the collisional sim
        ordered = true
        its = 0
        while ordered
            its += 1
            # here we choose a random energy index j
            j = randomIndexFromCDF(pVelCDF)
            #here we choose a random angle index k
            k = randomIndexFromCDF(pCollAngCDF[j])
            #Debug
            #println("pCollAngCDF[j] = ",pCollAngCDF[j])
            #println("j = ",j," k = ",k)
            #println("eFrac[k] = ",eFrac[k],"mean = ",mean(eFrac))
            #println("E[j] = ",E[j]," mean = ",mean(E))
            if E[j]*eFrac[k]>U
                ordered = false
                tReorders[i] = its/fColl #time to reorder is collisions until reorder * the time between collisions
                #Debug
                #println("MonteCarloReorder2 iterations = ",its," fColl = ",fColl," tReorders[i] = ",tReorders[i])
                #println("Iterations = ",its)
            end        
        end

    end
    
    return tReorders
end