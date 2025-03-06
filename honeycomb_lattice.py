"""Transmission and reflection of a uniform sphere."""
import grcwa
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d

def honeycomb_lattice(nG,L1,L2,theta,Nx,Ny,Np,epbkg,diameter):
    nG = nG
    # lattice constants
    L1 = L1
    L2 = L2
    # frequency and angles
    freq = 1.
    theta = theta
    phi = 0.
    # to avoid singular matrix, alternatively, one can add fictitious small loss to vacuum
    Qabs = np.inf
    freqcmp = freq*(1+1j/2/Qabs)
    # the patterned layer has a gridding: Nx*Ny
    Nx = Nx
    Ny = Ny

    ep0 = 1.                                                            # dielectric for layer 1 (uniform)
    epN = 1.                                                            # dielectric for the final layer N (uniform)
    epp = 1.                                                            # dielectric for patterned layer (outside of sphere)

    thick0 = 1.                                                         # thickness for vacuum layer 1
    thickN = 1.                                                         # thickness for vacuum layer N

    """This section is for building the layers, these parameters can be changed as you see fit."""
    Np = Np                                                             # number of patterned layers (must be greater than 2, the top and bottom layers are equal to zero. So Np = #-2)
    epbkg = epbkg                                                          # dielectric for holes in the patterned layer (sphere)
    diameter = diameter                                                      # diameter of the sphere
    thickp = diameter/Np
    top_thickness = np.linspace(0,0.5*diameter,num=int(Np/2),endpoint=True)
    thickness_array = np.append(top_thickness,np.linspace(0.5*diameter,0,num=int(Np/2),endpoint=True))
    graph_array = np.linspace(0,diameter,num=Np,endpoint=True)

    delta_radius = 0
    radius_top = np.linspace(0,0.5*diameter,int((Np+1)/2))              # radius of the top half of the sphere in increments for layer creation
    radius_bot = np.linspace(0.5*diameter,0,int((Np+1)/2))              # radius of the bottom half of the sphere in increments for layer creation

    # eps for patterned layer
    x0 = np.linspace(0,math.sqrt(3),Nx)
    y0 = np.linspace(0,1.,Ny)
    x, y = np.meshgrid(x0,y0,indexing='ij')

    """This section will determine the limit to which the radius will extend to, and makes sure the middle section"""
    """won't be repeated if there are an odd number of layers"""
    # This will check if the number of layers is odd and make sure the middle layer isn't repeated
    radius_limit = 0                                                    # The radius_limit will initially be set to zero
    if int(Np%2) == 1:                                                  # Modular divide and if we have a remainder of 1, we have an odd number of layers
        radius_limit = radius_top[int((Np+1)/2)-1]                      # Our radius limit won't repeat the middle layer when it's odd
        index = len(radius_top)-2                                       # Index starts one before the final value for the bottom index
        reverseindex = 0
        while (radius_top[index]!=0):
            radius_bot[reverseindex] = radius_top[index]                # Reverses the radius_top values and puts them into the radius_bot
            index-=1
            reverseindex+=1
        radius_bot[reverseindex] = 0                                    # Sets the last value to zero because I can't get np.delete to work

    ######### setting up RCWA
    obj = grcwa.obj(nG,L1,L2,freqcmp,theta,phi,verbose=1)
    # input layer information
    Layer = 0                                                           # We start our layers at four (which is basically zero)
    top = True                                                          # Helps flip our condition so we start looking at the bottom layer
    obj.Add_LayerUniform(thick0,ep0)
    for i in range(Np-2):
        obj.Add_LayerGrid(thickp,Nx,Ny)
    obj.Add_LayerUniform(thickN,epN)
    obj.Init_Setup()

    topindex = 1                                                        # We don't want to consider a radius of zero, so we start at the second value
    botindex = 0                                                        # Since our bottom index needs to know the first value, we start at zero
    epgrid = np.array([])                                               # We start with an empty array to append the flatten integer arrays into

    start_time = time.time()
    plt.figure()
    ax = plt.axes(projection='3d')


    for layer in range(0,Np):                                         # This adds Np layers with half increasing to radius_limit and the rest decreasing to zero
        if top == True:                                                 # This section covers the top half of the sphere
            if radius_limit == 0:                                       # This covers when there's an even # of layers
                delta_radius = math.sqrt((diameter*thickness_array[layer])-(thickness_array[layer]*thickness_array[layer]))                     # The radius we're considering changes through every iteration
                if topindex >= int((Np-1)/2):                           # This marks the halfway point that will move us to the bottom half
                    top = False
            else:                                                       # This covers when there's an odd # of layers
                delta_radius = math.sqrt((diameter*thickness_array[layer])-(thickness_array[layer]*thickness_array[layer]))                     # The radius we're considering changes through every iteration
                if topindex >= int((Np)/2):                             # This marks the halfway point that will move us to the bottom half
                    top = False
            topindex += 1                                               # This moves us to analyze the next top layer

        else:                                                           # This section covers the bottom half of the sphere (bottom = True)
            if radius_limit == 0:                                       # This covers when there's an even # of layers
                delta_radius = math.sqrt((diameter*thickness_array[layer])-(thickness_array[layer]*thickness_array[layer]))                     # The radius we're considering changes through every iteration
                if botindex >= int((Np-1)/2):                           # Once this condition is true, we have finished the calculation
                    print("FINISHED EVEN")
                    break
            else:
                delta_radius = math.sqrt((diameter*thickness_array[layer])-(thickness_array[layer]*thickness_array[layer]))                     # The radius we're considering changes through every iteration
                if botindex >= int((Np-2)/2):                           # Once this condition is true, we have finished the calculation
                    print("FINISHED ODD")
                    break
            botindex += 1                                               # This moves us to analyze the next bottom layer
        'The next 4 lines create the layers that we are analyzing'
        epname = np.ones((Nx, Ny), dtype=float)*epp
        honeycomb = np.logical_or(((x-.866)**2 + (y-.5)**2 < delta_radius**2), (x**2 + y**2 < delta_radius**2))
        honeycomb = np.logical_or(honeycomb, ((x-1.73205)**2) + y**2 < delta_radius**2)
        honeycomb = np.logical_or(honeycomb, (x**2 + (y-1)**2) < delta_radius**2)
        honeycomb = np.logical_or(honeycomb, ((x-1.73205)**2 + (y-1)**2) < delta_radius**2)
        epname[honeycomb] = epbkg
        epgrid = np.append(epgrid.flatten(),epname.flatten())

        ax.contour3D(x,y,honeycomb,offset=graph_array[layer],cmap='jet')
        #plt.figure()
        #plt.contourf(x,y,honeycomb,cmap='jet')
        #plt.axis('equal')
        #plt.tight_layout
        #plt.show()
        print("delta_radius is: " + str(delta_radius))
    ax.set_aspect('equal')
    ax.set_title('Graphical Representation')
    ax.set_xlabel('x side')
    ax.set_ylabel('y side')
    plt.show()

    # We combine all of the epsilon values
    obj.GridLayer_geteps(epgrid)

    # We create the planewave excitation
    planewave={'p_amp':0,'s_amp':1,'p_phase':0,'s_phase':0}
    obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)

    # solve for R and T
    R,T= obj.RT_Solve(normalize=1)
    print("This program took %s seconds to run" % round((time.time() - start_time),4))

    return R, T
