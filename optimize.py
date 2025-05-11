from IndexLookup import Index_Lookup
from refractiveindex import RefractiveIndexMaterial
import grcwa
import numpy as npf
import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
# import nlopt

def honeycomb_structure(Nx, Ny, Nz, centers=[(0, 0, 0)]):
    """returns 3D numpy boolean array of honeycomb structure"""
    x = np.linspace(-np.sqrt(3), np.sqrt(3), Nx)
    y = np.linspace(-1, 1, Ny)
    z = np.linspace(-1, 1, Nz)
    radius = 1

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Initialize mask
    sphere_mask = np.zeros((Nx, Ny, Nz), dtype=bool)

    # Create spheres at given centers
    for cx, cy, cz in centers:
        sphere_mask |= ((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) <= radius ** 2

    return sphere_mask

def honeycomb_lattice(obj,Nx,Ny,Np,eps,diameter):
    """apply honeycomb lattice to grcwa obj"""
    thickp = diameter/Np                                                # thickness of patterned layer

    for _ in range(Np):
        obj.Add_LayerGrid(thickp,Nx,Ny)

    centers = [
        (0, 0, 0), 
        (np.sqrt(3), 1, 0), 
        (np.sqrt(3), -1, 0), 
        (-np.sqrt(3), 1, 0), 
        (-np.sqrt(3), -1, 0)
    ]
    structure = honeycomb_structure(Nx, Ny, Np, centers)
    
    epgrid = np.array([])   
    for i in range(Np):
        epname = np.ones((Nx, Ny), dtype=complex)
        epname[structure[:, :, i]] = eps
        epgrid = np.append(epgrid.flatten(),epname.flatten())

    return epgrid

nm_to_um = 1e-3

SiO2 = RefractiveIndexMaterial(shelf='main', book='SiO2', page='Franta-25C')
HfO2 = RefractiveIndexMaterial(shelf='main', book='HfO2', page='Franta')
Ag = RefractiveIndexMaterial(shelf='main', book='Ag', page='Ciesielski')
Si = RefractiveIndexMaterial(shelf='main', book='Si', page='Franta-25C')
Ti = RefractiveIndexMaterial(shelf='main', book='Ti', page='Werner')
Ti_2 = RefractiveIndexMaterial(shelf='main', book='Ti', page='Ordal')
SodaLime = RefractiveIndexMaterial(shelf='glass', book='soda-lime', page='Rubin-IR')

dev_structure = [
    ('air',0.0*nm_to_um,'slab'),
    (SiO2,230,'honeycomb'),
    (HfO2,485,'slab'),
    (SiO2,688,'honeycomb'),
    (HfO2,13,'slab'),
    (SiO2,73,'honeycomb'),
    (HfO2,34,'slab'),
    (SiO2,54,'honeycomb'),
    (Ag,200,'slab'),
    (Ti,20,'slab'),
    (Si,750,'slab'),
    ('air',0.0*nm_to_um,'slab')
]

DEG_TO_RAD = np.pi / 180
Qabs = np.inf

def loss_fun(thicknesses, wv_sweep, nG=40, 
          theta_start=0, theta_end=80, n_theta=10, theta_sweep=True, 
          Nx=100, Ny=100, Np=10, structure=dev_structure, diameter=1, plot=False):
    """
    wv in nm
    nG = truncation order
    theta in degrees
    if theta_sweep == False: use theta_start as the angle
    """

    loss = np.array(0.)
             
    freqs = 1 / wv_sweep
    freqcmps = freqs*(1+1j/2/Qabs)

    # lattice constants
    L1 = [diameter*np.sqrt(3),0] # 1 um
    L2 = [0,1*diameter]
    
    theta = np.linspace(theta_start, theta_end * DEG_TO_RAD, num=n_theta, endpoint=True)
    phi = 0.

    Rs = []
    Ts = []
    Rss = []
    Tss = []

    for z in range(len(theta)):
        # print(f'theta {z}')
        for i in range(len(freqs)):
            epgrids = np.array([])

            # print(f'freq {freqs[i]}')
            ######### setting up RCWA
            obj = grcwa.obj(nG,L1,L2,freqcmps[i],theta[z],phi,verbose=0)
            wavelength = wv_sweep[i]
            for j in range(len(structure)):
                material, thickness, type = structure[j]
                if material == Ti:
                    print('Ti')
                    if wavelength > 2.3:
                        material = Ti_2
                if type == "slab":
                    obj.Add_LayerUniform(thicknesses[j], Index_Lookup(material,wavelength))
                elif type == "honeycomb":
                    assert thickness == diameter
                    epgrid = honeycomb_lattice(obj,Nx,Ny,Np,Index_Lookup(material,wavelength),thickness)
                    epgrids = np.append(epgrids.flatten(),epgrid.flatten())
                else:
                    raise NotImplementedError

            obj.Init_Setup()

            if len(epgrids) != 0:
                obj.GridLayer_geteps(epgrids)

            # planewave excitation
            planewave={'p_amp':1,'s_amp':0,'p_phase':0,'s_phase':0}
            obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)

            # compute reflection and transmission
            R,T= obj.RT_Solve(normalize=1)
            Rs.append(np.real(R))
            if wavelength > 8 and wavelength < 13:
                loss += np.real(R) ** 2
            else:
                loss += (1 - np.real(R)) ** 2

        # Rs = np.stack(Rs)
        # Ts = np.stack(Ts)

        if not theta_sweep:
            # As = 1 - Rs - Ts
            if plot:
                plt.clf()
                plt.plot(wv_sweep, 1 - np.array(Rs))
                plt.title('Absorbance')
                plt.xlabel('Wavelength (μm)')
                plt.ylim(0, 1)
                plt.show()
            return loss
        
        # needs to be checked
        Rss.append(Rs)
        Tss.append(Ts)

    Rs = np.mean(Rss,axis=0)
    Ts = np.mean(Tss,axis=0)
    # As = 1 - Rs - Ts

    return loss

def optimize(wv_sweep, nG=40, 
          theta_start=0, theta_end=80, n_theta=10, theta_sweep=True, 
          Nx=100, Ny=100, Np=10, structure=dev_structure, diameter=1, weight=1, epoch=100, print_every=10, plot=False):
    """
    wv in nm
    nG = truncation order
    theta in degrees
    if theta_sweep == False: use theta_start as the angle
    """
    fun = lambda x: loss_fun(x, wv_sweep, nG, 
          theta_start, theta_end, n_theta, theta_sweep, 
          Nx, Ny, Np, structure, diameter)
    grad_fun = grad(fun)

    thicknesses = np.array([thickness for _, thickness, _ in structure])
    # thicknesses = np.ones_like(thicknesses) * .01

    # print("Initial loss:", loss_fun(thicknesses, wv_sweep, nG, 
    #       theta_start, theta_end, n_theta, theta_sweep, 
    #       Nx, Ny, Np, structure, diameter, plot=plot))
    # start_time = time.time()
    for i in range(epoch):
        grad_res = grad_fun(thicknesses)
        thicknesses -= grad_res * weight
        thicknesses = np.maximum(thicknesses, 0)
        
        if i % print_every == 0:
            # current_time = time.time() - start_time
            print(f"Epoch {i}: ", loss_fun(thicknesses, wv_sweep, nG, 
                                theta_start, theta_end, n_theta, theta_sweep, 
                                Nx, Ny, Np, structure, diameter, plot=plot))
            print(thicknesses)

    # current_time = time.time() - start_time
    # print("Trained loss:", loss_fun(thicknesses, wv_sweep, nG, 
    #                             theta_start, theta_end, n_theta, theta_sweep, 
    #                             Nx, Ny, Np, structure, diameter,plot=plot))
    return thicknesses