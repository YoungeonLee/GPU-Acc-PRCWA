from IndexLookup import Index_Lookup
from refractiveindex import RefractiveIndexMaterial
import numpy as np
import torch
from matplotlib import pyplot as plt
import scipy.io
import time
import torcwa

# We need to use our own materials for this, not the ones they provide
#import Materials

# Hardware
# If GPU support TF32 tensor core, the matmul operation is faster than FP32 but with less precision.
# If you need accurate operation, you have to disable the flag below.
# torch.backends.cuda.matmul.allow_tf32 = False # Set to True if using RTX 3090 or above
sim_dtype = torch.complex128
geo_dtype = torch.float32
device = torch.device('cpu')

def honeycomb_structure_gpu(Nx, Ny, Nz, centers=[(0, 0, 0)]):
    """returns 3D numpy boolean array of honeycomb structure"""
    x = torch.linspace(-torch.sqrt(3), torch.sqrt(3), Nx, dtype=geo_dtype,device=device)
    y = torch.linspace(-1, 1, Ny, dtype=geo_dtype,device=device)
    z = torch.linspace(-1, 1, Nz, dtype=geo_dtype,device=device)
    radius = 1

    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # Initialize mask
    sphere_mask = torch.zeros((Nx, Ny, Nz), dtype=bool)

    # Create spheres at given centers
    for cx, cy, cz in centers:
        sphere_mask |= ((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) <= radius ** 2

    return sphere_mask

def honeycomb_lattice_gpu(obj,Nx,Ny,Np,eps,diameter):
    """apply honeycomb lattice to torcwa obj"""
    thickp = diameter/Np                                                # thickness of patterned layer

    for _ in range(Np):
        obj.Add_LayerGrid(thickp,Nx,Ny)

    centers = [
        (0, 0, 0), 
        (torch.sqrt(3), 1, 0), 
        (torch.sqrt(3), -1, 0), 
        (-torch.sqrt(3), 1, 0), 
        (-torch.sqrt(3), -1, 0)
    ]
    structure = honeycomb_structure_gpu(Nx, Ny, Np, centers)
    
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
    ('air',0.0,'slab'), 
    (SiO2,8000,'honeycomb'), 
    (SodaLime,1000000,'slab'), 
    ('air',0.0,'slab') 
]

DEG_TO_RAD = np.pi / 180
Qabs = np.inf

def gpu_acceleration(wv_sweep, nG=40, 
          theta_start=0, theta_end=80, n_theta=10, theta_sweep=False,
          Nx=100, Ny=100, Np=10, structure=dev_structure):
    """
    wv in nm
    nG = truncation order
    theta in degrees
    if theta_sweep == False: use theta_start as the angle
    """               
    freqs = 1 / (wv_sweep)
    freqcmps = freqs*(1+1j/2/Qabs)

    # lattice constants
    # L = [torch.sqrt(torch.tensor(3.0)),1] # sqrt(3) by 1 um
    L = [1, 1]
    torcwa.rcwa_geo.Lx = L[0]
    torcwa.rcwa_geo.Ly = L[1]
    torcwa.rcwa_geo.nx = Nx
    torcwa.rcwa_geo.ny = Ny
    # torcwa.rcwa_geo.grid()
    # torcwa.rcwa_geo.edge_sharpness = 1000.
    # torcwa.rcwa_geo.dtype = geo_dtype
    torcwa.rcwa_geo.device = device
    
    theta = torch.linspace(theta_start * DEG_TO_RAD, theta_end * DEG_TO_RAD, n_theta, dtype=torch.float32, device=device)
    phi = 0.

    Rs = torch.zeros_like(freqs)
    Ts = torch.zeros_like(freqs)
    Rss = torch.zeros((len(theta), len(Rs)))
    Tss = torch.zeros((len(theta), len(Ts)))

    nG_order = [nG,nG]

    for z in range(len(theta)):
        print(f'theta {z}')
        for i in range(len(freqs)):
            print(f'freq {i}')
            # epgrids = np.array([])

            # print(f'freq {freqs[i]}')
            ######### setting up RCWA
            #obj = grcwa.obj(nG,L1,L2,freqcmps[i],theta[z],phi,verbose=0)
            sim = torcwa.rcwa(freq=freqs[i],order=nG_order,L=L,dtype=sim_dtype,device=device)
            sim.set_incident_angle(inc_ang=theta[z],azi_ang=0)
            wavelength = wv_sweep[i]
            for material, thickness, type in structure:
                if material == Ti:
                    if wavelength > 2300:
                        material = Ti_2
                if type == "slab":
                    # print(f'slab added: {material}')
                    # print(torch.tensor(Index_Lookup(material,wavelength)))
                    # print(torch.tensor(Index_Lookup(material,wavelength)))
                    sim.add_layer(thickness, torch.tensor(Index_Lookup(material,wavelength)))
                elif type == "honeycomb":
                    raise NotImplementedError
                    epgrid = honeycomb_lattice_gpu(obj,Nx,Ny,Np,Index_Lookup(material,wavelength),thickness)
                    epgrids = np.append(epgrids.flatten(),epgrid.flatten())
                else:
                    raise NotImplementedError
            # sim.add_output_layer(eps=1.)

            sim.solve_global_smatrix()

            R = sim.S_parameters(orders=[0,0],direction='forward',port='reflection')
            T = sim.S_parameters(orders=[0,0],direction='forward',port='transmission')

            Rs[i] = np.real(R)
            Ts[i] = np.real(T)
            Rss[z,i] = np.real(R)
            Tss[z,i] = np.real(T)

        if not theta_sweep:
            As = 1 - Rs - Ts
            return Rs, Ts, As 

    Rs = np.mean(Rss,axis=0)
    Ts = np.mean(Tss,axis=0)
    As = 1 - Rs - Ts

    return Rss, Tss, As