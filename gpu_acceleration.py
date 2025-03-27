from IndexLookup import Index_Lookup
from refractiveindex import RefractiveIndexMaterial
import numpy as np
import torch
from matplotlib import pyplot as plt
import scipy.io
import torcwa

# We need to use our own materials for this, not the ones they provide
#import Materials

# Hardware
# If GPU support TF32 tensor core, the matmul operation is faster than FP32 but with less precision.
# If you need accurate operation, you have to disable the flag below.
torch.backends.cuda.matmul.allow_tf32 = False # Set to True if using RTX 3090 or above
sim_dtype = torch.complex64
geo_dtype = torch.float32
device = torch.device('cpu')

def honeycomb_lattice_gpu(Nx,Ny,Np,eps,diameter):
    """apply honeycomb lattice to torcwa obj"""
    thickp = diameter/Np

    centers = [
        (0, 0, 0), 
        (torch.sqrt(torch.tensor(3.)), 1, 0), 
        (torch.sqrt(torch.tensor(3.)), -1, 0), 
        (-torch.sqrt(torch.tensor(3.)), 1, 0), 
        (-torch.sqrt(torch.tensor(3.)), -1, 0)
    ]
    structure = honeycomb_structure_gpu(Nx, Ny, Np, centers)
    epgrid = np.array([])   
    for i in range(Np):
        epname = torch.ones((Nx, Ny), dtype=complex)
        epname[structure[:, :, i]] = eps
        epgrid = np.append(epgrid.flatten(),epname.flatten())

    return epgrid

def honeycomb_structure_gpu(Nx, Ny, Nz, centers=[(0, 0, 0)]):
    """returns 3D numpy boolean array of honeycomb structure"""
    x = torch.linspace(-torch.sqrt(torch.tensor(3.)), torch.sqrt(torch.tensor(3.)), Nx, dtype=geo_dtype,device=device)
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
    freqs = 1 / wv_sweep
    freqcmps = freqs*(1+1j/2/Qabs)

    # lattice constants
    L = [torch.sqrt(torch.tensor(3.)),1] # sqrt(3) by 1 um
    torcwa.rcwa_geo.Lx = L[0]
    torcwa.rcwa_geo.Ly = L[1]
    torcwa.rcwa_geo.nx = Nx
    torcwa.rcwa_geo.ny = Ny
    torcwa.rcwa_geo.grid()
    torcwa.rcwa_geo.edge_sharpness = 1000.
    torcwa.rcwa_geo.dtype = geo_dtype
    torcwa.rcwa_geo.device = device
    
    theta = torch.linspace(theta_start * DEG_TO_RAD, theta_end * DEG_TO_RAD, n_theta, dtype=geo_dtype,device=device)

    Rs = torch.zeros_like(freqs)
    Ts = torch.zeros_like(freqs)
    Rss = torch.zeros((len(theta), len(Rs)))
    Tss = torch.zeros((len(theta), len(Ts)))

    nG_order = [nG,nG]

    for z in range(len(theta)):
        print(f'theta {z}')
        for i in range(len(freqs)):
            epgrids = np.array([])

            # print(f'freq {freqs[i]}')
            ######### setting up RCWA
            obj = torcwa.rcwa(freq=i,order=nG_order,L=L,dtype=geo_dtype,device=device)
            wavelength = wv_sweep[i]
            obj.set_incident_angle(wavelength,0)
            for material, thickness, type in structure:
                if material == Ti:
                    if wavelength > 2300:
                        material = Ti_2
                if type == "slab":
                    obj.add_layer(thickness, Index_Lookup(material,wavelength))
                elif type == "honeycomb":
                    epgrid = honeycomb_lattice_gpu(Nx,Ny,Np,Index_Lookup(material,wavelength),thickness)
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

            Rs[i] = torch.real(R)
            Ts[i] = torch.real(T)
            Rss[z,i] = torch.real(R)
            Tss[z,i] = torch.real(T)

        if not theta_sweep:
            As = 1 - Rs - Ts
            return Rs, Ts, As 

    Rs = torch.mean(Rss,axis=0)
    Ts = torch.mean(Tss,axis=0)
    As = 1 - Rs - Ts

    return Rss, Tss, As