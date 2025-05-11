from IndexLookup import Index_Lookup
from refractiveindex import RefractiveIndexMaterial
import numpy as np
import grcwa_torch
import torch
import math
from functools import reduce

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

def honeycomb_lattice(obj,Nx,Ny,Np,eps,diameter,device,ratio):
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
    structure = honeycomb_structure(Nx//ratio, Ny//ratio, Np, centers)
    structure = np.tile(structure, (ratio, ratio, 1))
    import matplotlib.pyplot as plt
    for z in range(structure.shape[-1]):
        plt.imshow(structure[:,:,z].transpose(), cmap='gray')
        plt.title(f"Ratio: {ratio}")
        plt.show()
    
    epgrid = torch.empty(eps.shape[0], 0, device=device, dtype=complex)
    for i in range(Np):
        epname = torch.ones((eps.shape[0], Nx, Ny), dtype=complex, device=device)
        epname[:, structure[:, :, i]] = eps
        epgrid = torch.cat((epgrid, epname.flatten(start_dim=1)), dim=1)

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

def beads_torch(wv_sweep, device, nG=40, 
          theta_start=0, theta_end=80, n_theta=10, theta_sweep=True, 
          Nx=100, Ny=100, Np=10, structure=dev_structure, diameter=1):
    """
    wv in nm
    nG = truncation order
    theta in degrees
    if theta_sweep == False: use theta_start as the angle
    """               
    freqs = (1 / wv_sweep).reshape(-1, 1)
    wv_sweep = wv_sweep.cpu()
    freqcmps = freqs*(1+1j/2/Qabs)

    diameter_list = [thickness for _, thickness, type in structure if type == 'honeycomb']

    # lattice constants
    if len(diameter_list) == 0:
        L1 = [1,0] # 1 um
        L2 = [0,1]
    else:
        lcm = reduce(math.lcm, diameter_list)
        ratios = [lcm // thickness for thickness in diameter_list]
        # Upscale Nx and Ny
        max_ratio = max(ratios)
        Nx = int(Nx*np.sqrt(3)) * max_ratio
        Ny = Ny*max_ratio
        print(f"Nx: {Nx}, Ny: {Ny}")
        L1 = [lcm*np.sqrt(3),0] # 1 um
        L2 = [0,1*lcm]
    
    theta = torch.linspace(theta_start * DEG_TO_RAD, theta_end * DEG_TO_RAD, n_theta, device=device)
    if theta_sweep == False:
        theta = theta[0]
    else:
        theta = theta.reshape(1, -1)
    phi = torch.tensor(0., device=device)

    ######### setting up RCWA
    material_eps = torch.ones(len(freqs), len(structure), device=device, dtype=complex)
    
    for i in range(len(structure)):
        material, _, _ = structure[i]
        current_eps = torch.ones(len(freqs), dtype=complex)
        for j in range(len(wv_sweep)):
            wavelength = wv_sweep[j]
            current_eps[j] = Index_Lookup(material, wavelength)
            material_eps[:, i] = current_eps

    obj = grcwa_torch.obj(nG,L1,L2,freqcmps,theta,phi,verbose=0, eps_batch_=True)
    epgrids = torch.empty(len(freqs), 0, device=device, dtype=complex)
    for j in range(len(structure)):
        material, thickness, type = structure[j]
        if type == "slab":
            obj.Add_LayerUniform(thickness, material_eps[:, j:j+1])
        elif type == "honeycomb":
            ratio = ratios.pop(0)
            epgrid = honeycomb_lattice(obj,Nx,Ny,Np,material_eps[:, j:j+1],thickness,device,ratio)
            epgrids = torch.cat((epgrids, epgrid), dim=1)
        else:
            raise NotImplementedError

    obj.Init_Setup(device)

    if epgrids.shape[-1] != 0:
        obj.GridLayer_geteps(epgrids, device)

    # planewave excitation
    p_amp = torch.tensor(1, device=device)
    s_amp = torch.tensor(0, device=device)
    p_phase = torch.tensor(0, device=device)
    s_phase = torch.tensor(0, device=device)
    planewave={'p_amp':p_amp,'s_amp':s_amp,'p_phase':p_phase,'s_phase':s_phase}
    obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],device,order = 0)

    # compute reflection and transmission
    R,T= obj.RT_Solve(device, normalize=1)

    Rs = torch.real(R)
    Ts = torch.real(T)
    As = 1 - Rs - Ts

    return Rs, Ts, As