from IndexLookup import Index_Lookup
from refractiveindex import RefractiveIndexMaterial
import grcwa_torch
import numpy as np
import matplotlib.pyplot as plt
import torch

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

def honeycomb_lattice(obj,Nx,Ny,Np,eps,diameter,device):
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

def loss_fun(thicknesses, wv_sweep, device, nG=40, 
          theta_start=0, theta_end=80, n_theta=10, theta_sweep=True, 
          Nx=100, Ny=100, Np=10, structure=dev_structure, diameter=1, plot=False, title=""):
    """
    wv in nm
    nG = truncation order
    theta in degrees
    if theta_sweep == False: use theta_start as the angle
    """
    if plot and theta_sweep:
        Rs = torch.empty(n_theta, len(wv_sweep))

    loss = torch.tensor(0., requires_grad=True)
             
    freqs = (1 / wv_sweep).reshape(-1, 1)
    wv_sweep = wv_sweep.cpu()
    freqcmps = freqs*(1+1j/2/Qabs)

    # lattice constants
    if diameter:
        L1 = [diameter*np.sqrt(3),0] # 1 um
        L2 = [0,1*diameter]
    else:
        L1 = [1, 0] # 1 um
        L2 = [0, 1]
    theta = torch.linspace(theta_start, theta_end * DEG_TO_RAD, n_theta, device=device)
    phi = torch.tensor(0., device=device)

    # TODO: optimize looking up epsilon
    for z in range(len(theta)):
        ######### setting up RCWA
        material_eps = torch.ones(len(freqs), len(structure), device=device, dtype=complex)
        
        for i in range(len(structure)):
            material, _, _ = structure[i]
            if material == Ti:
                if wavelength > 2.3:
                    material = Ti_2

            current_eps = torch.ones(len(freqs), dtype=complex)
            for j in range(len(wv_sweep)):
                wavelength = wv_sweep[j]
                current_eps[j] = Index_Lookup(material, wavelength)
                material_eps[:, i] = current_eps

        obj = grcwa_torch.obj(nG,L1,L2,freqcmps,theta[z],phi,verbose=0, eps_batch_=True)
        epgrids = torch.empty(len(freqs), 0, device=device, dtype=complex)
        for j in range(len(structure)):
            material, thickness, type = structure[j]
            if type == "slab":
                obj.Add_LayerUniform(thicknesses[j], material_eps[:, j:j+1])
            elif type == "honeycomb":
                # assert thickness == diameter
                epgrid = honeycomb_lattice(obj,Nx,Ny,Np,material_eps[:, j:j+1],diameter,device)
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

        mask = (wv_sweep >= 8) & (wv_sweep <= 13)

        loss = loss + (R[mask].real**2).sum()
        loss = loss + ((1 - R[~mask].real)**2).sum()

        if not theta_sweep:
            if plot:
                plt.clf()
                plt.plot(wv_sweep, 1 - R)
                plt.xlabel('Wavelength (μm)')
                plt.ylabel('Absorbance')
                plt.ylim(0, 1)
                plt.title(title)
                plt.show()
            return loss
        else:
            if plot:
                Rs[z] = R.real.detach()

    if plot:
        R = torch.mean(Rs,dim=0)
        plt.clf()
        plt.plot(wv_sweep, 1 - R)
        plt.xlabel('Wavelength (μm)')
        plt.ylabel('Absorbance')
        plt.ylim(0, 1)
        plt.title(title)
        plt.show()

    return loss


def optimize_torch(wv_sweep, device, nG=40, 
          theta_start=0, theta_end=80, n_theta=10, theta_sweep=True, 
          Nx=100, Ny=100, Np=10, structure=dev_structure, diameter=1, weight=1e-3, epoch=100, print_every=10, plot=False):
    """
    wv in nm
    nG = truncation order
    theta in degrees
    if theta_sweep == False: use theta_start as the angle
    """
    thicknesses = torch.tensor([t for _, t, _ in structure], dtype=torch.float32, requires_grad=True)
    diameter = torch.tensor(diameter, requires_grad=diameter != False)

    optimizing_params = [thicknesses]
    if diameter:
        optimizing_params.append(diameter)
    optimizer = torch.optim.SGD(optimizing_params, lr=weight)

    print("Initial loss:", loss_fun(thicknesses, wv_sweep, device, nG, 
          theta_start, theta_end, n_theta, theta_sweep, 
          Nx, Ny, Np, structure, diameter, plot=plot, title="Absorption before optimization"))

    for i in range(epoch):
        optimizer.zero_grad()
        loss = loss_fun(thicknesses, wv_sweep, device, nG, 
                        theta_start, theta_end, n_theta, theta_sweep, 
                        Nx, Ny, Np, structure, diameter)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            thicknesses.clamp_(min=0)
        
        if i % print_every == 0:
            print(f"Epoch {i}: Loss = {loss.item()}")
            print("Thicknesses:", thicknesses.detach().numpy())
            if diameter:
                print("Diameter:", diameter.detach().numpy())

    print("Final loss:", loss_fun(thicknesses, wv_sweep, device, nG, 
                                  theta_start, theta_end, n_theta, theta_sweep, 
                                  Nx, Ny, Np, structure, diameter, plot=plot, title="Absorption after optimization"))
    
    return thicknesses.detach().numpy(), diameter.detach().numpy() if diameter else None