from IndexLookup import Index_Lookup
from refractiveindex import RefractiveIndexMaterial
import numpy as np
import matplotlib.pyplot as plt 
import grcwa
import time

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
    strucutre = honeycomb_structure(Nx, Ny, Np, centers)
    
    epgrid = np.array([])   
    for i in range(Np):
        epname = np.ones((Nx, Ny), dtype=complex)
        epname[strucutre[:, :, i]] = eps
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
    (SiO2,8000,'honeycomb'),
    (SodaLime,1000000,'slab'),
    ('air',0.0*nm_to_um,'slab')
]
start_wv = 5000  # 5 um
end_wv = 20000   # 20 um
wv_sweep = np.linspace(start_wv, end_wv, num=100, endpoint=True)

# grcwa

DEG_TO_RAD = np.pi / 180

# Truncation order (actual number might be smaller)
nG = 40
# lattice constants
L1 = [np.sqrt(3),0] # 1 um
L2 = [0,1]
# frequency and angles
# theta = 0 * DEG_TO_RAD
theta = np.linspace(0, 80 * DEG_TO_RAD, num=10, endpoint=True)
phi = 0.
# wls = np.linspace(5, 20, num=300, endpoint=True) # sweep from 5 um to 20 um
freqs = 1 / wv_sweep
Qabs = np.inf
freqcmps = freqs*(1+1j/2/Qabs)
Nx = 100
Ny = 100
Np = 10     # number of discrete layers for sphere

Rs = np.zeros_like(freqs)
Ts = np.zeros_like(freqs)
Rss = np.zeros((len(theta), len(Rs)))
Tss = np.zeros((len(theta), len(Ts)))
start_time = time.time()

epgrid = None
for z in range(len(theta)):
    print(f'theta {z}')
    for i in range(len(freqs)):
        print(f'freq {i}')
        ######### setting up RCWA
        obj = grcwa.obj(nG,L1,L2,freqcmps[i],theta[z],phi,verbose=0)
        wavelength = wv_sweep[i]
        for material, thickness, type in dev_structure:
            if type == "slab":
                obj.Add_LayerUniform(thickness, Index_Lookup(material,wavelength))
            elif type == "honeycomb":
                epgrid = honeycomb_lattice(obj,Nx,Ny,Np,Index_Lookup(material,wavelength),thickness)
            else:
                raise NotImplementedError

        obj.Init_Setup()

        if epgrid is not None:
            obj.GridLayer_geteps(epgrid)

        # planewave excitation
        planewave={'p_amp':1,'s_amp':0,'p_phase':0,'s_phase':0}
        obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)

        # compute reflection and transmission
        R,T= obj.RT_Solve(normalize=1)
        # print('R=',R,', T=',T,', R+T=',R+T)
        Rs[i] = np.real(R)
        Ts[i] = np.real(T)
        Rss[z,i] = np.real(R)
        Tss[z,i] = np.real(T)

Rs = np.mean(Rss,axis=0)
Ts = np.mean(Tss,axis=0)
print("This computation took %s seconds to run" % round((time.time() - start_time),4))
As = 1 - Rs - Ts

plt.plot(wv_sweep, Rs)
plt.plot(wv_sweep, Ts)
plt.plot(wv_sweep, As)
plt.title('Reflection, Absorption, and Transmission of Data Set')
plt.xlabel('Nanometers')
plt.ylabel('Percentage of Total R/T/A')
plt.legend(['Reflection','Transmission','Absorption'], loc='upper right')
plt.show()

plt.plot(wv_sweep, Ts)
plt.title('Transmission of Data Set')
plt.xlabel('Nanometers')
plt.ylabel('Percentage of Total Transmission')
plt.show()
