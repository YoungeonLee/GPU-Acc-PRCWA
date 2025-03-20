'''Testing file for created functions in other files.'''
import grcwa
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

from honeycomb_lattice import honeycomb_lattice


'''Input parameters here'''
nG = 100                    # Truncation order
L1 = [0.8,0]                # Lattice constant (x direction)
L2 = [0,0.8]                # Lattice constant (y direction)
theta = 0                   # Incidence light angle

# Patterned layer gridding (Nx*Ny)
Nx = 300
Ny = 300

Np = 100                     # Number of patterned layers
epbkg = 5                  # Dielectric value of uniform sphere
diameter = 1.              # diameter of sphere

print("Now running the honeycomb_lattice function:")
R,T = honeycomb_lattice(nG,L1,L2,theta,Nx,Ny,Np,epbkg,diameter)
print('R=',R,', T=',T,', R+T=',R+T)