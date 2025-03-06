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

#print("Now running the uniform_sphere function:")
#R,T = uniformsphere(nG,L1,L2,theta,Nx,Ny,Np,epbkg,diameter)
#print('R=',R,', T=',T,', R+T=',R+T)
print("Now running the honeycomb_lattice function:")
R,T = honeycomb_lattice(nG,L1,L2,theta,Nx,Ny,Np,epbkg,diameter)
print('R=',R,', T=',T,', R+T=',R+T)

'''

R = np.zeros(90)
T = np.zeros(90)
R1 = np.zeros(90)
T1 = np.zeros(90)
theta = np.linspace(0,89,90)
print(theta)
for angle in range(0,90):
    R[angle],T[angle] = uniformsphere(nG,L1,L2,math.radians(theta[angle]),Nx,Ny,Np,epbkg,diameter)
    R1[angle],T1[angle] = honeycomb_lattice(nG,L1,L2,math.radians(theta[angle]),Nx,Ny,Np,epbkg,diameter)

plt.plot(theta,R, label='Ru')
plt.plot(theta,T,'-.', label='Tu')
plt.plot(theta,R1, label='Rh')
plt.plot(theta,T1,'-.', label='Th')
plt.xlabel("Angle of Incidence Light")
plt.ylabel("Percent of Incidence Light Reflected Transmitted")
plt.legend()
plt.show()

'''

'''This section is for graphing, not always needed, takes about 10 seconds per calculation'''
