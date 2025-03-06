import numpy as np

def Index_Lookup(material, wavelength):
    if material == 'air':
        return (1+0j)
    else:
        return material.get_epsilon(wavelength)