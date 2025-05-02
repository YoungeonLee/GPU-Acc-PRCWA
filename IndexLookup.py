from refractiveindex import RefractiveIndexMaterial

SiO2 = RefractiveIndexMaterial(shelf='main', book='SiO2', page='Franta-25C')
HfO2 = RefractiveIndexMaterial(shelf='main', book='HfO2', page='Franta')
Ag = RefractiveIndexMaterial(shelf='main', book='Ag', page='Ciesielski')
Si = RefractiveIndexMaterial(shelf='main', book='Si', page='Franta-25C')
Ti = RefractiveIndexMaterial(shelf='main', book='Ti', page='Werner')
Ti_2 = RefractiveIndexMaterial(shelf='main', book='Ti', page='Ordal')
SodaLime = RefractiveIndexMaterial(shelf='glass', book='soda-lime', page='Rubin-IR')

def Index_Lookup(material, wavelength):
    wavelength = 1000 * wavelength
    if material == 'air':
        return (1+0j)
    elif material == 'sio2':
        return SiO2.get_epsilon(wavelength)
    elif material == 'hfo2':
        return HfO2.get_epsilon(wavelength) 
    elif material == 'ag':
        return Ag.get_epsilon(wavelength) 
    elif material == 'si':
        return Si.get_epsilon(wavelength)
    elif material == 'ti':
        if wavelength < 2480:
            return Ti.get_epsilon(wavelength)
        else:
            return Ti_2.get_epsilon(wavelength)
    elif material == 'sodalime':
        return SodaLime.get_epsilon(wavelength) 
    else:
        raise NotImplementedError
        