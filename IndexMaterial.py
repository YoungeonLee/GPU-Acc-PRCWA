import numpy as np
from refractiveindex import RefractiveIndexMaterial

def Index_Material(material):
    # a, b, and c reference the page lookups of the materials
    a = 'main'
    b = ''
    c = ''
    new_material = RefractiveIndexMaterial(shelf=a, book=b, page=c)
    return new_material

shelf_list = [
    'main',
    'glass',
    'organic',
    'other',
    'specs'
]