import mendeleev

ELEMENTS = ['Cs', 'Er', 'Xe', 'Tc', 'Eu', 'Gd', 'Li', 'Hf', 'Dy', 'F', 'Te', 'Ti', 'Hg', 'Bi', 'Pr', 'Ne', 'Sm', 'Be', 'Au', 'Pb', 'C', 'Zr', 'Ir', 'Pd', 'Sc', 'Yb', 'Os', 'Nb', 'Ac', 'Rb', 'Al', 'P', 'Ga', 'Na', 'Cr', 'Ta', 'Br', 'Pu', 'Ge', 'Tb', 'La', 'Se', 'V', 'Pa', 'Ni', 'In', 'Cu', 'Fe', 'Co', 'Pm', 'N', 'K', 'Ca', 'Rh', 'B', 'Tm', 'I', 'Ho', 'Sb', 'As', 'Tl', 'Ru', 'U', 'Np', 'Cl', 'Re', 'Ag', 'Ba', 'H', 'O', 'Mg', 'W', 'Sn', 'Mo', 'Pt', 'Zn', 'Sr', 'S', 'Kr', 'Cd', 'Si', 'Y', 'Lu', 'Th', 'Nd', 'Mn', 'He', 'Ce']

ELEMENTS_SMALL = ['Li', 'Na', 'K', 'Rb', 'Be', 'Ca', 'Mg', 'Sr', 'H', 'C', 'N', 'O', 'P', 'S', 'Se', 'F', 'Cl', 'Br']
ELEMENTS_MEDIUM = ['Li', 'Na', 'K', 'Rb', 'Be', 'Ca', 'Mg', 'Sr', 'H', 'C', 'N', 'O', 'P', 'S', 'Se', 'F', 'Cl','Br', 
                  'B', 'Si', 'Ge', 'Fe', 'Cu', 'Co', 'Ni', 'Mn', 'Al', 'Zn', 'Sn', 'Cr']
ELEMENTS_LARGE = ELEMENTS_MEDIUM + ['In', 'Sb', 'V', 'Mo', 'Ga', 'Ag', 'Ti', 'Ba', 'Y', 'Te', 'I', 'Pd', 'Rh', 'As', 'Pt', 'Cs', 'Au', 'Bi', 'Zr', 'La']

TRANSITION_METALS = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']
LANTHANIDES = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
ACTINIDES = ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu']
NOBLE = ['Xe', 'Ne', 'Kr', 'He']
HALOGENS = ['F', 'Br', 'Cl', 'I']
G1 = ['Li', 'Na', 'K', 'Rb', 'Cs']
G2 = ['Be', 'Mg', 'Ca', 'Sr', 'Ba'] 
NONMETALS = ['H','B', 'C', 'N', 'O', 'Si', 'P', 'S', 'As', 'Se', 'Te']
POST_TRANSITION_METALS = ['Al', 'Ga', 'Ge', 'In', 'Sn', 'Sb', 'Tl', 'Pb', 'Bi']


SPECIES_IND = {i:mendeleev.element(ELEMENTS[i]).atomic_number for i in range(len(ELEMENTS))}
SPECIES_IND_INV = {mendeleev.element(ELEMENTS[i]).atomic_number:i for i in range(len(ELEMENTS))}

SPECIES_IND_SMALL = {i:mendeleev.element(ELEMENTS_SMALL[i]).atomic_number for i in range(len(ELEMENTS_SMALL))}
SPECIES_IND_SMALL_INV = {mendeleev.element(ELEMENTS_SMALL[i]).atomic_number:i for i in range(len(ELEMENTS_SMALL))}
SPECIES_IND_MEDIUM = {i:mendeleev.element(ELEMENTS_MEDIUM[i]).atomic_number for i in range(len(ELEMENTS_MEDIUM))}
SPECIES_IND_MEDIUM_INV = {mendeleev.element(ELEMENTS_MEDIUM[i]).atomic_number:i for i in range(len(ELEMENTS_MEDIUM))}
SPECIES_IND_LARGE = {i:mendeleev.element(ELEMENTS_LARGE[i]).atomic_number for i in range(len(ELEMENTS_LARGE))}
SPECIES_IND_LARGE_INV = {mendeleev.element(ELEMENTS_LARGE[i]).atomic_number:i for i in range(len(ELEMENTS_LARGE))}

SPACE_GROUP_TYPE = {221:'sc', 225:'fcc', 229: 'bcc', 215: 'sc', 200: 'sc'}

CUBIC_MINI = [630, 2271, 8354, 8666, 8906]
