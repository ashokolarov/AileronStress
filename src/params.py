import numpy as np


d2r = np.pi / 180
r2d = 180 / np.pi

_Nz = 81 # Aerodynamic data grid points along chord
_Nx = 41 # Aerodynamic data grid points along span

datafile = "aerodynamicloadf100.dat"

_Ca = 0.505              # m
_la = 1.611              # m
_x1 = 0.125              # m
_x2 = 0.498              # m
_x3 = 1.494              # m
_xa = 0.245              # m
_ha = 0.161              # m
_tsk = 1.1E-3            # m
_tsp = 2.4E-3            # m
_tst = 1.2E-3            # m
_hst = 13.E-3            # m
_wst = 17.E-3            # m
_nst = 11                # -
_d1 = 0.00389            # m
_d3 = 0.01245            # m
_theta = 30 * d2r        # rad
_P = 49.2E3              # N
_E = 73.1E9              # E-modulus (Pa)
_G = 28.0E9              # G-modulus (Pa)


# List holding section parameters' names
section_params = ["Ca", "la", "x1", "x2", "x3", "xa",
                      "ha", "tsk", "tsp", "tst", "hst",
                      "wst", "nst", "E", "G"]

# Dictionary holding section parameters with their respective names
section_params_dict = dict((name, eval('_' + name)) for name in section_params)

loading_params = ["P", "theta", "d1", "d3", "Nx", "Nz"]
loading_params_dict = dict((name, eval('_' + name)) for name in loading_params)

