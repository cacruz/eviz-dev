"""
Project-wide constants for eViz.

This module contains only true constants: physical constants, fixed strings,
and other values that do not depend on the runtime environment or configuration.
"""

# Physical constants
AVOGADRO = 6.022140857e+23  # [mol-1]
BOLTZ = 1.38064852e-23      # [J/K]
G = 9.80665                 # [m s-2]
R_EARTH_M = 6371.0072e+3    # [m]
R_EARTH_KM = 6371.0072      # [km]
MW_AIR_G = 28.9644          # [g mol-1]
MW_AIR_KG = 28.9644e-3      # [kg mol-1]
MW_H2O_G = 18.016           # [g mol-1]
MW_H2O_KG = 18.016e-3       # [kg mol-1]
RD = 287.0                  # [J/K/kg]
RSTARG = 8.3144598          # [J/K/mol]
RV = 461.0                  # [J/K/kg]

# Project-wide constants
CARTOPY_DATA_DIR = '/discover/nobackup/projects/jh_tutorials/JH_examples/JH_datafiles/Cartopy'
AIRMASS_URL = 'https://portal.nccs.nasa.gov/datashare/astg/eviz/airmass/RefD2.tavg24_3d_dac_Np.AIRMASS.ANN.nc4'

SUPPORTED_MODELS = [
    'geos', 'ccm', 'cf', 'wrf', 'lis', 'gridded', 'crest',
    'fluxnet', 'airnow', 'test', 'omi', 'landsat', 'mopitt'
]

PLOT_TYPES = ['xy', 'yz', 'xt', 'tx', 'polar', 'sc', 'box', 'corr']

FORMAT_PNG = 'png'
META_ATTRS_NAME = 'meta_attributes.yaml'
META_COORDS_NAME = 'meta_coordinates.yaml'
SPECIES_DB_NAME = 'species_database.yaml'

# Derived/scaling constants
XP_CONST = (AVOGADRO * 10) / (MW_AIR_G * G) * 1e-09  # scaling factor for vmr to pcol (ppb)

CCM_YAML_PATH = "ccm.yaml"
CF_YAML_PATH = "cf.yaml"
GEOS_YAML_PATH = "geos.yaml"
LIS_YAML_PATH = "lis.yaml"
WRF_YAML_PATH = "wrf.yaml"
GRIDDED_YAML_PATH = "gridded.yaml"

XY_PLOT = 'xy'
YZ_PLOT = 'yz'
XT_PLOT = 'xt'
TX_PLOT = 'tx'
POLAR_PLOT = 'polar'
SC_PLOT = 'sc'
BOX_PLOT = 'box'
CORR_PLOT = 'corr'
