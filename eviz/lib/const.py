import os
import sys
import yaml


class _const:
    """ Class that defines all eViz constants used throughout the code. """
    ROOT_FILEPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    # Cached data in NCCS DISCOVER system:
    CARTOPY_DATA_DIR = '/discover/nobackup/projects/jh_tutorials/JH_examples/JH_datafiles/Cartopy'
    # Airmass file used for units conversion
    AIRMASS_URL = 'https://portal.nccs.nasa.gov/datashare/astg/eviz/airmass/RefD2.tavg24_3d_dac_Np.AIRMASS.ANN.nc4'

    class ConstError(TypeError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__.keys():
            # print(("Cannot rebind const(%s)" % name))
            return
            # raise self.ConstError("Cannot rebind const(%s)" % name)
        self.__dict__[name] = value

    def __init__(self):

        data_path = 'data'
        # Default config format
        if 'EVIZ_DATA_PATH' in os.environ:
            data_path = os.environ['EVIZ_DATA_PATH']
        setattr(self, 'data_path', data_path)
        if 'EVIZ_OUTPUT_PATH' in os.environ:
            output_path = os.environ['EVIZ_OUTPUT_PATH']
        else:
            output_path = './output_plots'
        setattr(self, 'output_path', output_path)
        if 'EVIZ_CONFIG_PATH' in os.environ:
            config_path = os.environ['EVIZ_CONFIG_PATH']
        else:
            config_path = os.path.join(self.ROOT_FILEPATH, 'config')
        setattr(self, 'config_path', config_path)

        setattr(self, 'default_input', data_path)
        setattr(self, 'default_output', output_path)

        # Define eViz models, plot types, supported models, and some constants
        supported_models = ['geos', 'ccm', 'cf', 'wrf', 'lis', 'generic',
                            'fluxnet', 'airnow', 'test', 'omi', 'landsat', 'mopitt']
        setattr(self, 'supported_models', supported_models)

        # All supported models have YAML file under the config directory
        for name in supported_models:
            setattr(self, name, name)
            setattr(self, name + '_yaml_name', name + '.yaml')
            setattr(self, name + '_yaml_path', os.path.join(self.ROOT_FILEPATH, 'config', name, name + '.yaml'))

        setattr(self, 'ccm_db_yaml_path', os.path.join(self.ROOT_FILEPATH, 'config', 'ccm', 'database.yaml'))

        # Define plot type names by appending "plot" to each type
        plot_types = ['xy', 'yz', 'xt', 'tx', 'polar', 'sc']
        setattr(self, 'plot_types', plot_types)
        for item in plot_types:
            setattr(self, item + 'plot', item + 'plot')
            setattr(self, 'plot_str_' + item, item)

        # Default image format
        setattr(self, 'format_png', 'png')

        # YAML files with supported model attributes and model coordinates names
        setattr(self, 'meta_attrs_name', 'meta_attributes.yaml')
        setattr(self, 'meta_coords_name', 'meta_coordinates.yaml')
        setattr(self, 'species_db_name', 'species_database.yaml')
        setattr(self, 'meta_attrs_path', os.path.join(self.ROOT_FILEPATH, 'config', 'meta_attributes.yaml'))
        setattr(self, 'meta_coords_path', os.path.join(self.ROOT_FILEPATH, 'config', 'meta_coordinates.yaml'))
        setattr(self, 'species_db_path', os.path.join(self.ROOT_FILEPATH, 'config', 'species_database.yaml'))

        # Physical constants
        setattr(self, 'AVOGADRO', 6.022140857e+23)  # [mol-1]
        setattr(self, 'BOLTZ', 1.38064852e-23)      # [J/K]
        setattr(self, 'G', 9.80665)                 # [m s-2]
        setattr(self, 'R_EARTH_m', 6371.0072e+3)    # [m]
        setattr(self, 'R_EARTH_km', 6371.0072)      # [km]
        setattr(self, 'MW_AIR_g', 28.9644)          # molar mass of air [ [g mol-1]
        setattr(self, 'MW_AIR_kg', 28.9644e-3)      # molar mass of air [ [kg mol-1]
        setattr(self, 'MW_H2O_g', 18.016)           # Molar mass of water [g mol-1]
        setattr(self, 'MW_H2O_kg', 18.016e-3)       # Molar mass of water [kg mol-1]
        setattr(self, 'RD', 287.0)                  # Gas constant in dry air [J/K/kg]
        setattr(self, 'RSTARG', 8.3144598)          # Molar gas constant [J/K/mol]
        setattr(self, 'RV', 461.0)                  # Gas constant for water vapor [J/K/kg]
        # --- scaling factor for turning vmr into pcol
        # --- (note 1*e-09 because in ppb))
        xp_const = (getattr(self, 'AVOGADRO') * 10) / (getattr(self, 'MW_AIR_g') * getattr(self, 'G')) * 1e-09
        setattr(self, 'XP_CONST', xp_const)

        self._set_meta_coords()
        self._set_meta_attrs()

    def _set_meta_coords(self):
        """ Read meta coordinates YAML file and set as class attributes"""
        root_path = os.path.dirname(os.path.abspath('autoviz_base.py'))
        coord_file_path = self.__getattribute__('meta_coords_path')
        if not os.path.exists(coord_file_path):
            coord_file_path = os.path.join(root_path, self.__getattribute__('meta_coords_path'))
        data_dict = read_yaml_file(coord_file_path)
        for k, v in data_dict.items():
            setattr(self, k, v)

    def _set_meta_attrs(self):
        """ Read meta attributes YAML file and set as class attributes"""
        root_path = os.path.dirname(os.path.abspath('autoviz_base.py'))
        attr_file_path = self.__getattribute__('meta_attrs_path')
        if not os.path.exists(attr_file_path):
            attr_file_path = os.path.join(root_path, self.__getattribute__('meta_attrs_path'))
        data_dict = read_yaml_file(attr_file_path)
        for k, v in data_dict.items():
            setattr(self, k, v)


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error while parsing YAML: {e}")
            return None


sys.modules[__name__] = _const()
