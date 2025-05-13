import re
import numpy as np
import logging

logger = logging.getLogger(__name__)


def apply_conversion(config, data2d, name):
    """ Apply a unit conversion based on SPECS file entries

    Parameters:
        config (ConfigManager) configuration object
        data2d (DataArray) with original units
        name (str)

    Returns:
        data2d (DataArray) with target units

    For single-plots, we rely on specs file to determine the units and unit conversion factor
    For comparison plots, we rely on the "target" units specified in the specs file and the unit
    conversion is provided by the Units conversion module.
    """
    # A user specifies units AND unitconversion factor:
    if 'units' in config.spec_data[name] and 'unitconversion' in config.spec_data[name]:
        if "AOA" in name.upper():
            data2d = data2d / np.timedelta64(1, 'ns') / 1000000000 / 86400
        else:
            data2d = data2d * float(config.spec_data[name]['unitconversion'])
    # A user specifies units AND no unitconversion factor, in that case we use units module
    elif 'units' in config.spec_data[name] and not 'unitconversion' in config.spec_data[name]:
        # If field name is a chemical species...
        if name in config.species_db.keys():
            data2d = config.units.convert_chem(data2d, name, config.spec_data[name]['units'])
        else:
            data2d = config.units.convert(data2d, name, config.spec_data[name]['units'])
    else:
        # Special hack for AOA metadump specs
        if "AOA" in name.upper():
            data2d = data2d / np.timedelta64(1, 'ns') / 1000000000 / 86400
        msg = f"No units found for {name}. Will use the given 'dataset' units."
        logger.debug(msg)
    return data2d


def apply_mean(config, d, level=None):
    """ Compute various averages over coordinates """
    if level:
        if level == 'all':
            data2d = d.mean(dim=config.get_model_dim_name('zc'))
        else:
            if len(d.dims) == 3:
                data2d = d.mean(dim=config.get_model_dim_name('tc'))
            else:  # 4D array - we need to select a level
                lev_to_plot = int(np.where(d.coords[config.get_model_dim_name('zc')].values == level)[0])
                logger.debug("Level to plot:" + str(lev_to_plot))
                # select level
                data2d = d.isel(lev=lev_to_plot)
                data2d = data2d.mean(dim=config.get_model_dim_name('tc'))
    else:
        if len(d.dims) == 3:
            data2d = d.mean(dim=config.get_model_dim_name('tc'))
        else:
            d = d.mean(dim=config.get_model_dim_name('xc'))
            data2d = d.mean(dim=config.get_model_dim_name('tc'))

    data2d.attrs = d.attrs.copy()  # retain units
    return data2d.squeeze()


def apply_zsum(config, data2d):
    """ Sum over vertical levels (column sum)"""
    data2d_zsum = data2d.sum(dim='lev')
    data2d_zsum.attrs = data2d.attrs.copy()
    return data2d_zsum.squeeze()
