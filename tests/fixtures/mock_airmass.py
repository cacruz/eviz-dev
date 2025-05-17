import numpy as np
import xarray as xr
import os
import eviz.lib.const as constants

def create_mock_airmass_dataset():
    """
    Create a synthetic airmass dataset for testing.
    
    Returns:
        xarray.Dataset: A dataset containing an AIRMASS variable with realistic dimensions
    """
    # Create coordinates
    lat = np.linspace(-90, 90, 18)  # 10-degree resolution
    lon = np.linspace(-180, 180, 36)  # 10-degree resolution
    lev = np.array([1000, 850, 700, 500, 300, 200, 100, 50, 10])  # Pressure levels in hPa
    
    # Create a realistic-looking airmass field
    # Shape will be (levels, lat, lon)
    airmass = np.zeros((len(lev), len(lat), len(lon)))
    
    # Create a basic pressure-based airmass distribution
    for i, pressure in enumerate(lev):
        # Base value decreases with height (pressure)
        base = (pressure / 1000.0) * 10  # kg/m²
        
        # Add latitudinal variation (more mass near equator)
        lat_variation = np.cos(np.radians(lat)) * 2
        
        # Create the layer
        layer = base * lat_variation[:, np.newaxis] * np.ones(len(lon))
        
        # Add some random variation (1% noise)
        noise = np.random.normal(0, 0.01 * base, layer.shape)
        
        airmass[i] = layer + noise
    
    # Create the dataset
    ds = xr.Dataset(
        data_vars={
            "AIRMASS": (["lev", "lat", "lon"], airmass),
        },
        coords={
            "lat": lat,
            "lon": lon,
            "lev": lev,
        },
        attrs={
            "title": "Mock AIRMASS Dataset for Testing",
            "source": "eviz test fixtures",
        }
    )
    
    # Add variable attributes
    ds.AIRMASS.attrs = {
        "long_name": "Air Mass",
        "units": "kg/m2",
        "standard_name": "atmosphere_mass_per_unit_area",
    }
    
    return ds

def create_mock_airmass_dataarray():
    """
    Create a synthetic airmass DataArray for testing.
    
    Returns:
        xarray.DataArray: An AIRMASS variable with realistic dimensions
    """
    return create_mock_airmass_dataset()["AIRMASS"]
