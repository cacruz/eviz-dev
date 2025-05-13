from mpl_toolkits.basemap import Basemap
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import xarray
 
split = xr.open_zarr('spatial_folds.zarr')
 
longitude = split['longitude'].values
latitude = split['latitude'].values
data = split['fold'].values
 
states = os.path.join('map_files', 'st99_d00')
kwargs = dict(zip(['llcrnrlat', 'urcrnrlat', 'llcrnrlon', 'urcrnrlon'], [20, 52, -128, -64]))
kwargs.update({'lat_0':30, 'lon_0':-98, 'resolution': 'i'})
m = Basemap(**kwargs)
m.readshapefile(states, name='states', drawbounds=True, color='gray', linewidth=0.5, zorder=11)
m.fillcontinents(color=(0,0,0,0), lake_color='#9abee0', zorder=9)
# m.drawrivers(linewidth=0.2, color='blue', zorder=9)
m.drawcountries(color='k', linewidth=0.5)
m.shadedrelief(scale=0.3)
 
lon_proj, lat_proj = np.meshgrid(longitude, latitude)
x, y = m(lon_proj, lat_proj)
mesh = m.pcolormesh(x, y, data, shading='auto', zorder=10)
 
plt.savefig('split.jpg', format='jpg', dpi=300, bbox_inches='tight')
 
# again replace the zarr file
 