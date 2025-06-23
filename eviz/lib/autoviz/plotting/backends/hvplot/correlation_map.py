def compute_correlation_map(self, data1, data2):
    """Compute pixel-wise Pearson correlation coefficient using xarray's optimized methods."""
    import xarray as xr
    
    # Ensure both datasets have the same dimensions
    if data1.shape != data2.shape:
        self.logger.warning(f"Data shapes don't match: {data1.shape} vs {data2.shape}")
    
    # For 3D data (time, lat, lon)
    if len(data1.dims) == 3:
        # Identify time dimension (usually the first dimension)
        time_dim = data1.dims[0]
        
        # Use xarray's built-in correlation method
        # This computes correlation along the time dimension for each grid point
        corr = xr.corr(data1, data2, dim=time_dim)
        
        # Add metadata
        corr.attrs['long_name'] = 'Pearson Correlation Coefficient'
        corr.attrs['units'] = 'dimensionless'
        corr.attrs['description'] = f'Correlation between {data1.name} and {data2.name} across time'
        
        return corr
    
    # For 2D data, compute spatial correlation
    elif len(data1.dims) == 2:
        # Flatten the arrays
        flat1 = data1.stack(points=data1.dims)
        flat2 = data2.stack(points=data2.dims)
        
        # Compute correlation
        corr_value = xr.corr(flat1, flat2, dim='points').values.item()
        
        # Create a correlation map with the same shape as data1
        corr = xr.full_like(data1, corr_value)
        
        # Add metadata
        corr.attrs['long_name'] = 'Pearson Correlation Coefficient (Spatial)'
        corr.attrs['units'] = 'dimensionless'
        
        return corr
    
    else:
        self.logger.error(f"Unsupported data dimensions: {data1.dims}")
        return None



def compute_correlation_map(self, data1, data2):
    """Compute correlation using Dask for parallel processing."""
    import xarray as xr
    
    # Convert to dask arrays if they aren't already
    if not data1.chunks:
        data1 = data1.chunk({'time': -1, 'lat': 'auto', 'lon': 'auto'})
    if not data2.chunks:
        data2 = data2.chunk({'time': -1, 'lat': 'auto', 'lon': 'auto'})
    
    # Identify time dimension
    time_dim = data1.dims[0]
    
    # Compute correlation along time dimension
    corr = xr.corr(data1, data2, dim=time_dim)
    
    # Compute the result
    result = corr.compute()
    
    return result



def compute_correlation_map(self, data1, data2):
    """Compute correlation using SciPy's optimized functions."""
    from scipy import stats
    import numpy as np
    import xarray as xr
    
    # For 3D data (time, lat, lon)
    if len(data1.dims) == 3:
        # Get dimension names
        time_dim = data1.dims[0]
        lat_dim = data1.dims[1]
        lon_dim = data1.dims[2]
        
        # Get shapes
        n_time, n_lat, n_lon = data1.shape
        
        # Initialize correlation array
        corr_array = np.full((n_lat, n_lon), np.nan)
        
        # Compute correlation for each grid point
        for i in range(n_lat):
            for j in range(n_lon):
                # Extract time series at this grid point
                ts1 = data1[:, i, j].values
                ts2 = data2[:, i, j].values
                
                # Create mask for valid values
                mask = ~np.isnan(ts1) & ~np.isnan(ts2)
                
                # Compute correlation if we have enough valid points
                if np.sum(mask) >= 2:
                    r, _ = stats.pearsonr(ts1[mask], ts2[mask])
                    corr_array[i, j] = r
        
        # Create xarray DataArray with the same coordinates as input
        corr = xr.DataArray(
            corr_array,
            dims=[lat_dim, lon_dim],
            coords={
                lat_dim: data1[lat_dim],
                lon_dim: data1[lon_dim]
            }
        )
        
        return corr



def compute_correlation_map(self, data1, data2):
    """Compute correlation using xarray's apply_ufunc with Dask."""
    import xarray as xr
    import numpy as np
    from scipy import stats
    
    def pearson_corr(x, y):
        """Compute Pearson correlation along first axis."""
        # Handle NaN values
        mask = ~np.isnan(x) & ~np.isnan(y)
        if np.sum(mask) < 2:
            return np.nan
        
        # Compute correlation
        return stats.pearsonr(x[mask], y[mask])[0]
    
    # Ensure data is chunked for parallel processing
    if not data1.chunks:
        data1 = data1.chunk({'time': -1, 'lat': 'auto', 'lon': 'auto'})
    if not data2.chunks:
        data2 = data2.chunk({'time': -1, 'lat': 'auto', 'lon': 'auto'})
    
    # Get time dimension
    time_dim = data1.dims[0]
    
    # Apply correlation function along time dimension
    corr = xr.apply_ufunc(
        pearson_corr,
        data1, data2,
        input_core_dims=[[time_dim], [time_dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
    )
    
    return corr


def compute_correlation_map_custom(self, data1, data2):
    """Compute pixel-wise Pearson correlation coefficient between two datasets.
    
    Args:
        data1: First dataset (xarray.DataArray)
        data2: Second dataset (xarray.DataArray)
        
    Returns:
        xarray.DataArray: Correlation coefficient map
    """
    self.logger.debug("Computing correlation map")
    
    # Ensure both datasets have the same dimensions and coordinates
    if not isinstance(data1, xr.DataArray) or not isinstance(data2, xr.DataArray):
        self.logger.error("Both inputs must be xarray DataArrays")
        return None
        
    # Check if we need to regrid
    if data1.shape != data2.shape:
        self.logger.warning(f"Data shapes don't match: {data1.shape} vs {data2.shape}, correlation may not be meaningful")
    
    dims1 = list(data1.dims)
    dims2 = list(data2.dims)
    
    if len(dims1) == 3 and len(dims2) == 3:
        self.logger.debug(f"Processing 3D data with dimensions {dims1}")
        
        # Identify time dimension (usually the first dimension)
        time_dim = dims1[0]
        
        spatial_dims = dims1[1:]
        
        # Create output array with same spatial coordinates as data1
        # but without the time dimension
        template = data1.isel({time_dim: 0}).copy()
        corr_data = xr.zeros_like(template)
        
        time_len = data1.shape[0]
        y_len, x_len = data1.shape[1], data1.shape[2]
        
        self.logger.debug(f"Computing correlation across {time_len} time points for a {y_len}x{x_len} grid")
        
        # Compute correlation coefficient for each grid point
        for i in range(y_len):
            for j in range(x_len):
                ts1 = data1[:, i, j].values
                ts2 = data2[:, i, j].values
                
                if np.isnan(ts1).all() or np.isnan(ts2).all():
                    corr_data[i, j] = np.nan
                    continue
                    
                # Remove NaNs for correlation calculation
                mask = ~np.isnan(ts1) & ~np.isnan(ts2)
                if np.sum(mask) < 2:  # Need at least 2 points for correlation
                    corr_data[i, j] = np.nan
                    continue
                    
                try:
                    r, _ = pearsonr(ts1[mask], ts2[mask])
                    corr_data[i, j] = r
                except Exception as e:
                    self.logger.debug(f"Error computing correlation at point ({i},{j}): {e}")
                    corr_data[i, j] = np.nan
        
        corr_data.attrs['long_name'] = 'Pearson Correlation Coefficient'
        corr_data.attrs['units'] = 'dimensionless'
        corr_data.attrs['description'] = f'Correlation between {data1.name} and {data2.name} across time'
        
        return corr_data
        
    elif len(dims1) == 2 and len(dims2) == 2:
        self.logger.debug(f"Processing 2D data with dimensions {dims1}")
        
        corr_data = xr.zeros_like(data1)
        
        # For 2D data, we can't compute pixel-wise correlation across time
        # Instead, compute spatial correlation or return a warning
        self.logger.warning("Input data is 2D, computing spatial correlation instead of pixel-wise temporal correlation")
        
        flat1 = data1.values.flatten()
        flat2 = data2.values.flatten()
        
        mask = ~np.isnan(flat1) & ~np.isnan(flat2)
        if np.sum(mask) < 2:
            self.logger.error("Not enough valid data points for correlation")
            return None
            
        try:
            r, _ = pearsonr(flat1[mask], flat2[mask])
            # Fill the entire map with this value
            corr_data.values.fill(r)
            
            corr_data.attrs['long_name'] = 'Pearson Correlation Coefficient (Spatial)'
            corr_data.attrs['units'] = 'dimensionless'
            corr_data.attrs['description'] = f'Spatial correlation between {data1.name} and {data2.name}'
            
            return corr_data
        except Exception as e:
            self.logger.error(f"Error computing spatial correlation: {e}")
            return None
    else:
        self.logger.error(f"Unsupported data dimensions: {dims1} and {dims2}")
        return None