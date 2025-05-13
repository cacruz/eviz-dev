import pytest
import xarray as xr


def test_timeseries_is_a_dataset(create_timeseries_dataset):
    ds = create_timeseries_dataset()
    assert (isinstance(ds, xr.Dataset))


def test_timeseries_dims_number_time_levels(create_timeseries_dataset):
    ds = create_timeseries_dataset()
    assert ds.dims['time'] == 731


def test_timeseries_coords_number_time_levels(create_timeseries_dataset):
    ds = create_timeseries_dataset()
    assert ds.coords['time'].size == 731


def test_timeseries_var_size(create_timeseries_dataset):
    ds = create_timeseries_dataset()
    assert ds.variables['tmin'].size == ds.dims['time']*len(ds.coords['location'])
    assert ds.variables['tmax'].size == ds.dims['time']*len(ds.coords['location'])


def test_4d_is_a_dataset(create_4d_dataset):
    ds = create_4d_dataset()
    assert (isinstance(ds, xr.Dataset))


def test_4d_dataset_number_time_levels(create_4d_dataset):
    ds = create_4d_dataset()
    assert ds.dims['time'] == 366


def test_get_dst_attribute_title(create_4d_dataset):
    ds = create_4d_dataset()
    assert ds.Title == "EViz test data"


def test_get_dst_attribute_start_date(create_4d_dataset):
    ds = create_4d_dataset()
    assert ds.Start_date == "2022-01-01"


def test_get_dst_attribute_map_projection(create_4d_dataset):
    ds = create_4d_dataset()
    assert ds.MAP_PROJECTION == "Lambert Conformal"


def test_get_dst_attribute_sw_corner_lat(create_4d_dataset):
    ds = create_4d_dataset()
    assert ds.SOUTH_WEST_CORNER_LAT == 35.0


def test_get_dst_attribute_sw_corner_lon(create_4d_dataset):
    ds = create_4d_dataset()
    assert ds.SOUTH_WEST_CORNER_LON == -105.0


def test_get_dst_attribute_truelat1(create_4d_dataset):
    ds = create_4d_dataset()
    assert ds.TRUELAT1 == 40.0


def test_get_dst_attribute_truelat2(create_4d_dataset):
    ds = create_4d_dataset()
    assert ds.TRUELAT2 == 35.0


def test_get_dst_attribute_standard_lon(create_4d_dataset):
    ds = create_4d_dataset()
    assert ds.STANDARD_LON == -99.0


@pytest.mark.parametrize(
    ('key', 'expected'),
    (
            ('sfc_press', 'Pa'),
            ('air_temp', 'K'),
            ('rel_humid', '%'),
            ('spc_humid', 'kg kg-1'),
            ('ozone', 'mol mol-1 dry'),
            ('nitrogen_dioxide', 'mol mol-1 dry'),
    )
)
def test_4d_dataset_vars_units(create_4d_dataset, key, expected):
    ds = create_4d_dataset()
    assert ds.data_vars[key].attrs.get("units") == expected
