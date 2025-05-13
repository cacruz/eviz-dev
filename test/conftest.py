import xarray as xr
import pytest
import os
from argparse import Namespace

from eviz.lib.data.data_source import make_fake_timeseries_dataset
from eviz.lib.data.data_source import make_fake_4D_dataset
from eviz.lib.data.data_source import make_fake_column_dataset
from eviz.lib.autoviz.config import Config
from eviz.lib.autoviz_base import Autoviz
from eviz.lib.autoviz.figure import Figure
import eviz.lib.const as constants


@pytest.fixture(scope="module")
def get_eviz():
    """Fixture that provides a function to create an autoviz object.

    Returns:
        autoviz (Autoviz): an autoviz object
    """
    def _get_eviz():
        # if config (testing)
        config_file = [os.path.join(constants.ROOT_FILEPATH, 'test', 'config', "test.yaml")]
        # else
        # configfile=['test.yaml']
        args = Namespace(sources=['test'], configfile=config_file, config=constants.ROOT_FILEPATH,
                         data_dirs=None, output_dirs=None)
        input_sources = [s.strip() for s in args.sources[0].split(',')]
        eviz = Autoviz(input_sources, args=args)
        return eviz

    return _get_eviz


@pytest.fixture(scope="module")
def get_config_instance(get_config):
    """Provides a single Config instance for all tests in this module."""
    return get_config()


@pytest.fixture(scope="module")
def get_config(create_timeseries_dataset, create_4d_dataset):
    """Fixture that provides a function to fetch a run configuration specified by its name.

    Returns:
        config (Config): a config object
    """
    _ = create_timeseries_dataset()
    _ = create_4d_dataset()

    def _get_config():
        # File path is relative to top-level
        args = Namespace(sources=['test'], configfile=['test.yaml'], config=constants.ROOT_FILEPATH,
                         data_dirs=None, output_dirs=None)
        input_sources = [s.strip() for s in args.sources[0].split(',')]
        # if config (testing)
        config_file = [os.path.join(constants.ROOT_FILEPATH, 'test', 'config', "test.yaml")]
        # if not config, use this:
        # config_file = args.configfile
        config = Config(source_names=input_sources, config_files=config_file)
        return config

    return _get_config


@pytest.fixture(scope="function")
def get_figure():
    """Fixture that provides a function to fetch a figure
    Returns:
        figure (Figure): a Figure object
    """
    def _get_figure(cfg, pt):
        return Figure(cfg, pt)

    return _get_figure


@pytest.fixture(scope='module')
def create_column_dataset(request):
    """Column dataset

    Returns:
        ds (Dataset): an xarray Dataset object
    """

    def _create_column_dataset(clean=False):
        if os.path.exists('test/data/column.nc'):
            return xr.open_dataset('test/data/column.nc')
        ds = make_fake_column_dataset()

        def cleanup():
            os.remove('test/data/column.nc')

        if not clean:
            pass
        else:
            request.addfinalizer(cleanup)
        return ds

    return _create_column_dataset


@pytest.fixture(scope='module')
def create_timeseries_dataset(request):
    """Create a synthetic xarray dataset containing a 1D time series data array.

    Returns:
        ds (Dataset): an xarray Dataset object
    """

    def _create_timeseries_dataset(clean=False):
        if os.path.exists('test/data/timeseries.nc'):
            return xr.open_dataset('test/data/timeseries.nc')
        ds = make_fake_timeseries_dataset()

        def cleanup():
            os.remove('test/data/timeseries.nc')

        if clean:
            request.addfinalizer(cleanup)
        return ds

    return _create_timeseries_dataset


@pytest.fixture(scope='module')
def create_4d_dataset(request):
    """Create a synthetic xarray dataset containing a 4D data array.

    Returns:
        ds (Dataset): an xarray Dataset object
    """

    def _create_4d_dataset(clean=False):
        if os.path.exists('test/data/spacetime.nc'):
            return xr.open_dataset('test/data/spacetime.nc')
        ds = make_fake_4D_dataset()

        def cleanup():
            os.remove('test/data/spacetime.nc')

        if clean:
            request.addfinalizer(cleanup)
        return ds

    return _create_4d_dataset
