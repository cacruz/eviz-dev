import eviz.lib.data.utils as du
import pytest


@pytest.mark.skip(reason="Need to create mock datasets")
def test_get_dst_attribute_str(create_4d_dataset):
    ds = create_4d_dataset()
    assert du.get_dst_attribute(ds, "Title") == "EViz test data"


@pytest.mark.skip(reason="Need to create mock datasets")
def test_get_dst_attribute_float(create_4d_dataset):
    ds = create_4d_dataset()
    assert du.get_dst_attribute(ds, "STANDARD_LON") == -99.0


@pytest.mark.skip(reason="Need to create mock datasets")
def test_compute_means_column_1M(create_column_dataset):
    ds = create_column_dataset()
    da = ds.LWdown
    assert du.compute_means(da, means="1M")[0].values == 0.4385722446796244


@pytest.mark.skip(reason="Need to create mock datasets")
def test_compute_means_column_1A(create_column_dataset):
    ds = create_column_dataset()
    da = ds.LWdown
    assert du.compute_means(da, means="1A")[0].values == 0.48957698650489045


@pytest.mark.skip(reason="Need to create mock datasets")
@pytest.mark.parametrize(
    ('location', 'expected'),
    (
        (0, -1.430873271138674),
        (1, -0.7304975217394388),
        (2, -1.213066694353026),
    )
)
def test_timeseries_dataset_seasonal(location, expected, create_timeseries_dataset):
    ds = create_timeseries_dataset()
    da = ds.tmin
    assert du.compute_means(da, means="QS-JAN")[0][location].values == pytest.approx(expected, abs=1e-7)
