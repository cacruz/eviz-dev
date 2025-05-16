import pytest
import eviz.lib.autoviz.utils as p


@pytest.mark.parametrize(
    ('panels_shape', 'fs_expected'),
    (
        ((1, 1), 12),
        ((3, 1), 12),
        ((2, 2), 12),
        (None, 8),
    )
)
def test_axis_tick_font_size(panels_shape, fs_expected):
    assert p.axis_tick_font_size(panels_shape) == fs_expected


@pytest.mark.parametrize(
    ('panels_shape', 'bfs_expected'),
    (
        ((1, 1), 12),
        ((3, 1), 10),
        ((2, 2), 10),
        (None, 8),
    )
)
def test_bar_font_size(panels_shape, bfs_expected):
    assert p.bar_font_size(panels_shape) == bfs_expected


@pytest.mark.parametrize(
    ('panels_shape', 'factor_expected'),
    (
        ((1, 1), 1.0),
        ((3, 1), 0.75),
        ((2, 2), 0.75),
        (None, 0.5),
    )
)
def test_cbar_shrink(panels_shape, factor_expected):
    assert p.cbar_shrink(panels_shape) == factor_expected


@pytest.mark.parametrize(
    ('panels_shape', 'label_fs_expected'),
    (
        ((1, 1), 12),
        ((3, 1), 10),
        ((2, 2), 10),
        (None, 8),
    )
)
def test_axes_label_font_size(panels_shape, label_fs_expected):
    assert p.axes_label_font_size(panels_shape) == label_fs_expected


@pytest.mark.skip(reason="Need to fix this test")
@pytest.mark.parametrize(
    ('panels_shape', 'cbar_pad_expected'),
    (
        ((1, 1), 0.05),
        ((3, 1), 0.15),
        ((2, 2), 0.05),
        (None, 0.05),
    )
)
def test_cbar_pad(panels_shape, cbar_pad_expected):
    assert p.cbar_pad(panels_shape) == cbar_pad_expected


@pytest.mark.parametrize(
    ('panels_shape', 'cbar_frac_expected'),
    (
        ((1, 1), 0.05),
        ((3, 1), 0.1),
        ((2, 2), 0.05),
        (None, 0.05),
    )
)
def test_cbar_fraction(panels_shape, cbar_frac_expected):
    assert p.cbar_fraction(panels_shape) == cbar_frac_expected


@pytest.mark.parametrize(
    ('panels_shape', 'image_fs_expected'),
    (
        ((1, 1), 16),
        ((3, 1), 14),
        ((2, 2), 14),
        (None, 'small'),
    )
)
def test_image_font_size(panels_shape, image_fs_expected):
    assert p.image_font_size(panels_shape) == image_fs_expected


@pytest.mark.parametrize(
    ('panels_shape', 'subplot_title_fs_expected'),
    (
        ((1, 1), 14),
        ((3, 1), 12),
        ((2, 2), 12),
        (None, 10),
    )
)
def test_subplot_title_font_size(panels_shape, subplot_title_fs_expected):
    assert p.subplot_title_font_size(panels_shape) == subplot_title_fs_expected


# TODO: Use title_font_size() or remove!
@pytest.mark.parametrize(
    ('panels_shape', 'title_fs_expected'),
    (
        ((1, 1), 14),
        ((3, 1), 12),
        ((2, 2), 12),
        (None, 12),
    )
)
def test_title_font_size(panels_shape, title_fs_expected):
    assert p.title_font_size(panels_shape) == title_fs_expected


# TODO: Use contour_label_size() or remove!
@pytest.mark.parametrize(
    ('panels_shape', 'contour_label_size_expected'),
    (
        ((1, 1), 8),
        ((3, 1), 8),
        ((2, 2), 8),
        (None, 8),
    )
)
def test_contour_label_size(panels_shape, contour_label_size_expected):
    assert p.contour_label_size(panels_shape) == contour_label_size_expected


# TODO: Use contour_levels_plot() or remove!
def test_contour_levels_plot_empty():
    clevs = []
    assert len(p.contour_levels_plot(clevs)) == 0
