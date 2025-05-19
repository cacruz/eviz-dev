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


def test_natural_key():
    assert p.natural_key("file10.png") == ['file', 10, '.png']
    assert p.natural_key("abc2def3") == ['abc', 2, 'def', 3, '']


def test_formatted_contours():
    assert p.formatted_contours([1.0, 2.0, 3.5, 4]) == [1, 2, 3.5, 4]


@pytest.mark.parametrize("panels,expected", [
    ((1, 1), 12),
    ((3, 1), 12),
    ((2, 2), 12),
    (None, 8),
])
def test_axis_tick_font_size(panels, expected):
    assert p.axis_tick_font_size(panels) == expected


@pytest.mark.parametrize("panels,expected", [
    ((1, 1), 10),
    ((3, 1), 8),
    ((2, 2), 8),
    (None, 8),
])
def test_contour_tick_font_size(panels, expected):
    assert p.contour_tick_font_size(panels) == expected


@pytest.mark.parametrize("panels,expected", [
    ((1, 1), 0.05),
    ((3, 1), 0.1),
    ((2, 2), 0.05),
    (None, 0.05),
])
def test_cbar_fraction(panels, expected):
    assert p.cbar_fraction(panels) == expected

def test_contour_levels_plot():
    assert p.contour_levels_plot([1, 2.0, 3.0, 4.5, 5.0]) == [1, 2, 3, 4.5, 5]
    assert p.contour_levels_plot([]) == []


def test_fmt_two_digits():
    assert p.fmt_two_digits(1.234, None) == '[1.23]'


def test_fmt():
    # Should return a string with scientific notation
    s = p.fmt(1234, None)
    assert r'\times 10^' in s


def test_fmt_once():
    s = p.fmt_once(1234, None)
    assert s.startswith('$')


def test_image_scaling():
    arr = [[1, 2], [3, 4]]
    scaled = p.image_scaling(arr, 2, 2)
    assert scaled == [[1, 2], [3, 4]]


def test_get_subplot_shape():
    assert p.get_subplot_shape(1) == (1, 1)
    assert p.get_subplot_shape(2) == (2, 1)
    assert p.get_subplot_shape(4) == (2, 2)
    assert p.get_subplot_shape(5) == (3, 2)


def test_contour_format_from_levels():
    fmt = p.contour_format_from_levels([1, 2, 3])
    assert isinstance(fmt, str) or callable(fmt)


def test_OOMFormatter_custom_format():
    f = p.OOMFormatter(order=0, prec=2)
    assert f._custom_format(1.234, 0) == '1.23'


def test_FlexibleOOMFormatter_custom_format():
    f = p.FlexibleOOMFormatter(order=0, min_val=1, max_val=100)
    assert f._custom_format(10, 0).startswith('$')


def test_FlexibleOOMFormatter_call():
    f = p.FlexibleOOMFormatter(order=0, min_val=1, max_val=100)
    assert isinstance(f(10), str)