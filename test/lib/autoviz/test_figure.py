import pytest
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


# TODO:
# def test_subplots_custom(get_config_instance, get_figure):
#     cfg = get_config_instance()
#     cfg.subplots = (3, 3)
#     figure = get_figure(cfg)
#     assert figure.subplots == (1, 1)

def test_subplots_single_plot_shape(get_config_instance, get_figure):
    cfg = get_config_instance
    cfg.compare = None
    cfg._extra_diff_plot = None
    figure = get_figure(cfg, 'xy')
    try:
        assert figure.subplots == (1, 1)
    finally:
        plt.close(figure.mfig)


def test_subplots_comparison_3x1shape(get_config, get_figure):
    cfg = get_config()
    cfg._extra_diff_plot = None
    figure = get_figure(cfg, 'xy')
    # compare option w/ exp_ids defaults to 3x1 panel
    try:
        assert figure.subplots == (3, 1)
    finally:
        plt.close(figure.mfig)


def test_subplots_comparison_2x2shape(get_config, get_figure):
    cfg = get_config()
    figure = get_figure(cfg, 'xy')
    # compare option w/ exp_ids and extra_diff plot defaults to 2x2 panel
    try:
        assert figure.subplots == (2, 2)
    finally:
        plt.close(figure.mfig)


def test_frame_len(get_config_instance, get_figure):
    cfg = get_config_instance
    figure = get_figure(cfg, 'xy')
    try:
        assert len(figure.frame_params) == 1
    finally:
        plt.close(figure.mfig)


def test_frame_params(get_config, get_figure):
    cfg = get_config()
    figure = get_figure(cfg, 'xy')
    try:
        assert figure.frame_params[0] == [2, 2, 12, 8]
    finally:
        plt.close(figure.mfig)


@pytest.mark.parametrize(
    ('projection', 'expected'),
    (
            ('lambert', ccrs.LambertConformal),
            ('albers', ccrs.AlbersEqualArea),
            ('polar', ccrs.NorthPolarStereo),  # SouthPolar?
            ('ortho', ccrs.Orthographic),
            ('mercator', ccrs.Mercator),
            ('stereo', ccrs.Stereographic),
    )
)
def test_get_projection(projection, expected, get_config_instance, get_figure):
    cfg = get_config_instance
    figure = get_figure(cfg, 'xy')
    try:
        assert isinstance(figure.get_projection(projection), expected)
    finally:
        plt.close(figure.mfig)


def test_get_default_projection(get_config_instance, get_figure):
    cfg = get_config_instance
    figure = get_figure(cfg, 'xy')
    try:
        assert isinstance(figure.get_projection(), ccrs.PlateCarree)
    finally:
        plt.close(figure.mfig)


def test_get_projection_mercator(get_config, get_figure):
    cfg = get_config()
    figure = get_figure(cfg, 'xy')
    try:
        assert isinstance(figure.get_projection('mercator'), ccrs.Mercator)
    finally:
        plt.close(figure.mfig)


@pytest.mark.parametrize(
    ('opt', 'expected'),
    (
            ('boundary', None),
            ('use_pole', 'north'),
            ('profile_dim', None),
            ('tave', True),
            ('taverange', 'all'),
            ('use_cmap', 'jet'),
            ('use_diff_cmap', 'jet'),
            ('use_cmap', 'jet'),
            ('extend_value', 'both'),
            ('norm', 'both'),
            ('title_fontsize', {'title.fontsize': 10}),
            ('subplot_title_fontsize', {'subplot_title.fontsize': 12}),
            ('axes_fontsize', {'axes.fontsize': 10}),
            ('colorbar_fontsize', {'colorbar.fontsize': 8}),
            ('contour_linestyle', {'lines.linewidth': 0.5, 'lines.linestyle': 'solid'}),
            ('time_series_plot_linestyle', {'lines.linewidth': 1, 'lines.linestyle': 'solid'}),
            ('plot_title', None),
            ('clevs', None),
            ('clabel', None),
            ('cscale', None),
            ('zscale', 'linear'),
            ('create_clevs', False),
            ('add_grid', False),
            ('line_contours', True),
            ('add_tropp_height', False),
            ('add_tropp_height', False),
            ('use_cartopy', False),
            ('is_diff_field', False),
            ('add_extra_field_type', False),
            ('torder', None),
            ('extent', [-180, 180, -90, 90]),
            ('projection', None),
            ('num_clevs', 10),
            ('time_lev', 0),
            ('use_cmap', 'jet'),
    )
)
def test_init_ax_opts_defaults(opt, expected, get_config_instance, get_figure):
    cfg = get_config_instance
    figure = get_figure(cfg, 'xy')
    opts = figure.init_ax_opts('air_temp')
    try:
        assert opts[opt] == expected
    finally:
        plt.close(figure.mfig)
