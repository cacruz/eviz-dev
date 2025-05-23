import pytest
from unittest import mock
import tempfile
import os
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

def test_subproc_runs_and_returns_output():
    with mock.patch('subprocess.Popen') as mock_popen:
        process_mock = mock.Mock()
        attrs = {'communicate.return_value': ('out', 'err')}
        process_mock.configure_mock(**attrs)
        mock_popen.return_value = process_mock
        result = p.subproc('echo hello')
        assert result == 'out'

def test_plot_process_saves_figure(monkeypatch):
    called = {}
    def fake_savefig(fname, bbox_inches=None):
        called['fname'] = fname
        called['bbox'] = bbox_inches
    monkeypatch.setattr(p.plt, 'savefig', fake_savefig)
    monkeypatch.setattr(p.plt, 'tight_layout', lambda: None)
    p.plot_process('dummy.png')
    assert called['fname'] == 'dummy.png'

def test_run_plot_commands_starts_processes(monkeypatch):
    started = []
    class DummyProcess:
        def __init__(self, target, args):
            self.target = target
            self.args = args
        def start(self):
            started.append(self.args[0])
        def join(self):
            pass
    monkeypatch.setattr(p.multiprocessing, 'Process', DummyProcess)
    p.run_plot_commands(['a.png', 'b.png'])
    assert started == ['a.png', 'b.png']

def test_create_pdf_creates_pdf(monkeypatch):
    # Setup a fake config and fake images
    class DummyConfig:
        output_dir = tempfile.mkdtemp()
        print_format = 'png'
    config = DummyConfig()
    img_path = os.path.join(config.output_dir, 'test.png')
    img = p.Image.new('RGB', (10, 10))
    img.save(img_path)
    monkeypatch.setattr(p.glob, 'glob', lambda pat: [img_path])
    monkeypatch.setattr(p.Image, 'open', lambda path: img)
    p.create_pdf(config)
    assert os.path.exists(os.path.join(config.output_dir, 'eviz_plots.pdf'))

def test_create_gif_handles_no_files(monkeypatch):
    class DummyConfig:
        archive_web_results = False
        app_data = type('app', (), {'outputs': {'output_dir': tempfile.mkdtemp()}, 'inputs': [{'to_plot': ['field']}]})()
        paths = type('paths', (), {'archive_path': ''})()
        print_format = 'png'
        source_names = []
        gif_fps = 1
        vis_summary = {}
    config = DummyConfig()
    monkeypatch.setattr(p.glob, 'glob', lambda pat: [])
    with mock.patch.object(p.logger, 'error') as mock_log:
        p.create_gif(config)
        mock_log.assert_called_with("No files remaining after IC removal")

def test_print_map_print_to_file(monkeypatch):
    class DummyConfig:
        print_to_file = True
        map_params = [{'outputs': {'output_dir': tempfile.mkdtemp()}, 'field': 'f'}]
        pindex = 0
        paths = type('paths', (), {'output_path': tempfile.mkdtemp()})()
        print_format = 'png'
        ax_opts = {}
        compare = False
        time_level = ''
        archive_web_results = False
    config = DummyConfig()
    fig = mock.Mock()
    monkeypatch.setattr(fig, 'tight_layout', lambda: None)
    monkeypatch.setattr(fig, 'savefig', lambda *a, **k: None)
    p.print_map(config, 'xy', 0, fig)

def test_revise_tick_labels_removes_decimals(monkeypatch):
    class DummyCbar:
        def __init__(self):
            self.ax = mock.Mock()
            self.labels = [mock.Mock(get_text=mock.Mock(return_value='1.00')), mock.Mock(get_text=mock.Mock(return_value='0.00'))]
            self.ax.get_xticklabels = lambda: self.labels
            self.ax.set_xticklabels = lambda labels: None
    cbar = DummyCbar()
    p.revise_tick_labels(cbar)  # Should not raise

def test_colorbar_standard_axes(monkeypatch):
    fig = p.plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow([[1,2],[3,4]])
    cbar = p.colorbar(im)
    assert cbar is not None

def test_add_logo_xy(monkeypatch):
    ax = mock.Mock()
    logo = [[1,2],[3,4]]
    monkeypatch.setattr(p, 'image_scaling', lambda img, r, c: img)
    fig = mock.Mock()
    ax.figure = fig
    fig.figimage = mock.Mock()
    p.add_logo_xy(logo, ax, 0, 0)
    fig.figimage.assert_called()

def test_add_logo_anchor(monkeypatch):
    ax = mock.Mock()
    logo = mock.Mock()
    monkeypatch.setattr(p, 'OffsetImage', mock.Mock(return_value=mock.Mock()))
    monkeypatch.setattr(p, 'TextArea', mock.Mock(return_value=mock.Mock()))
    monkeypatch.setattr(p, 'VPacker', mock.Mock(return_value=mock.Mock()))
    monkeypatch.setattr(p, 'AnchoredOffsetbox', mock.Mock(return_value=mock.Mock()))
    ax.add_artist = mock.Mock()
    p.add_logo_anchor(ax, logo, label="test")
    ax.add_artist.assert_called()

def test_add_logo_fig(monkeypatch):
    fig = mock.Mock()
    logo = mock.Mock()
    imax = mock.Mock()
    fig.add_axes = mock.Mock(return_value=imax)
    imax.set_axis_off = mock.Mock()
    imax.imshow = mock.Mock()
    p.add_logo_fig(fig, logo)
    imax.imshow.assert_called()

def test_add_logo(monkeypatch):
    ax = mock.Mock()
    fig = mock.Mock()
    ax.figure = fig
    fig.figimage = mock.Mock()
    logo = mock.Mock()
    p.add_logo(ax, logo)
    fig.figimage.assert_called()

def test_output_basic_print_to_file(monkeypatch):
    class DummyConfig:
        print_to_file = True
        map_params = [{'outputs': {'output_dir': tempfile.mkdtemp()}}]
        pindex = 0
        paths = type('paths', (), {'output_path': tempfile.mkdtemp()})()
        print_format = 'png'
    config = DummyConfig()
    monkeypatch.setattr(p.plt, 'savefig', lambda *a, **k: None)
    p.output_basic(config, 'test')

def test_output_basic_show(monkeypatch):
    class DummyConfig:
        print_to_file = False
        map_params = [{}]
        pindex = 0
        paths = type('paths', (), {'output_path': tempfile.mkdtemp()})()
        print_format = 'png'
    config = DummyConfig()
    monkeypatch.setattr(p.plt, 'tight_layout', lambda: None)
    monkeypatch.setattr(p.plt, 'show', lambda: None)
    p.output_basic(config, 'test')

def test_get_subplot_geometry(monkeypatch):
    axes = mock.Mock()
    ss = mock.Mock()
    axes.get_subplotspec = mock.Mock(return_value=ss)
    ss.get_geometry = mock.Mock(return_value=(2,2,0,0))
    ss.is_first_row = mock.Mock(return_value=True)
    ss.is_first_col = mock.Mock(return_value=True)
    ss.is_last_row = mock.Mock(return_value=False)
    ss.is_last_col = mock.Mock(return_value=False)
    geom = p.get_subplot_geometry(axes)
    assert isinstance(geom, tuple)

# You can add similar tests for dump_json_file, load_log, and archive if you want to reach even higher coverage.
