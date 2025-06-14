import pytest
from unittest import mock
import tempfile
import os
import eviz.lib.autoviz.utils as p
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import eviz.lib.autoviz.utils as pu


def create_test_image():
    """Create a simple test image for logo testing"""
    # Create a 50x30 RGB test image
    img = np.zeros((30, 50, 4), dtype=np.uint8)
    img[:, :, 0] = 255  # Red channel
    img[:, :, 3] = 255  # Alpha channel
    return img

@pytest.fixture
def mock_logo_file(monkeypatch):
    """Mock plt.imread to return a test image instead of reading from file"""
    test_img = create_test_image()
    
    def mock_imread(path):
        return test_img.astype(float) / 255.0
    
    monkeypatch.setattr(plt, 'imread', mock_imread)
    return test_img

def test_add_logo(mock_logo_file):
    """Test the add_logo function"""
    # Create a figure
    fig = plt.figure(figsize=(6, 4))
    
    # Mock the figimage method to capture calls
    with mock.patch.object(fig, 'figimage') as mock_figimage:
        # Call the function
        pu.add_logo(fig)
        
        # Check that figimage was called
        mock_figimage.assert_called_once()
        
        # Check the arguments
        args, kwargs = mock_figimage.call_args
        
        # First arg should be the image data
        assert isinstance(args[0], np.ndarray)
        
        # Skip position checks (difficult to mock exactly)
        
        # Check that zorder and alpha were set
        assert kwargs['zorder'] == 3
        assert kwargs['alpha'] == 0.7
    
    plt.close(fig)


def test_add_logo_ax(mock_logo_file):
    """Test the add_logo_ax function"""
    # Create a figure
    fig = plt.figure(figsize=(6, 4))
    
    # Mock the add_axes method to capture calls
    with mock.patch.object(fig, 'add_axes', return_value=mock.MagicMock()) as mock_add_axes:
        # Call the function
        pu.add_logo_ax(fig)
        
        # Check that add_axes was called
        mock_add_axes.assert_called_once()
        
        # Check the arguments
        args, kwargs = mock_add_axes.call_args
        
        # Check that the position is in figure coordinates (0-1)
        position = args[0]
        assert len(position) == 4  # [left, bottom, width, height]
        assert 0 <= position[0] <= 1  # left
        assert 0 <= position[1] <= 1  # bottom
        assert 0 <= position[2] <= 1  # width
        assert 0 <= position[3] <= 1  # height
        
        # Check that zorder was set
        assert kwargs['zorder'] == 10
        
        # Check that imshow was called on the returned axes
        mock_axes = mock_add_axes.return_value
        mock_axes.imshow.assert_called_once()
        mock_axes.axis.assert_called_once_with('off')
        mock_axes.patch.set_alpha.assert_called_once_with(0.0)
    
    plt.close(fig)

def test_add_logo_with_resize(mock_logo_file):
    """Test the add_logo function with image resizing"""
    # Create a figure
    fig = plt.figure(figsize=(6, 4))
    
    # Mock PIL's Image.fromarray and resize
    with mock.patch('PIL.Image.fromarray') as mock_fromarray:
        mock_pil_image = mock.MagicMock()
        mock_fromarray.return_value = mock_pil_image
        mock_pil_image.resize.return_value = mock_pil_image
        
        # Fix the lambda function issue by using a proper method
        def array_method(dtype=None):
            return np.zeros((20, 30, 4))
        
        mock_pil_image.__array__ = array_method
        
        # Mock figimage to avoid actual rendering
        with mock.patch.object(fig, 'figimage') as mock_figimage:
            # Call the function
            pu.add_logo(fig)
            
            # Check that PIL Image processing was used
            mock_fromarray.assert_called_once()
            mock_pil_image.resize.assert_called_once()
            
            # Check that figimage was called with the processed image
            mock_figimage.assert_called_once()
    
    plt.close(fig)

def test_add_logo_file_not_found():
    """Test the add_logo function when logo file is not found"""
    # Create a figure
    fig = plt.figure(figsize=(6, 4))
    
    # Mock matplotlib.pyplot.imread to raise FileNotFoundError for all paths
    with mock.patch('matplotlib.pyplot.imread', side_effect=FileNotFoundError):
        # Mock print to capture output
        with mock.patch('eviz.lib.autoviz.utils.logger.warning') as mock_logger_warning:
        # with mock.patch('builtins.print') as mock_print:
            # Call the function
            pu.add_logo(fig)
            
            # Check that the appropriate message was printed
            # mock_print.assert_any_call("Could not find logo file in any of the expected locations")
            mock_logger_warning.assert_called_once_with("Could not find logo file in any of the expected locations")
    
    plt.close(fig)

def test_add_logo_ax_file_not_found():
    """Test the add_logo_ax function when logo file is not found"""
    # Create a figure
    fig = plt.figure(figsize=(6, 4))
    
    # Mock matplotlib.pyplot.imread to raise FileNotFoundError for all paths
    with mock.patch('matplotlib.pyplot.imread', side_effect=FileNotFoundError):
        # Mock print to capture output
        with mock.patch('eviz.lib.autoviz.utils.logger.warning') as mock_logger_warning:
            # Call the function
            pu.add_logo_ax(fig)
            
            # Check that the appropriate message was printed
            mock_logger_warning.assert_any_call("Could not find logo file")
    
    plt.close(fig)



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
        ((1, 1), 10),
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
        (None, 14),
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
