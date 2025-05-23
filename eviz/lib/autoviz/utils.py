import decimal
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import multiprocessing
import logging
import time


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatter
import glob
from PIL import Image

from numpy import e
from matplotlib.offsetbox import (OffsetImage, TextArea, AnchoredOffsetbox, VPacker)
import eviz.lib.utils as u
from eviz.lib.utils import timer

logger = logging.getLogger('plot_utils')

UNIT_REGEX = re.compile(
    r'\A([-+]?[0-9._]+(?:[eE][-+]?[0-9_]+)?)(.*)\Z'  # float with trailing units
)
UNIT_DICT = {
    'in': 1.0,
    'ft': 12.0,
    'yd': 36.0,
    'm': 39.37,
    'dm': 3.937,
    'cm': 0.3937,
    'mm': 0.03937,
    'pc': 1 / 6.0,
    'pt': 1 / 72.0,
    'ly': 3.725e17,
}


def subproc(cmd):
    name = multiprocessing.current_process().name
    logger.debug(f'Starting {name} ')
    cmds = shlex.split(cmd)
    p = subprocess.Popen(cmds,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         universal_newlines=True)
    out, err = p.communicate()
    logger.debug(f"{name} Out:\n {out}")
    logger.debug(f"{name} Err:\n {err}")
    logger.debug(f'Exiting {name}')
    return out


def plot_process(filename):
    name = multiprocessing.current_process().name
    logger.info(f'Starting {name} ')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')


def run_plot_commands(filenames):
    njobs = len(filenames)
    logger.info(f"Processing {njobs} jobs - please wait ...")
    procs = list()
    for i in range(njobs):
        p = multiprocessing.Process(target=plot_process, args=(filenames[i],))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()


def create_pdf(config):
    from PIL import Image
    import glob
    irgb0 = None

    img_files = sorted(
        glob.glob(config.output_dir + '/*.' + config.print_format))
    pdf_file = 'eviz_plots.pdf'
    ilist = list()
    cnt = 0
    for im in img_files:
        if cnt == 0:
            i0 = Image.open(im)
            irgb0 = i0.convert('RGB')
        else:
            i = Image.open(im)
            irgb = i.convert('RGB')
            ilist.append(irgb)
        cnt += 1

    if cnt == 1 and irgb0:
        irgb0.save(config.output_dir + '/' + pdf_file)
    elif irgb0:
        irgb0.save(config.output_dir + '/' + pdf_file,
                   save_all=True, append_images=ilist)


def natural_key(filename):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]


def update(i, im, image_array):
    im.set_array(image_array[i])
    return im,


def create_gif(config):
    if config.archive_web_results:
        img_path = os.path.join(config.app_data.outputs['output_dir'],
                                config.paths.archive_path)
    else:
        img_path = config.app_data.outputs['output_dir']

    all_files = glob.glob(img_path+"/*."+config.print_format)
    files = sorted(all_files, key=natural_key)
    if len(files) == 1:
        return
    prefix = list(config.app_data.inputs[0]['to_plot'])[0]
    
    # remove IC (NUWRF only)
    if not config.archive_web_results:
        if {'lis', 'wrf'} & set(config.source_names):
            # Find the file that ends with "_0_0.png" instead of assuming exact name
            ic_file_pattern = f"*{prefix}*_0_0.{config.print_format}"
            ic_files = glob.glob(os.path.join(img_path, ic_file_pattern))
            
            if ic_files:
                # Remove the first matching IC file
                ic_file_to_remove = ic_files[0]
                logger.debug(f"Removing IC file: {ic_file_to_remove}")
                os.remove(ic_file_to_remove)
                
                # Remove it from the files list as well
                if ic_file_to_remove in files:
                    files.remove(ic_file_to_remove)
                else:
                    # If not found by exact match, find by basename
                    ic_basename = os.path.basename(ic_file_to_remove)
                    files = [f for f in files if os.path.basename(f) != ic_basename]
            else:
                logger.warning(f"Warning: No IC file found matching pattern {ic_file_pattern}")

    if not files:
        logger.error("No files remaining after IC removal")
        return

    image_array = []
    for my_file in files:
        image = Image.open(my_file)
        image_array.append(np.array(image))

    if not image_array:
        logger.error("No images to create GIF")
        return

    height, width, _ = image_array[0].shape
    fig, ax = plt.subplots(figsize=(width / 100, height / 100),
                           dpi=300)  # dpi here must be the same as in print_map()
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding

    fps = config.gif_fps
    duration_ms = int(1000 / fps)
    image_sequence = [Image.fromarray(img) for img in image_array]
    
    # Create GIF filename - use just the field name for cleaner naming
    gif_filename = f"{prefix}.gif"
    gif_path = os.path.join(img_path, gif_filename)
    
    image_sequence[0].save(
        gif_path,
        save_all=True,
        append_images=image_sequence[1:],
        duration=duration_ms,
        loop=0  # Infinite loop
    )
    
    logger.info(f"Created GIF: {gif_path}")

    if config.archive_web_results:
        json_filename = f"{prefix}.json"
        json_path = os.path.join(img_path, json_filename)
        with open(json_path, 'w') as fp:
            json.dump(config.vis_summary, fp)
            fp.close()

    # Clean up individual PNG files
    for my_file in files:
        try:
            os.remove(my_file)
            logger.debug(f"Removed: {os.path.basename(my_file)}")
        except OSError as e:
            logger.warning(f"Warning: Could not remove {my_file}: {e}")


def print_map(
    config,
    plot_type: str,
    findex: int,
    fig,
    level: int = None, 
) -> None:
    """Save or display a plot, handling output directory, file naming, and optional archiving.

    Args:
        config: Configuration object with plotting and output options.
        plot_type (str): Type of plot (e.g., 'xy', 'yz', etc.).
        findex (int): File index for naming.
        fig: Matplotlib figure object to save or show.
        level (int, optional): Vertical level for the plot, if applicable.
    """
    def resolve_output_dir(config) -> str:
        """Determine and create the output directory if needed."""
        map_params = config.map_params
        output_dir = u.get_nested_key_value(map_params[config.pindex], ['outputs', 'output_dir'])
        if not output_dir:
            output_dir = config.paths.output_path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.debug(f"Created output directory: {output_dir}")
        return output_dir

    def build_filename(config, plot_type: str, findex: int, level: int = None) -> str:
        """Construct the output filename based on config and plot type."""
        map_params = config.map_params
        field_name = map_params[config.pindex]['field']
        exp_id = map_params[config.pindex].get('exp_id', None)

        levstr = f"_{level}" if level is not None else ""
        time_level = getattr(config, "time_level", "")
        exp_id_suf = "."

        if not config.compare:
            if exp_id:
                exp_id_suf = f"_{exp_id}_{findex}_{time_level}."
            else:
                exp_id_suf = f"_{findex}_{time_level}."
        # else: exp_id_suf remains "."

        if 'xy' in plot_type:
            fname = f"{field_name}{levstr}{exp_id_suf}"
        elif 'yz' in plot_type:
            fname = f"{field_name}_yz{exp_id_suf}"
        else:
            fname = f"{field_name}_{plot_type}{exp_id_suf}"

        return fname

    output_dir = resolve_output_dir(config)
    fname = build_filename(config, plot_type, findex, level)
    map_filename = f"{fname}{config.print_format}"
    filename = os.path.join(output_dir, map_filename)

    if config.print_to_file:
        fig.tight_layout()
        # Save with or without bbox_inches depending on extent
        if config.ax_opts.get('extent'):
            fig.savefig(filename, dpi=300)
        else:
            fig.savefig(filename, bbox_inches='tight', dpi=300)

        logger.debug(f"Figure saved to {filename}")

        if getattr(config, "archive_web_results", False):
            # Remove file extension from fname for JSON
            json_fname = fname.split('.')[0]
            dump_json_file(
                json_fname, config, plot_type, findex, map_filename, fig, output_dir
            )
            logger.info(f"Archived web results for {json_fname}")
    else:
        plt.tight_layout()
        plt.show()
    logger.debug("Clearing figure")




def formatted_contours(clevs):
    new_clevs = []
    for lev in clevs:
        str_lev = str(lev)
        if "." in str_lev:
            rhs = str_lev.split(".")[1]
            if "e" in rhs or "E" in rhs:
                new_clevs.append(lev)
            elif int(rhs) == 0:
                new_clevs.append(int(float(str_lev)))
            else:
                new_clevs.append(lev)
        else:
            new_clevs.append(lev)
    return new_clevs


def axis_tick_font_size(panels=None):
    if panels == (1, 1):  # single image on a page
        font_size = 12
    elif panels == (3, 1):
        font_size = 12
    elif panels == (2, 2):
        font_size = 12
    else:
        font_size = 8
    return font_size


def bar_font_size(panels=None):
    if panels == (1, 1):  # single image on a page
        font_size = 12
    elif panels == (3, 1):
        font_size = 10
    elif panels == (2, 2):
        font_size = 10
    else:
        font_size = 8
    return font_size


def cbar_shrink(panels=None):
    if panels == (1, 1):  # single image on a page
        frac = 1.0
    elif panels == (3, 1):
        frac = 0.75
    elif panels == (2, 2):
        frac = 0.75
    else:
        frac = 0.5
    return frac


def contour_tick_font_size(panels):
    if panels == (1, 1):  # single image on a page
        font_size = 10
    elif panels == (3, 1):
        font_size = 8
    elif panels == (2, 2):
        font_size = 8
    else:
        font_size = 8
    return font_size


def axes_label_font_size(panels=None):
    if panels == (1, 1):  # single image on a page
        font_size = 12
    elif panels == (3, 1):
        font_size = 10
    elif panels == (2, 2):
        font_size = 10
    else:
        font_size = 8
    return font_size


def cbar_pad(panels=None):
    """
    Fraction of original axes between colorbar and new image axes
    """
    if panels == (1, 1):  # single image on a page
        pad = 0.05
    elif panels == (3, 1):
        pad = 0.03
    elif panels == (2, 2):
        pad = 0.05
    else:
        pad = 0.05
    return pad


def cbar_fraction(panels=None):
    """ Fraction of original axes to use for colorbar """
    if panels == (1, 1):  # single image on a page
        fraction = 0.05
    elif panels == (3, 1):
        fraction = 0.1
    elif panels == (2, 2):
        fraction = 0.05
    else:
        fraction = 0.05
    return fraction


def image_font_size(panels=None):
    if panels == (1, 1):  # single image on a page
        font_size = 16
    elif panels == (3, 1):
        font_size = 14
    elif panels == (2, 2):
        font_size = 14
    else:
        font_size = 'small'
    return font_size


def subplot_title_font_size(panels=None):
    if panels == (1, 1):  # single image on a page
        font_size = 14
    elif panels == (3, 1):
        font_size = 12
    elif panels == (2, 2):
        font_size = 12
    else:
        font_size = 10
    return font_size


def title_font_size(panels=None):
    if panels == (1, 1):  # single image on a page
        font_size = 14
    elif panels == (3, 1):
        font_size = 12
    elif panels == (2, 2):
        font_size = 12
    else:
        font_size = 12
    return font_size


def contour_label_size(panels=None):
    if panels == (1, 1):  # single image on a page
        label_size = 8
    elif panels == (3, 1):
        label_size = 8
    elif panels == (2, 2):
        label_size = 8
    else:
        label_size = 8
    return label_size


def contour_levels_plot(clevs):
    new_clevs = []
    for lev in clevs:
        clevs_string = str(lev)
        if "e" in clevs_string:
            # print ("Digits are in scientific notation")
            new_clevs.append(lev)
        else:
            if "." not in clevs_string:
                new_clevs.append(int(lev))
            else:
                digits = clevs_string.split('.')[1]  # just get RHS of number

                if int(digits) == 0:
                    #            print ("The level: ", lev, " has no RHS!", int(lev))
                    new_clevs.append(int(lev))
                else:
                    new_clevs.append(lev)
    return new_clevs


def contour_format_from_levels(levels, scale=None):
    digits_list = []
    num_sci_format = 0
    for lev in levels:  # check each contour level
        clevs_string = str(lev)
        if "e" in clevs_string or "E" in clevs_string:  # sci notation
            num_sci_format = num_sci_format + 1
            if "e" in clevs_string or "E" in clevs_string:
                pres = abs(int(clevs_string.split('e')[1]))
                if "E" in str(pres):
                    pres = abs(int(clevs_string.split('E')[1]))
                number = decimal.Decimal(lev)
                clevs_string = str(round(number, pres + 2))
                digits1 = clevs_string.split('.')[1]  # just get RHS of number
                if "E" in str(digits1):
                    digits = digits1.split('E')[0]
                    digits_list.append(len(digits))
                elif "e" in str(digits1):
                    digits = digits1.split('e')[0]
                    digits_list.append(len(digits))
                else:
                    digits_list.append(len(digits1))
        elif "." not in clevs_string:  # not floating point
            digits_list.append(0)
        else:
            digits_list.append(len(clevs_string.split('.')[1]))  # just get RHS of number
    digits_list.sort()
    num_type = "f"
    if num_sci_format > 1:
        num_type = "e"
    if digits_list[-1] == 0:
        contour_format = "%d"
    elif digits_list[-1] <= 3:
        contour_format = "%1." + str(digits_list[-1]) + num_type
    else:
        contour_format = "%1.1e"
    if scale:
        contour_format = LogFormatter(base=e, labelOnlyBase=False)

    return contour_format


def fmt_two_digits(x, pos):
    return f'[{x:.2f}]'


def fmt(x, pos):
    """
    Format color bar labels to show scientific label
    """
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


def fmt_once(x, pos):
    """
    Format color bar labels to show scientific label but not the x10^x
    """
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${}$'.format(a)


def revise_tick_labels(cbar):
    # remove unecessary decimals
    count = 0
    labels = [item.get_text() for item in cbar.ax.get_xticklabels()]
    for label in labels:
        str_label = str(label)
        if "." in str_label:
            rhs = str_label.split(".")[1]
            lhs = str_label.split(".")[0]
            if "e" in rhs or "E" in rhs:
                if "e" in rhs:
                    split_sci_not = rhs.split("e")
                else:
                    split_sci_not = rhs.split("E")
                if int(split_sci_not[0]) == 0:
                    new_sci_not = lhs + ".e" + split_sci_not[1]
                    labels[count] = new_sci_not
            elif int(rhs) == 0:
                labels[count] = str_label.split(".")[0]
        count = count + 1
    cbar.ax.set_xticklabels(labels)
    # remove trailing zeros
    count = 0
    labels = [item.get_text() for item in cbar.ax.get_xticklabels()]
    for label in labels:
        str_label = str(label)
        if "e" not in str_label and "E" not in str_label:
            if "." in str_label:
                strip_str_label = str_label.rstrip('0')
                labels[count] = strip_str_label
        count = count + 1
    cbar.ax.set_xticklabels(labels)
    labels = [item.get_text() for item in cbar.ax.get_xticklabels()]
    # labels minus sign is not accepted by float()
    # make it acceptable:
    labels = [x.replace('\U00002212', '-') for x in labels]
    if float(labels[0]) == float(0):
        labels[0] = "0"
    cbar.ax.set_xticklabels(labels)


def colorbar(mappable):
    """
    Create a colorbar that works with both standard Matplotlib Axes and Cartopy GeoAxes.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from cartopy.mpl.geoaxes import GeoAxes

    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure

    # Check if the axes is a GeoAxes
    if isinstance(ax, GeoAxes):
        # Create a new axes for the colorbar with the same projection
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=type(ax), projection=ax.projection)
    else:
        # Standard Matplotlib Axes
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

    # Create the colorbar
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


def image_scaling(image, num_rows, num_cols):
    """
    Scale the image to desired length(num_rows) and width (num_cols)
    """
    number_rows = len(image)  # source number of rows
    number_columns = len(image[0])  # source number of columns
    return [[image[int(number_rows * r / num_rows)][int(number_columns * c / num_cols)]
             for c in range(num_cols)] for r in range(num_rows)]


def add_logo_xy(logo, ax, x0, y0, scale=50):
    """
    adds image logo and positions it on the figure
    at position x0, y0
    """
    # scale Image
    logo = image_scaling(logo, scale, scale)
    ax.figure.figimage(logo, x0, y0, alpha=1.0, zorder=1, origin="upper")


def add_logo_anchor(ax, logo, label=None, logo_loc='upper left', alpha=0.5):
    """
    adds image logo and optionally, text
    """
    image_box = OffsetImage(logo, alpha=alpha, zoom=0.05)
    if label:
        textbox = TextArea(label, textprops=dict(alpha=alpha))
        packer = VPacker(children=[image_box, textbox], mode='fixed', pad=0, sep=0, align='center')
        ao = AnchoredOffsetbox(logo_loc, pad=0, borderpad=0, child=packer)
    else:
        ao = AnchoredOffsetbox(logo_loc, pad=0.01, borderpad=0, child=image_box)
        ao.patch.set_alpha(0)
    ax.add_artist(ao)


def add_logo_fig(fig, logo):
    """
    adds image logo to a figure
    """
    # (x, y, width, height)
    imax = fig.add_axes([0.9, 0.9, 0.1, 0.1])
    # remove ticks & the box from imax
    imax.set_axis_off()
    # print the logo with aspect="equal" to avoid distorting the logo
    imax.imshow(logo, aspect="equal", alpha=1)


def add_logo(ax, logo):
    """
    adds image logo to axes
    """
    ax.figure.figimage(logo, 1, 1, zorder=3, alpha=.5)


def output_basic(config, name):
    if config.print_to_file:
        output_fname = name + "." + config.print_format
        output_dir = u.get_nested_key_value(config.map_params[config.pindex], ['outputs', 'output_dir'])
        if not output_dir:
            output_dir = config.paths.output_path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(output_dir, output_fname)
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()


def get_subplot_geometry(axes):
    ss = axes.get_subplotspec()
    geom = ss.get_geometry()[0:2]
    return geom, int(ss.is_first_row()), int(ss.is_first_col()), int(ss.is_last_row()), int(ss.is_last_col())


def get_subplot_shape(n):
    if n <= 3:
        return n, 1
    
    # Try to make the layout as square as possible
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        if rows * cols >= n and abs(rows - cols) <= 1:
            return rows, cols

    # Fallback to 1 row if no better fit (shouldn't happen with 2 <= n <= 12)
    return 1, n


# def dump_json_file(config, plot_type, findex, map_filename, fig, output_dir):
def dump_json_file(fname, config, plot_type, findex, map_filename, fig, output_dir):
    vis_summary = {}
    source_name = config.source_names[config.ds_index]
    # event_stamp = source_name + '_web'

    map_params = config.map_params
    exp_name = map_params[config.pindex]['exp_name']
    field_name = map_params[config.pindex]['field']
    figure = fig
    vis_summary[findex] = {}
    if exp_name:
        vis_summary[findex]['title'] = exp_name
    else:
        axes = figure.get_axes()
        title = axes[0].get_title(loc='left')
        vis_summary[findex]['title'] = title
    vis_summary[findex]['model'] = source_name
    vis_summary[findex]['config_name'] = map_params[config.pindex]['filename']
    vis_summary[findex]['plot_type'] = plot_type
    # vis_summary[findex]['level'] = config.level
    if hasattr(config, 'level'):
        vis_summary[findex]['level'] = config.level
    if not hasattr(config, 'level'):
        vis_summary[findex]['level'] = 'Surface'
    # if config.level:
    #     vis_summary[findex]['level'] = config.level

    if config.make_gif:  # one file per field_name
        vis_summary[findex]['filename'] = field_name+".gif"
    else:
        vis_summary[findex]['filename'] = map_filename
    vis_summary[findex]['field_name'] = field_name

    if not os.path.exists(os.path.join(output_dir, config.event_stamp)):
        os.makedirs(os.path.join(output_dir, config.event_stamp))

    summary_path = os.path.join(output_dir, config.event_stamp, fname + '.json')
    vis_summary['time_now'] = time.time()
    # vis_summary['log'] = load_log()
    vis_summary['input_files'] = config.file_list
    vis_summary['time_exec'] = timer(config.start_time, time.time())
    vis_summary['output_findex'] = findex

    archive(config, output_dir, config.event_stamp)
    config.summary_path = summary_path
    config.vis_summary = vis_summary

    if config.make_gif:
        return

    with open(summary_path, 'w') as fp:
        json.dump(vis_summary, fp)
        fp.close()


def load_log():
    """ Retrieve Eviz.LOG used by streamlit """
    with open('Eviz.LOG') as fp:
        lines = fp.readlines()
        fp.close()

    return lines


def archive(config, output_dir, event_stamp):
    """ Archive data for web results

    Parameters:
        output_dir (str) : Output directory to store images
        event_stamp (str) : Time stamp for archived web results
    """
    fs = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    full_fs = [os.path.join(output_dir, f) for f in fs]
    archive_path = os.path.join(output_dir, event_stamp)
    config.archive_path = archive_path
    config.full_fs = full_fs

    if not os.path.exists(archive_path):
        os.mkdir(archive_path)

    # Added for streamlit viewing
    for i, f in enumerate(full_fs):
        if not os.path.exists(os.path.join(archive_path, fs[i])):
            shutil.move(f, archive_path)
        else:
            os.remove(f)


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, prec=0, offset=True, math_text=True):
        self.prec = prec
        self.oom = order
        self.fformat = "%1."+str(self.prec)+"f"
        matplotlib.ticker.ScalarFormatter.__init__(self, useOffset=offset, useMathText=math_text)

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom

    def _set_format(self):
        self.format = self.fformat
        if self._useMathText:
             self.format = r'$\mathdefault{%s}$' % self.format

    def _custom_format(self, value, tick_number):
        return f'{value:.2f}'


class FlexibleOOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, min_val=None, max_val=None, offset=True, math_text=True):
        self.oom = order
        self.min_val = min_val
        self.max_val = max_val
        matplotlib.ticker.ScalarFormatter.__init__(self, useOffset=offset, useMathText=math_text)
        # set oom explicitly
        self._set_order_of_magnitude()

    def _set_order_of_magnitude(self):
        # set oom dynamically based on the min/max values of the data
        if self.min_val is not None and self.max_val is not None:
            if self.min_val == self.max_val:  # avoid log10 issues
                self.oom = 0
            else:
                self.oom = np.floor(np.log10(np.abs(self.max_val))).astype(int)

    def _set_format(self):
        if self.oom != 0:
            self.format = "%1.2f"
            # self.format = r'$10^{%d}$' % self.oom
        else:
            self.format = "%1.2f"
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format

    def _custom_format(self, value, tick_number):
        if value != 0:
            exp = int(np.floor(np.log10(np.abs(value))))
            coeff = value / (10**exp)
            # return r'$%1.2f \times 10^{%d}$' % (coeff, exp)
            return r'$%1.2f$' % coeff
        else:
            return '0'

    def __call__(self, x, pos=None):
        return self._custom_format(x, pos)
