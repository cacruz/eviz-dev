import logging
import os
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import xarray as xr

from eviz.lib.data.pipeline import DataProcessor
from eviz.models.esm.gridded import Gridded
from eviz.lib.data.utils import apply_conversion
from eviz.lib.data.utils import apply_mean

warnings.filterwarnings("ignore")


@dataclass()
class Geos(Gridded):
    """ Define GEOS-specific model data and functions.
    The GEOS-specific functionality centers around the HISTORY.rc file which contains
    information about the GEOS data sources. In Eviz, the HISTORY.rc is parsed during
    initialization and the relevant information is stored in the Config object.

    Parameters:
        config (Config) : Config object associated with this model
    """
    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__name__)
    
    def __post_init__(self):
        self.logger.info("Start init")
        self.season = None
        super().__post_init__()

def parse_history(hist_fname):
    """ Parse a GEOS HISTORY.rc file.

    Args:
        hist_fname (str) : Name of HISTORY.rc file

    Returns:
        A dictionary containing the following keys:
         - EXPID
         - EXPDSC
         - EXPSRC
         - COLLECTIONS
      The value associated with the COLLECTIONS key is itself a dictionary of all the
      collections produced by HISTORY. Each collection is also a dictionary of the
      settings (template, mode, resolution, fields, etc.) associated with that collection.
    """

    with open(hist_fname) as fid:
        lines = fid.readlines()

    history_dict = dict()

    for line in lines:
        if 'EXPID:' in line:
            history_dict['EXPID'] = line.split()[1].strip()
            break

    for line in lines:
        if 'EXPDSC:' in line:
            history_dict['EXPDSC'] = line.split()[1].strip()
            break

    for line in lines:
        if 'EXPSRC:' in line:
            history_dict['EXPSRC'] = line.split()[1].strip()
            break

    # Obtain the line number having 'COLLECTIONS:'
    icol = 0
    for line in lines:
        if 'COLLECTIONS:' in line:
            break
        icol += 1

    # Obtain the line (number) ending the list of collections
    jcol = icol + 1
    for line in lines[icol + 1:]:
        if '::' in line:
            break
        jcol += 1

    # Loop over the lines to update the list of collections
    list_collections = list()

    for line in lines[icol + 1:jcol]:
        if '#' not in line.strip():
            list_collections.append(line.strip()[1:-1])

    dict_collections = dict()

    for col in list_collections:
        dict_collections[col] = get_collection(col, lines)

    history_dict['COLLECTIONS'] = dict_collections

    return history_dict


def get_collection(col_name, lines):
    """
       Given a collection name and all the lines (as a list) in the HISTORY.rc file,
       parses the lines to create a dictionary of the settings associated with the
       collection.

       Args:
          col_name: (str) name of a collection
          lines: (list) list of lines in the HISTORY.rc file

       Returns:
          - Dictionary of all the settings for the collection.
    """

    col_dict = dict()

    # Loop over the lines to extract settings other than fields
    ic_min = 0
    coll_name = col_name + '.'
    for line in lines:
        if coll_name in line:
            # split the line to extract a variable and its value
            var, val = line.strip().split(':')
            if coll_name + 'fields' == var:
                break
            else:
                var = var.split('.')[-1]
                val = val.strip()
                if val[0].isdigit() or val[0] == '>':
                    col_dict[var] = val.split(",")[0].strip()
                else:
                    col_dict[var] = val[1:].split("'")[0]
        ic_min += 1

    list_fields = list()

    # Obtain the line (number) ending the list of fields
    ic_max = ic_min + 1
    for line in lines[ic_min + 1:]:
        if '::' in line:
            break
        ic_max += 1

    # Loop over the lines to update the list of fields

    # We first need to take care of the first line
    split_line = lines[ic_min].split(',')
    field_name = split_line[2].strip()
    if field_name:
        list_fields.append(field_name[1:-1])
    else:
        list_fields.append(split_line[0].split()[1].strip()[1:-1])

    for line in lines[ic_min + 1:ic_max]:
        new_line = line.strip()
        if new_line[0] == '#':
            continue
        split_line = new_line.split(',')
        field_name = split_line[2].strip()

        if field_name:
            list_fields.append(field_name[1:-1])
        else:
            list_fields.append(split_line[0].strip()[1:-1])

    col_dict['fields'] = list_fields

    return col_dict


def get_collection_with_field(myhistory_dict, field_name):
    """
      List all the collections in HISTORY that contains the provided field name.
    """
    d = myhistory_dict['COLLECTIONS']
    list_collections = [key for key in d if field_name in d[key]['fields']]

    return list_collections


def is_collection_time_averaged(myhistory_dict, col_name):
    """
      Check if a collection is time-averaged
    """
    col_mode = myhistory_dict['COLLECTIONS'][col_name]['mode']
    return col_mode.strip() == 'time-averaged'
