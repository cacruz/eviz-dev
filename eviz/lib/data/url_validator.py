import logging
import re
from urllib.parse import urlparse


def is_url(path):
    """
    Check if a path is a URL.
    
    Args:
        path (str): The path to check
        
    Returns:
        bool: True if the path is a URL, False otherwise
    """
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except:
        return False


def is_opendap_url(url):
    """
    Check if a URL is an OpenDAP endpoint.
    
    Args:
        url (str): The URL to check
        
    Returns:
        bool: True if the URL is an OpenDAP endpoint, False otherwise
    """
    if not is_url(url):
        return False

    # Common OpenDAP URL patterns
    opendap_patterns = [
        r'thredds/dodsC',  # THREDDS Data Server
        r'opendap',  # Generic OpenDAP
        r'dods',  # Generic DODS
        r'dap',  # Generic DAP
        r'\.nc$',  # NetCDF file extension
        r'\.nc\?',  # NetCDF with query parameters
        r'\.nc#',  # NetCDF with fragment
    ]

    for pattern in opendap_patterns:
        if re.search(pattern, url, re.IGNORECASE):
            return True

    return False


def get_logger():
    return logging.getLogger(__name__)
