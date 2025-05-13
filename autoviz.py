import sys
import time
import subprocess
import argparse

from eviz.lib.utils import logger_setup, timer
from eviz.lib.autoviz.autoviz_base import Autoviz


def parse_command_line() -> argparse.Namespace:
    """Parse command line arguments.

    Example:

        >>> python autoviz.py -s generic
        >>> python autoviz.py -s generic -c /path/to/config
        >>> python autoviz.py -s generic -f /path/to/config/my_config.yaml
        >>> python autoviz.py -h

    Note:
        The first case requires that the EVIZ_CONFIG_PATH environment variable be defined.
        Setting EVIZ_CONFIG_PATH is simply another way to specify the location of the config files.

    Returns:
        parser: populated namespace containing parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Arguments being passed')

    parser.add_argument('-s', '--sources', type=str, nargs='+', required=False,
                        help='Sources of data input')
    parser.add_argument('--compare', '-m', action='store_true',
                        help='Perform comparison for specified exp_name(s)')
    parser.add_argument('--file', nargs='+', required=False, default=None,
                        help='Enter the path of file to be processed, default=None')
    parser.add_argument('--vars', nargs='+', required=False, default=None,
                        help='Enter a space-separated list of variables to be processed, default=None')
    parser.add_argument('--configfile', '-f', nargs='+', required=False, default=None,
                        help='Enter the full config file path, default=None')
    parser.add_argument('--config', '-c', nargs='+', required=False, default=None,
                        help='Enter the directory wherein YAML specifications can be found, default=None')
    parser.add_argument('--verbose', '-v', nargs='+', required=False, default=1,
                        help='Set logging verbosity to DEBUG (2) or ERROR(0), default=1 (INFO)')
    parser.add_argument('--log', '-l', nargs='+', required=False, default=1,
                        help='Create LOG file (Eviz.LOG)')
    parser.add_argument('--integrate', action='store_true',
                        help='Integrate data from multiple files')
    parser.add_argument('--composite', nargs='+', required=False, default=None,
                        help='Create a composite field from multiple variables (format: field1,field2,operation)')

    args = parser.parse_args()

    if not args.file and not args.sources:
        parser.error("The --sources argument is required unless --file is specified.")

    return args


def main():
    """ Main driver for the autoviz plotting tool """
    start_time = time.time()
    args = parse_command_line()

    if args.file and args.vars:
        subprocess.run(['python',
                        'metadump.py', args.file[0],
                        '--vars', *args.vars])
        sys.exit()
    else:
        if args.file:
            subprocess.run(['python',
                            'metadump.py', args.file[0]])
            sys.exit()

    verbose = int(args.verbose[0] if isinstance(args.verbose, list) else '1')
    log = int(args.log[0] if isinstance(args.log, list) else '1')
    logger_setup('autoviz', log=log, verbose=verbose)

    input_sources = [s.strip() for s in args.sources[0].split(',')]

    autoviz = Autoviz(input_sources, args=args)
    autoviz.run()
    print(f"Time taken = {timer(start_time, time.time())}")


if __name__ == "__main__":
    main()
