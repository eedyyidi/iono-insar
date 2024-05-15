#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Wang Yidi, May 2024                              #
############################################################


import os
import sys

from mintpy.defaults.template import get_template_content
from mintpy.utils.arg_utils import create_argument_parser

# key configuration parameter name
key_prefix = 'mintpy.ionosphericDelay.'

###############################################################
TEMPLATE = get_template_content('correct_ionosphere')

REFERENCE = """reference:
  Yunjun, Z., Fattahi, H., Pi, X., Rosen, P., Simons, M., Agram, P., & Aoki, Y. (2022). Range Geolocation Accuracy of C-/L-band SAR and its Implications for Operational Stack Coregistration. IEEE Transactions on Geoscience and Remote Sensing, 60, 5227219.
"""

EXAMPLE = """example:
"""


def create_parser(subparsers=None):
    synopsis = 'Ionospheric correction using GIM (from ISCE-2 stack processing)'
    epilog = REFERENCE + '\n' + TEMPLATE + '\n' + EXAMPLE
    name = __name__.split('.')[-1]
    parser = create_argument_parser(
        name, synopsis=synopsis, description=synopsis, epilog=epilog, subparsers=subparsers)

    # inputs
    parser.add_argument('-t', '--template', dest='template_file', required=True,
                        help='template file with ionospheric delay options.')
    parser.add_argument('-f', '--file', dest='dis_file',
                        help='time-series HDF5 file to be corrected, e.g. timeseries.h5')
    parser.add_argument('--tec_dir', dest='tec_dir', default='~/data/aux/IONEX',
                        help='path of IONEX, e.g. ~/data/aux/IONEX')
    parser.add_argument('--geo_file', dest='geo_file', default='./inputs/geometryRadar.h5',
                        help='geometryRadar file, e.g. ../inputs/geometryRadar.h5')

    # outputs
    parser.add_argument('--iono-file', dest='iono_file', default='ion.h5',
                        help='output ionospheric delay time series file (default: %(default)s).')
    parser.add_argument('-o', '--output', dest='cor_dis_file',
                        help='output corrected time-series file, e.g. timeseries_ion.h5')
    return parser


def cmd_line_parse(iargs=None):
    """Command line parser."""
    # parse
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    # default: use absolute path for all files
    inps.template_file    = os.path.abspath(inps.template_file)
    inps.dis_file       = os.path.abspath(inps.dis_file)
    inps.tec_dir       = os.path.expanduser(inps.tec_dir)
    inps.geo_file          = os.path.abspath(inps.geo_file)
    inps.iono_file   = os.path.abspath(inps.iono_file)
    inps.cor_dis_file     = os.path.abspath(inps.cor_dis_file)
    return inps



###############################################################
def main(iargs=None):
    # parse
    inps = cmd_line_parse(iargs)

    # import
    from mintpy.iono_GIM import run_iono_GIM

    # run
    run_iono_GIM(inps)

###############################################################
if __name__ == '__main__':
    main(sys.argv[1:])