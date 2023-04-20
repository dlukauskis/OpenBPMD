#!/usr/bin/env python
# coding: utf-8
descriptn = \
    """
    Analyses OpenBPMD simulations, makes time-resolved plots of the three 
    scores and calculates the final score for the ligand pose given.

    Writes its outputs to '{dir}_results'.
    """

import os
import argparse
from analysis import *

# args parse bit
# Parse the CLI arguments
parser = argparse.ArgumentParser(
       formatter_class=argparse.RawDescriptionHelpFormatter,
       description=descriptn)

parser.add_argument("-i", "--dir", type=str,
                    help='directory where OpenBPMD simulations were run')
parser.add_argument("--v", action='store_true', default=False,
                    help='Be verbose (default: %(default)s)')

args = parser.parse_args()

input_dir = args.dir
res_dir = input_dir + '_results'  # save results here
if args.v:
    print(f"Reading '{input_dir}', writting to '{res_dir}/'")
if not os.path.isdir(res_dir):
    os.mkdir(res_dir)

if args.v:
    print('Making time-resolved score plots')
plot_all_reps('stable_pose',
              save_fig=os.path.join(res_dir,f'plots.png'))

if args.v:
    print('Collating the results from all repeats into a final score')
collect_results(res_dir, res_dir)

if args.v:
    print('Done')

