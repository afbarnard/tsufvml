"""Entry points for scripts and APIs for foreign code"""

# Copyright (c) 2018 Aubrey Barnard.  This is free software released
# under the MIT License.  See `LICENSE.txt` for details.


import argparse
import os.path
import sys

import tsufvml


def run_decision_trees_api(
        data_matrix_filename,
        feature_table_filename=None,
        feature_blacklist_filename=None,
        concept_table_filename=None,
        output=sys.stdout
):
    pass


def decision_tree(prog_name, *args):
    arg_prsr = argparse.ArgumentParser(
        prog = prog_name,
        description = "Model feature vector data with decision trees.",
    )
    arg_prsr.add_argument(
        '--version',
        action='version',
        version='tsufvml {}'.format(tsufvml.__version__),
    )
    arg_prsr.add_argument(
        'data',
        type=argparse.FileType('rt'),
        help='Feature vector data in SVMLight format',
    )
    arg_prsr.add_argument(
        '--features',
        type=argparse.FileType('rt'),
        metavar='TABLE',
        help='Table of features in delimited format',
    )
    arg_prsr.add_argument(
        '--concepts',
        type=argparse.FileType('rt'),
        metavar='TABLE',
        help='Table of concepts in delimited format',
    )
    env = arg_prsr.parse_args(args)
    from pprint import pprint; pprint(env)


def decision_tree_main():
    decision_tree(os.path.basename(sys.argv[0]), *sys.argv[1:])
