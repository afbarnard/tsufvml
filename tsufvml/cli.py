"""Entry points for scripts and APIs for foreign code"""

# Copyright (c) 2018-2019 Aubrey Barnard.
#
# This is free software released under the MIT License.  See
# `LICENSE.txt` for details.


import argparse
import operator
import os.path
import pathlib
import sys

from barnapy import logging
from barnapy import parse

# Use absolute imports so that this file can be used from anywhere.
# Postpone expensive imports (i.e. sklearn) until needed.
import tsufvml
from tsufvml import common


def parse_args_as_dict(*args, key_prefix='--', value_parser=None):
    env = {}
    arg_idx = 0
    while arg_idx < len(args):
        arg = args[arg_idx]
        if key_prefix is not None and not arg.startswith(key_prefix):
            raise argparse.ArgumentError(
                None, 'Unrecognized option: {}  (Must start with {!r})'
                .format(arg, key_prefix))
        key = arg[len(key_prefix):]
        if not key:
            raise argparse.ArgumentError(
                None, 'Empty option name: {}'.format(arg))
        if '=' in key:
            key, value = key.split('=', 1)
        else:
            if arg_idx + 1 >= len(args):
                raise argparse.ArgumentError(
                    None, 'Missing value after option: {}'.format(arg))
            arg_idx += 1
            value = args[arg_idx]
        if value_parser is not None:
            value, err = value_parser(value)
            if err:
                raise argparse.ArgumentError(None, err)
        env[key] = value
        arg_idx += 1
    return env


def run_decision_trees_api(
        data_matrix_filename,
        feature_table_filename=None,
        concept_table_filename=None,
        tree_pdf_filename=None,
        decision_tree_args={},
        output=sys.stdout,
):
    # Do expensive imports
    from tsufvml import ml # sklearn
    import numpy
    # Load the data
    data, labels = common.load_svmlight_as_matrix(data_matrix_filename)
    # Load the feature table
    rm2orig_idxs = {}
    features = {}
    if feature_table_filename is not None:
        features = common.load_feature_table(feature_table_filename)
        # Limit the data to the features
        data, rm2orig_idxs = common.limit_matrix_to_features(
            data, features)
    # Load the concept table
    concepts = {}
    if concept_table_filename is not None:
        concepts = common.load_concept_table(concept_table_filename)
    # Construct the decision tree classifier
    dt_model = ml.tree.DecisionTreeClassifier(**decision_tree_args)
    # Run the decision tree classifier
    final_model, cv_roc_areas, feature_importances, final_roc = (
        ml.run_cv_and_final_model(dt_model, data, labels))
    # Average the feature importances over all folds
    avg_feature_importances = list(
        numpy.array(feature_importances).mean(axis=0))
    # Gather report data
    feature_table_header, feature_table = (
        common.mk_feature_importance_table(
            avg_feature_importances,
            rm2orig_idxs, features, concepts))
    feature_table.sort(key=operator.itemgetter(0), reverse=True)
    # Only report features with positive importances
    feature_table = [row for row in feature_table if row[0] > 0]
    dot_text = ml.render_decision_tree_as_graphviz(final_model)
    dot_text, feature_legend = (
        common.replace_variable_references_with_features(
            dot_text, rm2orig_idxs, features, concepts))
    # Generate report
    common.print_report(
        cv_roc_areas=cv_roc_areas,
        final_model_roc_area=final_roc,
        feature_table=feature_table,
        feature_table_header=feature_table_header,
        limit_n_features=100,
        model_text=dot_text,
        feature_legend=feature_legend,
    )
    # Render tree as PDF if requested
    if tree_pdf_filename is not None:
        render_ok = common.render_dot_as_pdf(dot_text, tree_pdf_filename)
        if not render_ok:
            print(
                """

Warning: Unable to render the decision tree as a PDF using either the
    `pydot` or `graphviz` packages.  If you want PDF rendering, make
    sure one of those packages is installed and try again.

                """.strip(),
                file=sys.stderr,
            )


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
        type=argparse.FileType('rb'),
        help='Feature vector data in SVMLight format',
    )
    arg_prsr.add_argument(
        '--features',
        type=argparse.FileType('rt'),
        metavar='FILE',
        help=(
            'Table of features in delimited format.  Only the features '
            'found in this table will be used for modeling.  '
            'Otherwise, all the features found in the data will be '
            'used.'),
    )
    arg_prsr.add_argument(
        '--concepts',
        type=argparse.FileType('rt'),
        metavar='FILE',
        help=(
            'Table of concepts in delimited format.  If provided, '
            'concepts will be included in the report alongside '
            'matching feature names.'),
    )
    arg_prsr.add_argument(
        '--pdf',
        type=pathlib.Path,
        metavar='FILE',
        help=(
            'Filename for PDF rendering of decision tree.  If you want '
            'a PDF, you must specify a filename with this option.'),
    )
    # Parse regular CLI arguments
    env, extra_args = arg_prsr.parse_known_args(args)
    env = vars(env) # Convert `argparse.Namespace` to dictionary
    # Parse decision tree arguments
    try:
        dt_args = parse_args_as_dict(
            *extra_args,
            key_prefix='--dt.',
            value_parser=parse.atom_err)
    except argparse.ArgumentError as e:
        arg_prsr.error(str(e))
    # Start!
    logging.default_config()
    run_decision_trees_api(
        data_matrix_filename=env.get('data'),
        feature_table_filename=env.get('features'),
        concept_table_filename=env.get('concepts'),
        tree_pdf_filename=env.get('pdf'),
        decision_tree_args=dt_args,
    )


def decision_tree_main():
    decision_tree(os.path.basename(sys.argv[0]), *sys.argv[1:])
