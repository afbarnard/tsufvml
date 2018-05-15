"""Entry points for scripts and APIs for foreign code"""

# Copyright (c) 2018 Aubrey Barnard.  This is free software released
# under the MIT License.  See `LICENSE.txt` for details.


import argparse
import operator
import os.path
import pathlib
import sys

from barnapy import logging

# Use absolute imports so that this file can be used from anywhere.
# Postpone expensive imports (i.e. sklearn) until needed.
import tsufvml
from tsufvml import common


def run_decision_trees_api(
        data_matrix_filename,
        feature_table_filename=None,
        concept_table_filename=None,
        tree_pdf_filename='tree.pdf',
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
    # Run the decision tree classifier
    dt_model = ml.mk_decision_tree()
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
    # Render tree as PDF
    render_ok = common.render_dot_as_pdf(dot_text, tree_pdf_filename)
    if not render_ok:
        print(
            """

Warning: Unable to render the decision tree as a PDF using either the
    `pydot` or `graphviz` packages.  If you want automatic rendering,
    make sure one of those packages is installed and try again.

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
        metavar='TABLE',
        help='Table of features in delimited format',
    )
    arg_prsr.add_argument(
        '--concepts',
        type=argparse.FileType('rt'),
        metavar='TABLE',
        help='Table of concepts in delimited format',
    )
    arg_prsr.add_argument(
        '--pdf',
        type=pathlib.Path,
        metavar='PATH',
        help='Filename for PDF output',
    )
    env = arg_prsr.parse_args(args)
    env = vars(env) # Convert to dictionary
    logging.default_config()
    run_decision_trees_api(
        env.get('data'),
        env.get('features'),
        env.get('concepts'),
        env.get('pdf'),
    )


def decision_tree_main():
    decision_tree(os.path.basename(sys.argv[0]), *sys.argv[1:])
