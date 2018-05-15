"""Common functionality"""

# Copyright (c) 2018 Aubrey Barnard.  This is free software released
# under the MIT License.  See `LICENSE.txt` for details.


import csv
import io
import pathlib
import re
import statistics
import textwrap

from barnapy import logging


def open_file(filename, mode='rt'):
    if isinstance(filename, io.IOBase):
        return filename
    elif isinstance(filename, pathlib.Path):
        return open(str(filename), mode)
    else:
        return open(filename, mode)


def load_svmlight_as_matrix(filename):
    """
    Return (data, labels) as defined in the given svmlight file.

    Note that 1-based indices in svmlight files are 0-based in the
    matrix.
    """
    from sklearn import datasets
    with open_file(filename, 'rb') as file:
        return datasets.load_svmlight_file(file)


def read_csv(filename, num_header_lines=1, **csv_opts):
    with open_file(filename, 'rt') as file:
        for row_idx, row in enumerate(csv.reader(file, **csv_opts)):
            if row_idx < num_header_lines:
                continue
            yield row


def mk_rm2all_idxs(items, rm_items):
    """
    Given a collection of items and a subset of the items to exclude,
    make a mapping of the indices of the subset to the indices of the
    corresponding items in the original collection.
    """
    rm2all = {}
    rm_items = set(rm_items)
    rm_idx = 0
    for item_idx, item in enumerate(items):
        if item in rm_items:
            continue
        rm2all[rm_idx] = item_idx
        rm_idx += 1
    return rm2all


def rm_cols(data_matrix, rm_col_idxs):
    all_col_idxs = range(data_matrix.shape[1])
    rm_col_idxs = set(rm_col_idxs)
    rm2orig_idxs = mk_rm2all_idxs(all_col_idxs, rm_col_idxs)
    if rm_col_idxs:
        incl_idxs = sorted(set(all_col_idxs) - rm_col_idxs)
        data_matrix = data_matrix[:, incl_idxs]
    return data_matrix, rm2orig_idxs


def limit_matrix_to_features(data_matrix, feature_table):
    # Find out what columns to remove
    remove_col_idxs = []
    for col_idx in range(data_matrix.shape[1]):
        # Feature indices are 1-based while column indices are 0-based
        feat_idx = col_idx + 1
        if feat_idx not in feature_table:
            remove_col_idxs.append(col_idx)
    # Remove the columns
    return rm_cols(data_matrix, remove_col_idxs)


# Handling specific files


def load_feature_table(filename):
    logging.getLogger(__name__).info(
        'Loading feature table from: {}', filename)
    id_idx = 0
    nm_idx = 1
    val_idx = 4
    rows = read_csv(
        filename,
        delimiter='|',
        num_header_lines=1,
        quoting=csv.QUOTE_NONE,
    )
    return {int(r[id_idx]): (r[nm_idx], r[val_idx]) for r in rows}


def load_concept_table(filename):
    logging.getLogger(__name__).info(
        'Loading concept table from: {}', filename)
    id_idx = 0
    desc_idx = 1
    rows = read_csv(
        filename,
        delimiter='\t',
        num_header_lines=1,
        quoting=csv.QUOTE_NONE,
    )
    return {r[id_idx]: r[desc_idx] for r in rows}


# Reporting


def mk_feature_importance_table(
        feature_importances,
        rm2orig_idxs={},
        features={},
        concepts={},
):
    header = ('importance', 'col_idx', 'feat_id', 'feat_name',
              'concept_id', 'concept_desc')
    table = []
    for col_idx, importance in enumerate(feature_importances):
        # Add 1 to convert 0-based column indices to 1-based feature IDs
        feat_id = rm2orig_idxs.get(col_idx, col_idx) + 1
        feat_nm, cncpt_id = features.get(feat_id, (None, None))
        cncpt_desc = concepts.get(cncpt_id)
        row = [importance, col_idx, feat_id, feat_nm,
               cncpt_id, cncpt_desc]
        table.append(row)
    return header, table


def replace_variable_references_with_features(
        text,
        rm2orig_idxs={},
        features={},
        concepts={},
):
    # Replace the variable references with features
    feature_legend = {}
    var_pattern = re.compile(r'X\[(\d+)\]')
    new_text = io.StringIO()
    pos = 0
    match = var_pattern.search(text)
    while match is not None:
        # Copy skipped input to output
        sta_pos, end_pos = match.span()
        new_text.write(text[pos:sta_pos])
        pos = end_pos
        # Write new variable name to output
        col_idx = int(match.group(1))
        # Add 1 to convert 0-based column indices to 1-based feature IDs
        feat_id = rm2orig_idxs.get(col_idx, col_idx) + 1
        if feat_id in features:
            feat_nm, cncpt_id = features[feat_id]
            new_name = 'X[{}_{}]'.format(feat_id, feat_nm)
            if cncpt_id in concepts:
                cncpt_desc = concepts[cncpt_id]
                feature_legend[new_name] = cncpt_desc
        else:
            new_name = 'X[{}]'.format(feat_id)
        new_text.write(new_name)
        # Find next match
        match = var_pattern.search(text, end_pos)
    # Copy to output the unmatched, trailing input
    new_text.write(text[pos:])
    return new_text.getvalue(), feature_legend


def print_report(
        cv_roc_areas=None,
        final_model_roc_area=None,
        feature_table=None,
        feature_table_header=None,
        limit_n_features=100,
        model_text=None,
        feature_legend=None,
):
    logging.getLogger(__name__).info('Printing report')
    print('%YAML 1.2')
    print('---')
    print()
    # Report ROC areas
    if cv_roc_areas:
        print('ROC areas by fold:')
        for idx, score in enumerate(cv_roc_areas):
            print(' ', idx + 1, ':', score)
        print('sorted ROC areas:')
        for score in sorted(cv_roc_areas):
            print(' ', '-', score)
        print('mean ROC area:', statistics.mean(cv_roc_areas))
        print()
    if final_model_roc_area is not None:
        print('final model ROC area:', final_model_roc_area)
        print()
    # Report features
    if feature_table:
        print('features ranked by importance:')
        if feature_table_header:
            header = ['rank']
            header.extend(feature_table_header)
            print('  -', header)
        for rank, feat_row in enumerate(
                feature_table[:limit_n_features]):
            print('  - [', end='')
            print(rank + 1, *feat_row, sep=', ', end=']\n')
        print()
    # Include a textual description of the model
    if model_text:
        print('model: |')
        print(textwrap.indent(model_text, '  '))
        print()
    # Include the feature legend
    if feature_legend:
        print('feature legend:')
        for feat_nm in sorted(feature_legend.keys()):
            print('  ', feat_nm, ': ', repr(feature_legend[feat_nm]),
                  sep='')
        print()
    # EOF
    print('...')


def render_dot_as_pdf(dot_text, pdf_filename):
    logging.getLogger(__name__).info(
        'Rendering Dot text into PDF: {}', pdf_filename)
    pdf_path = pathlib.Path(pdf_filename)
    render_success = False
    # Render using PyDot if available
    try:
        import pydot
        graphs = pydot.graph_from_dot_data(dot_text)
        graphs[0].write(str(pdf_path), format='pdf', prog='dot')
        render_success = True
    except ImportError:
        pass
    # Return if successful
    if render_success:
        return True
    # Render tree as PDF using Graphviz if available
    try:
        import graphviz
        graph = graphviz.Source(dot_text, format='pdf', engine='dot')
        # Remove suffix because `graphviz` appends a suffix
        pdf_path = pdf_path.with_suffix('')
        graph.render(str(pdf_path), cleanup=True)
        render_success = True
    except ImportError:
        pass
    # Return whether successful
    return render_success
