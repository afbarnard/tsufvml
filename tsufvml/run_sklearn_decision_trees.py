"""
Runs scikit-learn decision trees on feature vector data in SVMLight
format
"""

# Copyright (c) 2018 Aubrey Barnard.  This is free software released
# under the MIT License.  See `LICENSE.txt` for details.


# TODO split into script to do machine learning and script to interpret model / output (still map internal data IDs to feature IDs in output, though)
# TODO break into functions
# TODO put functions in importable module
# TODO convert script to main()
# TODO have setuptools install script
# TODO allow blacklist, feature table, concept table to be specified as options on command line
# TODO allow concept table columns to be specified in options; need ID, desc
# TODO clean up spacing in report
# TODO generate tree PDF (if dot package installed, if requested with command line option for file, e.g. `--tree-pdf=dt.pdf`)
# TODO update README with instructions for invoking new script


import csv
import io
import operator
import pprint
import re
import statistics
import sys
import textwrap


# Check command line arguments
if not (2 <= len(sys.argv) <= 5):
    print('Error: Incorrect command line arguments', file=sys.stderr)
    print('Usage: <data-file> [<features-file> [<concept-table> [<blacklist-file>]]]', file=sys.stderr)
    sys.exit(2)

# Delay these expensive imports until after args have been checked
import numpy
import sklearn
import sklearn.cross_validation as cv
import sklearn.datasets as datasets
import sklearn.metrics as metrics
import sklearn.tree as tree

# Get command line arguments
data_filename = sys.argv[1]
features_table_filename = (sys.argv[2] if len(sys.argv) >= 3 else None)
concept_table_filename = (sys.argv[3] if len(sys.argv) >= 4 else None)
blacklist_filename = (sys.argv[4] if len(sys.argv) >= 5 else None)

# Load the data.  Open in binary mode because that's how
# load_svmlight_file expects it.  1-based indices in svmlight file are
# loaded as 0-based.
with open(data_filename, 'rb') as data_file:
    data, labels = datasets.load_svmlight_file(data_file)
print(data[0:3, :], file=sys.stderr)
# Densify data for older sklearns.  I actually don't know the minimum
# version for trees to handle sparse data but I'm assuming it's 0.16.0.
skl_version = tuple(map(int, sklearn.__version__.split('.')))
if skl_version < (0, 16, 0):
    data = data.toarray()

# Load the features table
features_table_id_idx = 0
features_table_name_idx = 1
features_table_value_idx = 4
features = {}
if features_table_filename:
    with open(features_table_filename, 'rt') as csvfile:
        for row_idx, row in enumerate(csv.reader(csvfile, delimiter='|', quoting=csv.QUOTE_NONE)):
            # Skip the header
            if row_idx == 0:
                continue
            feat_id = int(row[features_table_id_idx])
            feat_name = row[features_table_name_idx]
            feat_val = row[features_table_value_idx]
            features[feat_id] = (feat_name, feat_val)

# Load the concept table
concept_table_id_idx = 0
concept_table_desc_idx = 1
concepts = {}
if concept_table_filename:
    with open(concept_table_filename, 'rt') as csvfile:
        for row_idx, row in enumerate(csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)):
            # Skip the header
            if row_idx == 0:
                continue
            concept_id = row[concept_table_id_idx]
            concept_desc = row[concept_table_desc_idx]
            concepts[concept_id] = concept_desc

# Load the blacklist
blacklisted_feature_ids = {0, 1} # Remove data ID, label
if blacklist_filename is not None:
    with open(blacklist_filename, 'rt') as text:
        blacklisted_feature_ids += set(map(lambda i: int(i) - 1, text))
# Make sure the blacklisted feature IDs are a set for membership testing
blacklisted_feature_ids = set(blacklisted_feature_ids)
print('Blacklisted feature IDs:', file=sys.stderr)
pprint.pprint(blacklisted_feature_ids, stream=sys.stderr)
# Apply the blacklist
all_feature_ids = range(data.shape[1])
if blacklisted_feature_ids:
    print('Removing blacklisted features...', file=sys.stderr)
    include_feature_ids = sorted(set(all_feature_ids) -
                                 set(blacklisted_feature_ids))
    data = data[:, include_feature_ids]
print(data[0:3, :], file=sys.stderr)
# Map column indices to feature IDs
col_idxs2feat_ids = {}
col_idx = 0
for feat_id in sorted(all_feature_ids):
    if feat_id in blacklisted_feature_ids:
        continue
    col_idxs2feat_ids[col_idx] = feat_id + 1 # Adjust to 1-based indices
    col_idx += 1

# TODO Split the data into folds such that multiple examples from a
# patient are always in the same fold and that the distribution of
# labels in each fold is the same

# Set up decision tree classifier
model = tree.DecisionTreeClassifier(
    max_features=None, # Consider all features
    max_depth=4,
    min_samples_leaf=10,
    class_weight='balanced', # Treat classes equally regardless of skew
    )

# Run 10-fold cross validation and evaluate it with ROC area
cv_folds = cv.StratifiedKFold(labels, 10)
scores = []
importances = []
for fold in cv_folds:
    train_idxs, test_idxs = fold
    # Select the training data
    train_data = data[train_idxs, :]
    train_labels = labels[train_idxs]
    # Select the testing data
    test_data = data[test_idxs, :]
    test_labels = labels[test_idxs]
    # Fit the model
    model.fit(train_data, train_labels)
    # Test
    predictions = model.predict(test_data)
    # Score
    roc_area = metrics.roc_auc_score(test_labels, predictions)
    scores.append(roc_area)
    # Save feature importances.  Copy because we aren't guaranteed to
    # get our own array.  (I don't know what is returned because it's a
    # property whose return value depends on rather impenetrable C
    # code.)
    importances.append(model.feature_importances_.copy())

# Run a final tree on all the data for visualization purposes
model.fit(data, labels)
# Save the model in Dot format
dot_text = io.StringIO()
tree.export_graphviz(model, out_file=dot_text)
dot_text_str = dot_text.getvalue()

# Replace column references with features
feat_legend = {}
var_pattern = re.compile(r'X\[(\d+)\]')
new_dot_text = io.StringIO()
pos = 0
match = var_pattern.search(dot_text_str)
while match is not None:
    # Copy skipped input to output
    sta_pos, end_pos = match.span()
    new_dot_text.write(dot_text_str[pos:sta_pos])
    pos = end_pos
    # Write new variable name to output
    col_idx = int(match.group(1))
    feat_id = col_idxs2feat_ids[col_idx]
    if feat_id in features:
        feat_nm, cncpt_id = features[feat_id]
        new_name = 'X[{}_{}]'.format(feat_id, feat_nm)
        if cncpt_id in concepts:
            cncpt_desc = concepts[cncpt_id]
            feat_legend[new_name] = cncpt_desc
    else:
        new_name = 'X[{}]'.format(feat_id)
    new_dot_text.write(new_name)
    # Find next match
    match = var_pattern.search(dot_text_str, end_pos)
# Copy skipped input to output
new_dot_text.write(dot_text_str[pos:])

# Average feature importances
avg_importances = numpy.zeros_like(importances[0])
for imps in importances:
    avg_importances += imps
if len(importances) > 1:
    avg_importances /= len(importances)

# Rank features by average importance
ranked_importances = sorted(
    enumerate(avg_importances),
    key=operator.itemgetter(1), reverse=True)

# Print report
print('%YAML 1.2')
print('---')
print()
# Report ROC areas
print('ROC areas by fold:')
for idx, score in enumerate(scores):
    print(' ', idx + 1, ':', score)
print('sorted ROC areas:')
for score in sorted(scores):
    print(' ', '-', score)
print('mean ROC area:', statistics.mean(scores))
print()
# Report up to the first 100 nonzero feature importances
print('ranked average feature importances:')
print('  - [rank, importance, col_idx, feat_id, feat_name, concept_id, concept_desc]')
for (rank, (col_idx, feat_avg_imp)) in enumerate(ranked_importances[:100]):
    if feat_avg_imp > 0:
        feat_id = col_idxs2feat_ids[col_idx]
        feat_nm, cncpt_id = features.get(feat_id, (None, None))
        cncpt_desc = concepts.get(cncpt_id)
        print('  - [', end='')
        print(rank + 1, feat_avg_imp, col_idx, feat_id, repr(feat_nm), repr(cncpt_id), repr(cncpt_desc), sep=', ', end=']\n')
    else:
        # Don't print zeros
        break
print()
# Report the overall tree (in Dot format)
#print('overall tree: |')
#print(textwrap.indent(dot_text.getvalue(), '  '))
print('overall tree: |')
print(textwrap.indent(new_dot_text.getvalue(), '  '))
print('tree feature legend:')
for feat_nm in sorted(feat_legend.keys()):
    print('  ', feat_nm, ': ', repr(feat_legend[feat_nm]), sep='')
# EOF
print('...')
