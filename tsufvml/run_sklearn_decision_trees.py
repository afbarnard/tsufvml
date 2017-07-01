"""Runs scikit-learn decision trees on feature vector data in SVMLight
format

"""

# Copyright (c) 2016 Aubrey Barnard.  This is free software released
# under the MIT License.  See `LICENSE.txt` for details.


import io
import lzma
import operator
import pprint
import statistics
import sys
import textwrap


# Check command line arguments
if not (2 <= len(sys.argv) <= 3):
    print('Incorrect command line arguments', file=sys.stderr)
    print('Usage: <data-file> <blacklist-file>?', file=sys.stderr)
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
if len(sys.argv) >= 3:
    blacklist_filename = sys.argv[2]
else:
    blacklist_filename = None

# Load the data.  Open in binary mode because that's how
# load_svmlight_file expects it.
data_file = lzma.open(data_filename, 'rb')
data, labels = datasets.load_svmlight_file(data_file)
# Densify data for older sklearns.  I actually don't know the minimum
# version for trees to handle sparse data but I'm assuming it's 0.16.0.
skl_version = tuple(map(int, sklearn.__version__.split('.')))
if skl_version < (0, 16, 0):
    data = data.toarray()

# Load the blacklist
blacklisted_feature_ids = None
if blacklist_filename is not None:
    with open(blacklist_filename, 'rt') as text:
        blacklisted_feature_ids = set(map(int, text))
    print('Blacklisted feature IDs:', file=sys.stderr)
    pprint.pprint(blacklisted_feature_ids, stream=sys.stderr)
# Apply the blacklist
if blacklisted_feature_ids:
    print('Removing blacklisted features...', file=sys.stderr)
    all_feature_ids = set(range(data.shape[1]))
    include_feature_ids = sorted(all_feature_ids
                                 - set(blacklisted_feature_ids))
    data = data[:, include_feature_ids]

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
for feat_idx, feat_avg_imp in ranked_importances[:100]:
    if feat_avg_imp > 0:
        print(' ', feat_idx, ':', feat_avg_imp)
    else:
        # Don't print zeros
        break
print()
# Report the overall tree (in Dot format)
print('overall tree: |')
print(textwrap.indent(dot_text.getvalue(), '  '))
# EOF
print('...')
