"""Machine learning algorithms"""

# Copyright (c) 2018-2019 Aubrey Barnard.
#
# This is free software released under the MIT License.  See
# `LICENSE.txt` for details.


import io

from sklearn import model_selection
from sklearn import metrics
from sklearn import tree

from barnapy import logging


def run_cv_and_final_model(model, data, labels, weights=None):
    logger = logging.getLogger(__name__)
    logger.info(
        'run_cv_and_final_model:\n'
        '  model:   {}\n'
        '  data:    {} {}\n'
        '  labels:  {} {}\n'
        '  weights: {} {}',
        model,
        data.shape, data.dtype,
        labels.shape, labels.dtype,
        weights.shape if weights is not None else None,
        weights.dtype if weights is not None else None,
    )
    # Run 10-fold cross validation and evaluate it with ROC area
    cv = model_selection.StratifiedKFold(10, shuffle=True)
    cv_folds = cv.split(data, labels)
    scores = []
    importances = []
    for fold_idx, fold in enumerate(cv_folds):
        logger.info('CV fold {}', fold_idx + 1)
        train_idxs, test_idxs = fold
        # Select the training data
        train_data = data[train_idxs, :]
        train_labels = labels[train_idxs]
        train_wgts = (weights[train_idxs]
                      if weights is not None else None)
        # Select the testing data
        test_data = data[test_idxs, :]
        test_labels = labels[test_idxs]
        test_wgts = weights[test_idxs] if weights is not None else None
        # Fit the model
        model.fit(train_data, train_labels, sample_weight=train_wgts)
        # Test
        predictions = model.predict(test_data)
        # Score
        roc_area = metrics.roc_auc_score(
            test_labels, predictions, sample_weight=test_wgts)
        scores.append(roc_area)
        # Save feature importances.  Copy because we aren't guaranteed
        # to get our own array.  (I don't know what is returned because
        # it's a property whose return value depends on rather
        # impenetrable C code.)
        importances.append(model.feature_importances_.copy())
    # Fit a final model on all the data
    logger.info('Fitting final model')
    model.fit(data, labels, sample_weight=weights)
    predictions = model.predict(data)
    final_score = metrics.roc_auc_score(
        labels, predictions, sample_weight=weights)
    logger.info('Done run_cv_and_final_model')
    return model, scores, importances, final_score


def render_decision_tree_as_graphviz(dt_model):
    # Render the tree as Graphviz Dot text
    dot_text = io.StringIO()
    tree.export_graphviz(dt_model, out_file=dot_text)
    return dot_text.getvalue()
