import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score


def compute_auc(y_true, y_pred):
    """
    Computes the area under curve (auc) score.

    Args:
        y_true: the true labels.
        y_pred: the predicted probabilities.

    Returns: auc score.

    """
    return roc_auc_score(
        np.asarray(y_true, dtype=np.float32), np.asarray(y_pred, dtype=np.float32)
    )


def auc(y_true, y_pred):
    """
    TensorFlow wrapper function over compute_auc.

    Args:
        y_true: the true labels.
        y_pred: the predicted labels.

    Returns: a TensorFlow decorated auc function.

    """

    return tf.py_function(compute_auc, (y_true, y_pred), tf.double)


def true_pos_rate(y_expected, y_predicted):
    """
    Computes the true positive rate (TPR).

    Args:
        y_expected: true labels.
        y_predicted: predicted probabilities.

    Returns: return tpr score.

    """

    y_expected = np.argmax(y_expected, axis=1)
    y_predicted = np.argmax(y_predicted, axis=1)
    tn, fp, fn, tp = confusion_matrix(y_expected, y_predicted, labels=[0, 1]).ravel()

    if tp + fn == 0:
        return 1 / 2

    return float(tp / (tp + fn))


def tpr(y_true, y_pred):
    """
    TensorFlow wrapper function over tpr.

    Args:
        y_expected: true labels.
        y_predicted: predicted probabilities.

    Returns: a TensorFlow decorated tpr function.

    """

    return tf.py_function(true_pos_rate, (y_true, y_pred), tf.double)


def true_neg_rate(y_expected, y_predicted):
    """
    Computes the true negative rate (TNR).

    Args:
       y_expected: true labels.
       y_predicted: predicted probabilities.

    Returns: return tnr score.

    """

    y_expected = np.argmax(y_expected, axis=1)
    y_predicted = np.argmax(y_predicted, axis=1)

    tn, fp, fn, tp = confusion_matrix(y_expected, y_predicted, labels=[0, 1]).ravel()

    if tn + fp == 0:
        return 1 / 2

    return float(tn / (tn + fp))


def tnr(y_true, y_pred):
    """
    TensorFlow wrapper function over tnr.

    Args:
        y_expected: true labels.
        y_predicted: predicted probabilities.

    Returns: a TensorFlow decorated tnr function.

    """

    return tf.py_function(true_neg_rate, (y_true, y_pred), tf.double)
