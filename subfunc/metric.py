"""
This module provides function for error metrics
"""

import numpy as np


def rmse_score(pred, targ):
    """
    Root-mean-square deviation between two arrays
    :param pred, trag (predictions, targets arrays)
    :return rmse (stand for root-mean-squre error)
    >>> rmse_score(np.array([1.,2.,3.]), np.array([1., 5., -1.]))
    >>> 2.886751 
    """
    assert pred.shape == targ.shape, "Shape mismatch"
    rmse = np.sqrt(np.mean(np.square(pred - targ)))
    return rmse


def mape_score(pred, targ):
    """
    Mean-absolute-percenatge-error (MAPE) between two scalars
    :param: pred, trag (predictions, targets, a scalar)
    :return: mape (mean-absolute-percenatge-error)
    """
    return np.abs(pred/targ-1)*100