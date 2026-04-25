"""
Backward-compatibility shim. Plotting functions have moved to scalex/pl/plot.py.
Importing from this module will raise a DeprecationWarning.
"""
import warnings

from scalex.pl.plot import (
    embedding as _embedding,
    plot_expr as _plot_expr,
    plot_meta as _plot_meta,
    plot_meta2 as _plot_meta2,
    plot_confusion as _plot_confusion,
    plot_subplots as _plot_subplots,
    reassign_cluster_with_ref as _reassign_cluster_with_ref,
    plot_radar as _plot_radar,
)

_MSG = "scalex.plot.{name} is deprecated; use scalex.pl.plot.{name} instead."


def embedding(*args, **kwargs):
    warnings.warn(_MSG.format(name='embedding'), DeprecationWarning, stacklevel=2)
    return _embedding(*args, **kwargs)


def plot_expr(*args, **kwargs):
    warnings.warn(_MSG.format(name='plot_expr'), DeprecationWarning, stacklevel=2)
    return _plot_expr(*args, **kwargs)


def plot_meta(*args, **kwargs):
    warnings.warn(_MSG.format(name='plot_meta'), DeprecationWarning, stacklevel=2)
    return _plot_meta(*args, **kwargs)


def plot_meta2(*args, **kwargs):
    warnings.warn(_MSG.format(name='plot_meta2'), DeprecationWarning, stacklevel=2)
    return _plot_meta2(*args, **kwargs)


def plot_confusion(*args, **kwargs):
    warnings.warn(_MSG.format(name='plot_confusion'), DeprecationWarning, stacklevel=2)
    return _plot_confusion(*args, **kwargs)


def plot_subplots(*args, **kwargs):
    warnings.warn(_MSG.format(name='plot_subplots'), DeprecationWarning, stacklevel=2)
    return _plot_subplots(*args, **kwargs)


def reassign_cluster_with_ref(*args, **kwargs):
    warnings.warn(_MSG.format(name='reassign_cluster_with_ref'), DeprecationWarning, stacklevel=2)
    return _reassign_cluster_with_ref(*args, **kwargs)


def plot_radar(*args, **kwargs):
    warnings.warn(_MSG.format(name='plot_radar'), DeprecationWarning, stacklevel=2)
    return _plot_radar(*args, **kwargs)
