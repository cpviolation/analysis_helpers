import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from analysis_helpers.plotting import (
    adaptive_bin_edges,
    create_subplots,
    plot_2darray,
    plot_hist,
    plot_hist2d,
    plot_vertical_lines,
)


def test_plot_hist_sets_labels_and_limits():
    data = np.array([0.0, 1.0, 2.0, 3.0])

    fig, ax = plot_hist(data, name="mass", unit="GeV", bins=4, range=(0.0, 4.0))

    assert ax.get_xlabel() == "mass [GeV]"
    assert ax.get_ylabel() == "Events / (1.0 GeV)"
    assert ax.get_xlim() == pytest.approx((0.0, 4.0))
    fig.clf()


def test_plot_vertical_lines_adds_requested_markers():
    fig, ax = plot_hist(np.array([0.0, 1.0]), bins=2, range=(0.0, 2.0))

    plot_vertical_lines(ax, [0.5, 1.5])

    assert len(ax.lines) == 2
    fig.clf()


def test_plot_hist2d_sets_axis_labels():
    x = np.array([0.0, 0.5, 1.0, 1.5])
    y = np.array([0.0, 0.5, 1.0, 1.5])

    fig, ax = plot_hist2d(x, y, xlabel="x", ylabel="y", bins=2)

    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert len(fig.axes) == 2
    fig.clf()


def test_adaptive_bin_edges_respects_requested_range():
    data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    edges = adaptive_bin_edges(data, bins=4, range=(1.0, 3.0))

    assert edges[0] == pytest.approx(1.0)
    assert edges[-1] == pytest.approx(3.0)
    assert np.all(np.diff(edges) >= 0)


def test_create_subplots_hides_unused_axes():
    fig, axes = create_subplots(3)

    assert len(axes) == 4
    assert axes[3].get_visible() is False
    fig.clf()


def test_plot_2darray_sets_metadata():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])

    fig, ax = plot_2darray(
        data,
        xlabel="x",
        ylabel="y",
        title="heatmap",
        colorbar_label="Entries",
        zrange=(0.0, 4.0),
    )

    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert ax.get_title() == "heatmap"
    assert len(fig.axes) == 2
    fig.clf()
