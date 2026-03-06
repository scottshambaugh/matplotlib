import numpy as np
import numpy.testing as nptest
import pytest

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal

from matplotlib.backend_bases import MouseEvent
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d.art3d import (
    get_dir_vector,
    Line3DCollection,
    Poly3DCollection,
    _all_points_on_plane,
)


@pytest.mark.parametrize("zdir, expected", [
    ("x", (1, 0, 0)),
    ("y", (0, 1, 0)),
    ("z", (0, 0, 1)),
    (None, (0, 0, 0)),
    ((1, 2, 3), (1, 2, 3)),
    (np.array([4, 5, 6]), (4, 5, 6)),
])
def test_get_dir_vector(zdir, expected):
    res = get_dir_vector(zdir)
    assert isinstance(res, np.ndarray)
    nptest.assert_array_equal(res, expected)


def test_scatter_3d_projection_conservation():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # fix axes3d projection
    ax.roll = 0
    ax.elev = 0
    ax.azim = -45
    ax.stale = True

    x = [0, 1, 2, 3, 4]
    scatter_collection = ax.scatter(x, x, x)
    fig.canvas.draw_idle()

    # Get scatter location on canvas and freeze the data
    scatter_offset = scatter_collection.get_offsets()
    scatter_location = ax.transData.transform(scatter_offset)

    # Yaw -44 and -46 are enough to produce two set of scatter
    # with opposite z-order without moving points too far
    for azim in (-44, -46):
        ax.azim = azim
        ax.stale = True
        fig.canvas.draw_idle()

        for i in range(5):
            # Create a mouse event used to locate and to get index
            # from each dots
            event = MouseEvent("button_press_event", fig.canvas,
                               *scatter_location[i, :])
            contains, ind = scatter_collection.contains(event)
            assert contains is True
            assert len(ind["ind"]) == 1
            assert ind["ind"][0] == i


def test_zordered_error():
    # Smoke test for https://github.com/matplotlib/matplotlib/issues/26497
    lc = [(np.fromiter([0.0, 0.0, 0.0], dtype="float"),
           np.fromiter([1.0, 1.0, 1.0], dtype="float"))]
    pc = [np.fromiter([0.0, 0.0], dtype="float"),
          np.fromiter([0.0, 1.0], dtype="float"),
          np.fromiter([1.0, 1.0], dtype="float")]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.add_collection(Line3DCollection(lc), autolim="_datalim_only")
    ax.scatter(*pc, visible=False)
    plt.draw()


def test_all_points_on_plane():
    # Non-coplanar points
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert not _all_points_on_plane(*points.T)

    # Duplicate points
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert _all_points_on_plane(*points.T)

    # NaN values
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, np.nan]])
    assert _all_points_on_plane(*points.T)

    # Less than 3 unique points
    points = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert _all_points_on_plane(*points.T)

    # All points lie on a line
    points = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]])
    assert _all_points_on_plane(*points.T)

    # All points lie on two lines, with antiparallel vectors
    points = np.array([[-2, 2, 0], [-1, 1, 0], [1, -1, 0],
                       [0, 0, 0], [2, 0, 0], [1, 0, 0]])
    assert _all_points_on_plane(*points.T)

    # All points lie on a plane
    points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 2, 0]])
    assert _all_points_on_plane(*points.T)


def test_generate_normals():
    # Smoke test for https://github.com/matplotlib/matplotlib/issues/29156
    vertices = ((0, 0, 0), (0, 5, 0), (5, 5, 0), (5, 0, 0))
    shape = Poly3DCollection([vertices], edgecolors='r', shade=True)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.add_collection3d(shape)
    plt.draw()


# --- set_offsets3d / get_offsets3d tests ---

def test_offsets3d_patch3d_roundtrip():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    col = ax.scatter([0, 1], [0, 1], [0, 1])
    offsets = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    col.set_offsets3d(offsets)
    xs, ys, zs = col.get_offsets3d()
    nptest.assert_array_almost_equal(xs, offsets[:, 0])
    nptest.assert_array_almost_equal(ys, offsets[:, 1])
    nptest.assert_array_almost_equal(zs, offsets[:, 2])


def test_offsets3d_poly3d_roundtrip():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    verts = [[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]]
    poly = Poly3DCollection(verts)
    ax.add_collection3d(poly)
    offsets = np.array([[1, 2, 3]], dtype=float)
    poly.set_offsets3d(offsets)
    xs, ys, zs = poly.get_offsets3d()
    nptest.assert_array_almost_equal(xs, offsets[:, 0])
    nptest.assert_array_almost_equal(ys, offsets[:, 1])
    nptest.assert_array_almost_equal(zs, offsets[:, 2])


def test_offsets3d_single_offset_broadcast():
    """A (3,) offset should be broadcast to (1, 3)."""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    col = ax.scatter([0], [0], [0])
    col.set_offsets3d([1, 2, 3])
    xs, ys, zs = col.get_offsets3d()
    nptest.assert_array_almost_equal(xs, [1])
    nptest.assert_array_almost_equal(ys, [2])
    nptest.assert_array_almost_equal(zs, [3])


@pytest.mark.parametrize("zdir", ['x', 'y', 'z'])
def test_offsets3d_zdir_roundtrip(zdir):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    col = ax.scatter([0], [0], [0])
    col.set_offsets3d([[1, 2, 3]], zdir=zdir)
    xs, ys, zs = col.get_offsets3d()
    # The offsets are juggled according to zdir, so just verify
    # the values are preserved (in some axis order)
    result = sorted([xs[0], ys[0], zs[0]])
    assert result == [1.0, 2.0, 3.0]


@check_figures_equal()
def test_offsets3d_scatter_set(fig_test, fig_ref):
    """set_offsets3d moves scatter points to new positions."""
    # Reference: scatter at target positions directly
    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.scatter([1, 2], [3, 4], [5, 6])
    ax_ref.set_xlim(0, 3)
    ax_ref.set_ylim(2, 5)
    ax_ref.set_zlim(4, 7)

    # Test: scatter at origin, then move with set_offsets3d
    ax_test = fig_test.add_subplot(projection='3d')
    col = ax_test.scatter([0, 0], [0, 0], [0, 0])
    col.set_offsets3d([[1, 3, 5], [2, 4, 6]])
    ax_test.set_xlim(0, 3)
    ax_test.set_ylim(2, 5)
    ax_test.set_zlim(4, 7)


@check_figures_equal()
def test_offsets3d_poly3d_translate(fig_test, fig_ref):
    """Poly3DCollection: set_offsets3d translates polygon faces."""
    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                      dtype=float)
    offset = np.array([2, 3, 4], dtype=float)

    # Reference: polygon placed at target position directly
    ax_ref = fig_ref.add_subplot(projection='3d')
    poly_ref = Poly3DCollection([square + offset], alpha=0.5)
    ax_ref.add_collection3d(poly_ref)
    ax_ref.set_xlim(0, 5)
    ax_ref.set_ylim(0, 5)
    ax_ref.set_zlim(0, 5)

    # Test: polygon at origin, moved with set_offsets3d
    ax_test = fig_test.add_subplot(projection='3d')
    poly_test = Poly3DCollection([square], alpha=0.5)
    ax_test.add_collection3d(poly_test)
    poly_test.set_offsets3d([offset])
    ax_test.set_xlim(0, 5)
    ax_test.set_ylim(0, 5)
    ax_test.set_zlim(0, 5)


@check_figures_equal()
def test_offsets3d_poly3d_multiple(fig_test, fig_ref):
    """Poly3DCollection: N offsets translate N faces independently."""
    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                      dtype=float)
    offsets = np.array([[0, 0, 0], [3, 3, 3]], dtype=float)

    # Reference: two polygons placed at target positions directly
    ax_ref = fig_ref.add_subplot(projection='3d')
    poly_ref = Poly3DCollection([square + offsets[0], square + offsets[1]],
                                alpha=0.5)
    ax_ref.add_collection3d(poly_ref)
    ax_ref.set_xlim(-1, 5)
    ax_ref.set_ylim(-1, 5)
    ax_ref.set_zlim(-1, 5)

    # Test: two polygons at origin, moved with set_offsets3d
    ax_test = fig_test.add_subplot(projection='3d')
    poly_test = Poly3DCollection([square, square], alpha=0.5)
    ax_test.add_collection3d(poly_test)
    poly_test.set_offsets3d(offsets)
    ax_test.set_xlim(-1, 5)
    ax_test.set_ylim(-1, 5)
    ax_test.set_zlim(-1, 5)


@check_figures_equal()
def test_offsets3d_poly3d_broadcast_single(fig_test, fig_ref):
    """Poly3DCollection: 1 offset broadcasts to all N faces."""
    square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                      dtype=float)
    offset = np.array([2, 2, 2], dtype=float)

    # Reference: two polygons shifted by same offset directly
    ax_ref = fig_ref.add_subplot(projection='3d')
    poly_ref = Poly3DCollection(
        [square + offset, square + np.array([0, 3, 0]) + offset],
        alpha=0.5)
    ax_ref.add_collection3d(poly_ref)
    ax_ref.set_xlim(0, 5)
    ax_ref.set_ylim(0, 6)
    ax_ref.set_zlim(0, 5)

    # Test: two polygons, single offset broadcast to both
    ax_test = fig_test.add_subplot(projection='3d')
    poly_test = Poly3DCollection(
        [square, square + np.array([0, 3, 0])], alpha=0.5)
    ax_test.add_collection3d(poly_test)
    poly_test.set_offsets3d([offset])
    ax_test.set_xlim(0, 5)
    ax_test.set_ylim(0, 6)
    ax_test.set_zlim(0, 5)
