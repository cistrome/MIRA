
import matplotlib.pyplot as plt
import numpy as np
import warnings
from mira.plots.base import map_colors
from sklearn.preprocessing import minmax_scale

class Beeswarm:
    """Modifies a scatterplot artist to show a beeswarm plot."""
    def __init__(self, orient="v", width=0.8, warn_thresh=.05):

        # XXX should we keep the orient parameterization or specify the swarm axis?

        self.orient = orient
        self.width = width
        self.warn_thresh = warn_thresh

    def __call__(self, points, center):
        """Swarm `points`, a PathCollection, around the `center` position."""
        # Convert from point size (area) to diameter

        ax = points.axes
        dpi = ax.figure.dpi

        # Get the original positions of the points
        orig_xy_data = points.get_offsets()

        # Reset the categorical positions to the center line
        cat_idx = 1 if self.orient == "h" else 0
        orig_xy_data[:, cat_idx] = center

        # Transform the data coordinates to point coordinates.
        # We'll figure out the swarm positions in the latter
        # and then convert back to data coordinates and replot
        orig_x_data, orig_y_data = orig_xy_data.T
        orig_xy = ax.transData.transform(orig_xy_data)

        # Order the variables so that x is the categorical axis
        if self.orient == "h":
            orig_xy = orig_xy[:, [1, 0]]

        # Add a column with each point's radius
        sizes = points.get_sizes()
        if sizes.size == 1:
            sizes = np.repeat(sizes, orig_xy.shape[0])
        edge = points.get_linewidth().item()
        radii = (np.sqrt(sizes) + edge) / 2 * (dpi / 72)
        orig_xy = np.c_[orig_xy, radii]

        # Sort along the value axis to facilitate the beeswarm
        sorter = np.argsort(orig_xy[:, 1])
        orig_xyr = orig_xy[sorter]

        # Adjust points along the categorical axis to prevent overlaps
        new_xyr = np.empty_like(orig_xyr)
        new_xyr[sorter] = self.beeswarm(orig_xyr)

        # Transform the point coordinates back to data coordinates
        if self.orient == "h":
            new_xy = new_xyr[:, [1, 0]]
        else:
            new_xy = new_xyr[:, :2]
        new_x_data, new_y_data = ax.transData.inverted().transform(new_xy).T

        swarm_axis = {"h": "y", "v": "x"}[self.orient]
        log_scale = getattr(ax, f"get_{swarm_axis}scale")() == "log"

        # Add gutters
        if self.orient == "h":
            self.add_gutters(new_y_data, center, log_scale=log_scale)
        else:
            self.add_gutters(new_x_data, center, log_scale=log_scale)

        # Reposition the points so they do not overlap
        if self.orient == "h":
            points.set_offsets(np.c_[orig_x_data, new_y_data])
        else:
            points.set_offsets(np.c_[new_x_data, orig_y_data])

    def beeswarm(self, orig_xyr):
        """Adjust x position of points to avoid overlaps."""
        # In this method, `x` is always the categorical axis
        # Center of the swarm, in point coordinates
        midline = orig_xyr[0, 0]

        # Start the swarm with the first point
        swarm = np.atleast_2d(orig_xyr[0])

        # Loop over the remaining points
        for xyr_i in orig_xyr[1:]:

            # Find the points in the swarm that could possibly
            # overlap with the point we are currently placing
            neighbors = self.could_overlap(xyr_i, swarm)

            # Find positions that would be valid individually
            # with respect to each of the swarm neighbors
            candidates = self.position_candidates(xyr_i, neighbors)

            # Sort candidates by their centrality
            offsets = np.abs(candidates[:, 0] - midline)
            candidates = candidates[np.argsort(offsets)]

            # Find the first candidate that does not overlap any neighbors
            new_xyr_i = self.first_non_overlapping_candidate(candidates, neighbors)

            # Place it into the swarm
            swarm = np.vstack([swarm, new_xyr_i])

        return swarm

    def could_overlap(self, xyr_i, swarm):
        """Return a list of all swarm points that could overlap with target."""
        # Because we work backwards through the swarm and can short-circuit,
        # the for-loop is faster than vectorization
        _, y_i, r_i = xyr_i
        neighbors = []
        for xyr_j in reversed(swarm):
            _, y_j, r_j = xyr_j
            if (y_i - y_j) < (r_i + r_j):
                neighbors.append(xyr_j)
            else:
                break
        return np.array(neighbors)[::-1]

    def position_candidates(self, xyr_i, neighbors):
        """Return a list of coordinates that might be valid by adjusting x."""
        candidates = [xyr_i]
        x_i, y_i, r_i = xyr_i
        left_first = True
        for x_j, y_j, r_j in neighbors:
            dy = y_i - y_j
            dx = np.sqrt(max((r_i + r_j) ** 2 - dy ** 2, 0)) * 1.05
            cl, cr = (x_j - dx, y_i, r_i), (x_j + dx, y_i, r_i)
            if left_first:
                new_candidates = [cl, cr]
            else:
                new_candidates = [cr, cl]
            candidates.extend(new_candidates)
            left_first = not left_first
        return np.array(candidates)

    def first_non_overlapping_candidate(self, candidates, neighbors):
        """Find the first candidate that does not overlap with the swarm."""

        # If we have no neighbors, all candidates are good.
        if len(neighbors) == 0:
            return candidates[0]

        neighbors_x = neighbors[:, 0]
        neighbors_y = neighbors[:, 1]
        neighbors_r = neighbors[:, 2]

        for xyr_i in candidates:

            x_i, y_i, r_i = xyr_i

            dx = neighbors_x - x_i
            dy = neighbors_y - y_i
            sq_distances = np.square(dx) + np.square(dy)

            sep_needed = np.square(neighbors_r + r_i)

            # Good candidate does not overlap any of neighbors which means that
            # squared distance between candidate and any of the neighbors has
            # to be at least square of the summed radii
            good_candidate = np.all(sq_distances >= sep_needed)

            if good_candidate:
                return xyr_i

        raise RuntimeError(
            "No non-overlapping candidates found. This should not happen."
        )

    def add_gutters(self, points, center, log_scale=False):
        """Stop points from extending beyond their territory."""
        half_width = self.width / 2
        if log_scale:
            low_gutter = 10 ** (np.log10(center) - half_width)
        else:
            low_gutter = center - half_width
        off_low = points < low_gutter
        if off_low.any():
            points[off_low] = low_gutter
        if log_scale:
            high_gutter = 10 ** (np.log10(center) + half_width)
        else:
            high_gutter = center + half_width
        off_high = points > high_gutter
        if off_high.any():
            points[off_high] = high_gutter

        gutter_prop = (off_high + off_low).sum() / len(points)
        if gutter_prop > self.warn_thresh:
            msg = (
                "{:.1%} of the points cannot be placed; you may want "
                "to decrease the size of the markers or use stripplot."
            ).format(gutter_prop)
            warnings.warn(msg, UserWarning)

        return points


def _get_swarm_colors(*, ax, features, palette, show_legend, hue_order):

    cbar_kwargs = dict(location = 'left', pad = 0.01, shrink = 0.25, aspect = 15, anchor = (0, 0.5))
    legend_kwargs = dict(loc="upper left", markerscale = 1, frameon = False, title_fontsize='x-large', fontsize='large',
                    bbox_to_anchor=(1.05, 0.5))

    colors = map_colors(ax, features, palette, 
                add_legend = show_legend, hue_order = hue_order, 
                cbar_kwargs = cbar_kwargs,
                legend_kwargs = legend_kwargs)

    return colors


def _plot_swarm_segment(is_leaf = False, centerline = 0, palette = 'inferno', feature_labels = None, linecolor = 'black', 
    linewidth = 0.1, hue_order = None, show_legend = True, size = 15, is_root = True, max_swarm_density = 2000,
        color = 'black', min_pseudotime = 0., max_bar_height = 0.5,*, ax, features, pseudotime, cell_colors, **kwargs,):
    
    features = np.ravel(features)
    assert(len(features) == len(pseudotime))

    swarm_density = len(features)/(pseudotime.max() - pseudotime.min())

    if swarm_density > max_swarm_density:
        downsample_rate = max_swarm_density/swarm_density
        downsample_mask = np.random.rand(len(features)) < downsample_rate

        features = features[downsample_mask]
        pseudotime = pseudotime[downsample_mask]
        cell_colors = cell_colors[downsample_mask]
    
    plot_order = features.argsort()
    points = ax.scatter(
        pseudotime[plot_order], 
        np.ones_like(pseudotime) * centerline, 
        s = size, 
        c = cell_colors[plot_order],
        edgecolors = linecolor,
        linewidths = linewidth,
    )

    Beeswarm(orient='h', width = max_bar_height)(points, centerline)
    ax.axis('off')