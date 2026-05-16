"""Shared colors for signed response heatmap rendering.

Inputs: normalized response-map values in the range -1 to 1.
Outputs: a matplotlib colormap used consistently by live Qt rendering and
exported figures.

The diverging blue-black-orange scale avoids red/green contrasts while keeping
negative inhibition and positive activation visually distinct. Zero is black so
the neutral point is visible on the colorbar without introducing a white band.
"""

from matplotlib.colors import LinearSegmentedColormap

__all__ = ["RESPONSE_HEATMAP_COLORMAP"]

RESPONSE_HEATMAP_COLORMAP = LinearSegmentedColormap.from_list(
    "twopy_response_blue_orange",
    (
        (0.0, "#2c7bb6"),
        (0.50, "#000000"),
        (1.0, "#d95f02"),
    ),
    N=257,
)
