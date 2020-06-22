"""Plot functions for the profiling report."""

from typing import Optional, Union

import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
from sklearn.decomposition import PCA

from pandas_profiling.config import config
from pandas_profiling.utils.resources import get_resource
from pandas_profiling.visualisation.utils import hex_to_rgb, plot_360_n0sc0pe


register_matplotlib_converters()
matplotlib.style.use(str(get_resource("styles/pandas_profiling.mplstyle")))
sns.set_style(style="white")

def _plot_boxplot(
        series: np.ndarray,
        series_description: dict,
        figsize: tuple = (6, 4),
    ):
    """Plot a boxplot from the data and return the AxesSubplot object.

    Args:
        series: The data to plot
        figsize: The size of the figure (width, height) in inches, default (6,4)

    Returns:
        The boxplot.


    """
    fig = plt.figure(figsize=figsize)
    plot = fig.add_subplot(111)
    plot.boxplot(
        series, meanline=True,
    )
    plot.set_ylabel("Value")
    plot.set_xticklabels([])
    return plot


def boxplot(series: np.ndarray, series_description: dict) -> str:
    """Plot a boxplot of the data.

    Args:
      series: The data to plot.
      series_description:

    Returns:
      The resulting boxplot encoded as a string.

    """
    with matplotlib.style.context(["seaborn-ticks", str(get_resource("styles/pandas_profiling_frame.mplstyle"))]):
        plot = _plot_boxplot(series, series_description)
        plot.figure.tight_layout()
        return plot_360_n0sc0pe(plt)


def _plot_clustermap(
        data: pd.DataFrame,
        figsize: tuple = (6, 6),
    ):
    """Plot a clustermap from the data and return the AxesSubplot object.

    Args:
        data: The data to plot
        figsize: The size of the figure (width, height) in inches, default (6,4)

    Returns:
        The clustermap.


    """
    fig = plt.figure(figsize=figsize)
    plot = fig.add_subplot(111)
    plot = sns.clustermap(
        data, figsize=figsize
    )
    return plot


def clustermap(data: pd.DataFrame) -> str:
    """Plot a clustermap of the data.

    Args:
      series: The data to plot.

    Returns:
      The resulting clustermap encoded as a string.
      :param data:

    """
    with matplotlib.style.context(["seaborn-ticks", str(get_resource("styles/pandas_profiling_frame.mplstyle"))]):
        plot = _plot_clustermap(data)
        return plot_360_n0sc0pe(plt)


def _plot_histogram(
    series: np.ndarray,
    series_description: dict,
    bins: Union[int, np.ndarray],
    figsize: tuple = (6, 4),
):
    """Plot an histogram from the data and return the AxesSubplot object.

    Args:
        series: The data to plot
        figsize: The size of the figure (width, height) in inches, default (6,4)
        bins: number of bins (int for equal size, ndarray for variable size)

    Returns:
        The histogram plot.


    """
    fig = plt.figure(figsize=figsize)
    plot = fig.add_subplot(111)
    plot.set_ylabel("Frequency")
    plot.hist(
        series, facecolor=config["html"]["style"]["primary_color"].get(str), bins=bins,
    )
    return plot


def histogram(
    series: np.ndarray, series_description: dict, bins: Union[int, np.ndarray]
) -> str:
    """Plot an histogram of the data.

    Args:
      series: The data to plot.
      series_description:
      bins: number of bins (int for equal size, ndarray for variable size)

    Returns:
      The resulting histogram encoded as a string.

    """
    with matplotlib.style.context(["seaborn-ticks", str(get_resource("styles/pandas_profiling_frame.mplstyle"))]):
        plot = _plot_histogram(series, series_description, bins)
        plot.xaxis.set_tick_params(rotation=45)
        plot.figure.tight_layout()
        return plot_360_n0sc0pe(plt)


def mini_histogram(
    series: np.ndarray, series_description: dict, bins: Union[int, np.ndarray]
) -> str:
    """Plot a small (mini) histogram of the data.

    Args:
      series: The data to plot.
      series_description:
      bins: number of bins (int for equal size, ndarray for variable size)

    Returns:
      The resulting mini histogram encoded as a string.
    """
    plot = _plot_histogram(series, series_description, bins, figsize=(2, 1.5))
    plot.axes.get_yaxis().set_visible(False)
    plot.set_facecolor("w")

    xticks = plot.xaxis.get_major_ticks()
    for tick in xticks:
        tick.label1.set_fontsize(8)
    plot.xaxis.set_tick_params(rotation=45)
    plot.figure.tight_layout()

    return plot_360_n0sc0pe(plt)


def get_cmap_half(cmap):
    """Get the upper half of the color map

    Args:
        cmap: the color map

    Returns:
        A new color map based on the upper half of another color map

    References:
        https://stackoverflow.com/a/24746399/470433
    """
    # Evaluate an existing colormap from 0.5 (midpoint) to 1 (upper end)
    colors = cmap(np.linspace(0.5, 1, cmap.N // 2))

    # Create a new colormap from those colors
    return LinearSegmentedColormap.from_list("cmap_half", colors)


def get_correlation_font_size(n_labels) -> Optional[int]:
    """Dynamic label font sizes in correlation plots

    Args:
        n_labels: the number of labels

    Returns:
        A font size or None for the default font size
    """
    if n_labels > 100:
        font_size = 4
    elif n_labels > 80:
        font_size = 5
    elif n_labels > 50:
        font_size = 6
    elif n_labels > 40:
        font_size = 8
    else:
        return None
    return font_size


def correlation_matrix(data: pd.DataFrame, vmin: int = -1) -> str:
    """Plot image of a matrix correlation.

    Args:
      data: The matrix correlation to plot.
      vmin: Minimum value of value range.

    Returns:
      The resulting correlation matrix encoded as a string.
    """
    with matplotlib.style.context(["seaborn-ticks", str(get_resource("styles/pandas_profiling_frame.mplstyle"))]):
        fig_cor, axes_cor = plt.subplots()
        cmap_name = config["plot"]["correlation"]["cmap"].get(str)
        cmap_bad = config["plot"]["correlation"]["bad"].get(str)

        cmap = plt.get_cmap(cmap_name)
        if vmin == 0:
            cmap = get_cmap_half(cmap)
        cmap.set_bad(cmap_bad)

        labels = data.columns
        matrix_image = axes_cor.imshow(
            data, vmin=vmin, vmax=1, interpolation="nearest", cmap=cmap
        )
        cbar = plt.colorbar(matrix_image)
        cbar.outline.set_visible(False)

        if data.isnull().values.any():
            legend_elements = [Patch(facecolor=cmap(np.nan), label="invalid\ncoefficient")]

            plt.legend(
                handles=legend_elements, loc="upper right", handleheight=2.5,
            )

        axes_cor.set_xticks(np.arange(0, data.shape[0], float(data.shape[0]) / len(labels)))
        axes_cor.set_yticks(np.arange(0, data.shape[1], float(data.shape[1]) / len(labels)))

        font_size = get_correlation_font_size(len(labels))
        axes_cor.set_xticklabels(labels, rotation=90, fontsize=font_size)
        axes_cor.set_yticklabels(labels, fontsize=font_size)
        plt.subplots_adjust(bottom=0.2)
        return plot_360_n0sc0pe(plt)


def get_predictivity_font_size(data):
    """Calculate font size based on number of columns

    Args:
        data: DataFrame

    Returns:
        Font size for predictivity plots.
    """
    max_label_length = max([len(label) for label in data.columns])

    if len(data.columns) < 20:
        font_size = 13
    elif 20 <= len(data.columns) < 40:
        font_size = 12
    elif 40 <= len(data.columns) < 60:
        font_size = 10
    else:
        font_size = 8

    font_size *= min(1.0, 20.0 / max_label_length)
    return font_size


def predictivity(data: pd.DataFrame) -> str:
    """Plot image of a matrix correlation.

    Args:
      data: The matrix correlation to plot.

    Returns:
      The resulting predictivity plot encoded as a string.
    """
    with matplotlib.style.context(["seaborn-ticks", str(get_resource("styles/pandas_profiling_frame.mplstyle"))]):
        target_variables = config["correlations"]["targets"].get()
        if len(target_variables) == 0:
            target_variables = list(data.select_dtypes(include=np.number).columns)
        palette = sns.color_palette().as_hex()
        tmp = palette[3]
        palette[3] = palette[1]
        palette[1] = tmp

        fig_pred, axes_pred = plt.subplots()
        axes_pred.set_ylim(0, 100)

        # Rescale in range [0, 100] for better visualization
        predictivity = (100 * data[target_variables].round(2).abs()).astype(int)

        # Barplot predictivity
        predictivity.plot.bar(
            figsize=(10, 6), width=0.8, legend=True, fontsize=get_predictivity_font_size(predictivity), rot=45, ax=axes_pred, color=palette)
        for patch in axes_pred.patches:
            axes_pred.annotate(patch.get_height(), (patch.get_x() + patch.get_width() / 2., 100),
                               ha="center", va="center", xytext=(0, 15), textcoords="offset points", rotation=45)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.2)
        return plot_360_n0sc0pe(plt)


def scatter_complex(series: pd.Series) -> str:
    """Scatter plot (or hexbin plot) from a series of complex values

    Examples:
        >>> complex_series = pd.Series([complex(1, 3), complex(3, 1)])
        >>> scatter_complex(complex_series)

    Args:
        series: the Series

    Returns:
        A string containing (a reference to) the image
    """
    with matplotlib.style.context(["seaborn-ticks", str(get_resource("styles/pandas_profiling_frame.mplstyle"))]):
        plt.ylabel("Imaginary")
        plt.xlabel("Real")

        color = config["html"]["style"]["primary_color"].get(str)
        scatter_threshold = config["plot"]["scatter_threshold"].get(int)

        if len(series) > scatter_threshold:
            cmap = sns.light_palette(color, as_cmap=True)
            plt.hexbin(series.real, series.imag, cmap=cmap)
        else:
            plt.scatter(series.real, series.imag, color=color)

        return plot_360_n0sc0pe(plt)


def scatter_series(series, x_label="Width", y_label="Height") -> str:
    """Scatter plot (or hexbin plot) from one series of sequences with length 2

    Examples:
        >>> scatter_series(file_sizes, "Width", "Height")

    Args:
        series: the Series
        x_label: the label on the x-axis
        y_label: the label on the y-axis

    Returns:
        A string containing (a reference to) the image
    """
    with matplotlib.style.context(["seaborn-ticks", str(get_resource("styles/pandas_profiling_frame.mplstyle"))]):
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        color = config["html"]["style"]["primary_color"].get(str)
        scatter_threshold = config["plot"]["scatter_threshold"].get(int)

        if len(series) > scatter_threshold:
            cmap = sns.light_palette(color, as_cmap=True)
            plt.hexbin(*zip(*series.tolist()), cmap=cmap)
        else:
            plt.scatter(*zip(*series.tolist()), color=color)
        return plot_360_n0sc0pe(plt)


def scatter_pairwise(series1, series2, x_label, y_label) -> str:
    """Scatter plot (or hexbin plot) from two series

    Examples:
        >>> widths = pd.Series([800, 1024])
        >>> heights = pd.Series([600, 768])
        >>> scatter_series(widths, heights, "Width", "Height")

    Args:
        series1: the series corresponding to the x-axis
        series2: the series corresponding to the y-axis
        x_label: the label on the x-axis
        y_label: the label on the y-axis

    Returns:
        A string containing (a reference to) the image
    """
    with matplotlib.style.context(["seaborn-ticks", str(get_resource("styles/pandas_profiling_frame.mplstyle"))]):
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        color = config["html"]["style"]["primary_color"].get(str)
        scatter_threshold = config["plot"]["scatter_threshold"].get(int)

        if len(series1) > scatter_threshold:
            cmap = sns.light_palette(color, as_cmap=True)
            plt.hexbin(series1.tolist(), series2.tolist(), gridsize=15, cmap=cmap)
        else:
            plt.scatter(series1.tolist(), series2.tolist(), color=color)
        return plot_360_n0sc0pe(plt)


def _plot_dataset(ax, data, labels=None, visualisation=PCA(n_components=2)):
    transformed_data = visualisation.fit_transform(data)

    if visualisation.n_components == 2:
        xs = transformed_data[:, 0]
        ys = transformed_data[:, 1]
        if labels is None:
            sns.scatterplot(xs, ys, ax=ax)
            return
        for label in np.unique(labels):
            indices = np.where(labels == label)
            x = xs[indices]
            y = ys[indices]
            sns.scatterplot(x, y, label=label, ax=ax)

    elif visualisation.n_components == 3:
        xs = transformed_data[:, 0]
        ys = transformed_data[:, 1]
        zs = transformed_data[:, 2]
        if labels is None:
            ax.scatter(xs, ys, zs)
            return
        for label in np.unique(labels):
            indices = np.where(labels == label)
            x = xs[indices]
            y = ys[indices]
            z = zs[indices]
            ax.scatter(x, y, z, label=label)

    else:
        raise ValueError(f"Invalid number of components: {dimensionality_reducer.n_components}")


def scatter_dataset(data: pd.DataFrame, labels=None, visualisation=PCA(random_state=0), n_components=2, figsize=(6.5, 6.5)) -> str:
    """Generate scatter plot of the whole dataset

    Args:
      data: Pandas DataFrame to generate scatter plot from.
      visualisation: visualisation technique.
      n_components: number of components.

    Returns:
      The resulting scatter plot encoded as a string.
      :param labels:
      :param figsize:
    """
    with matplotlib.style.context(["seaborn-ticks", str(get_resource("styles/pandas_profiling_frame.mplstyle"))]):
        fig = plt.figure(figsize=figsize if n_components == 2 else (figsize[0] + 1, figsize[1] + 1))
        plot = fig.add_subplot(111)
        if n_components == 3:
            plot = fig.add_subplot(111, projection="3d")
        plot.set_xlabel("x")
        plot.set_ylabel("y")
        if n_components == 3:
            plot.set_zlabel("z")
        visualisation.n_components = n_components
        _plot_dataset(
            plot,
            data,
            labels,
            visualisation
        )
        plt.subplots_adjust(bottom=0.2)
        return plot_360_n0sc0pe(plt)
