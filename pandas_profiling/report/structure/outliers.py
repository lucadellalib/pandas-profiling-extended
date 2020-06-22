from typing import List, Optional

import numpy as np
# Necessary for reading configuration file
from sklearn import decomposition, manifold

from pandas_profiling.config import config
from pandas_profiling.report.presentation.core import Container, Image
from pandas_profiling.report.presentation.core.renderable import Renderable
from pandas_profiling.visualisation.plot import scatter_dataset


def get_outlier_items(summary) -> Optional[Renderable]:
    """Create the list of outlier items

    Args:
        summary: dict of outliers

    Returns:
        List of outlier items to show in the interface.
    """
    items: List[Renderable] = []

    image_format = config["plot"]["image_format"].get(str)
    for key, outlier in summary["outliers"].items():
        data = outlier.drop(["Outlier"], axis=1)
        labels = np.array(list(map(lambda x: "inlier" if x == 1 else "outlier", outlier["Outlier"])))
        visualisation_items = []
        for name, visualisation in config["outliers"]["visualisations"].get():
            n_components_items = []
            for n_components in [2,3]:
                diagram = Image(
                    scatter_dataset(data, labels, visualisation=eval(visualisation), n_components=n_components),
                    image_format=image_format,
                    alt="{n_components}D",
                    anchor_id=f"{key}_{name}_{n_components}D_diagram",
                    name=f"{n_components}D",
                    classes=f"{key}-{name}-{n_components}D-diagram",
                )
                n_components_items.append(diagram)

            visualisation_items.append(
                Container(
                    n_components_items,
                    sequence_type="tabs",
                    name=name,
                    anchor_id=f"{key}_{name}_tab",
                )
            )

        items.append(
            Container(
                visualisation_items,
                sequence_type="tabs",
                name=key,
                anchor_id=f"{key}_tab",
            )
        )
    return items
