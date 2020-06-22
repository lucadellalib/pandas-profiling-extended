from typing import List, Optional

import numpy as np
# Necessary for reading configuration file
from sklearn import decomposition, manifold

from pandas_profiling.config import config
from pandas_profiling.report.presentation.core import Container, Image
from pandas_profiling.report.presentation.core.renderable import Renderable
from pandas_profiling.visualisation.plot import clustermap, scatter_dataset


def get_cluster_items(summary) -> Optional[Renderable]:
    """Create the list of cluster items

    Args:
        summary: dict of clusters

    Returns:
        List of cluster items to show in the interface.
    """
    items: List[Renderable] = []

    image_format = config["plot"]["image_format"].get(str)
    for key, cluster in summary["clusters"].items():
        data = cluster.drop(["Cluster"], axis=1)
        labels = np.array(list(map(lambda x: "noise" if x == -1 else f"cluster {x}", cluster["Cluster"])))
        visualisation_items = []
        for name, visualisation in config["clusters"]["visualisations"].get():
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
    data = list(summary["clusters"].values())[0].drop(["Cluster"], axis=1)
    # Clustermap
    if config["clusters"]["clustermap"].get(bool):
        items.append(
            Image(
                clustermap(data[list(data.select_dtypes(include=np.number).columns)]),
                image_format=image_format,
                alt="Clustermap",
                anchor_id=f"clustermap_diagram",
                name=f"Clustermap",
                classes=f"clustermap-diagram",
            )
        )
    return items
