"""Organize the calculation of statistics for each series in this DataFrame."""
from datetime import datetime
import warnings

import pandas as pd
# Necessary for reading configuration file
from sklearn import cluster, ensemble, neighbors
from tqdm.auto import tqdm

from pandas_profiling.config import config as config
from pandas_profiling.model.base import Variable
from pandas_profiling.model.correlations import calculate_correlation
from pandas_profiling.model.summary import (
    get_duplicates,
    get_messages,
    get_missing_diagrams,
    get_sample,
    get_scatter_matrix,
    get_series_descriptions,
    get_table_stats,
)
from pandas_profiling.version import VERSION


def describe(title, df: pd.DataFrame) -> dict:
    """Calculate the statistics for each series in this DataFrame.

    Args:
        df: DataFrame.

    Returns:
        This function returns a dictionary containing:
            - table: overall statistics.
            - variables: descriptions per series.
            - correlations: correlation matrices.
            - missing: missing value diagrams.
            - messages: direct special attention to these patterns in your data.
            - package: package details.
            :param title:
    """

    if df is None:
        raise ValueError("Can not describe a `lazy` ProfileReport without a DataFrame.")

    if not isinstance(df, pd.DataFrame):
        warnings.warn("df is not of type pandas.DataFrame")

    if df.empty:
        raise ValueError("df can not be empty")

    disable_progress_bar = not config["progress_bar"].get(bool)

    date_start = datetime.utcnow()

    correlation_names = [
        correlation_name
        for correlation_name in ["pearson", "spearman", "kendall", "phi_k", "cramers",]
        if config["correlations"][correlation_name]["calculate"].get(bool)
    ]

    number_of_tasks = 9 + len(df.columns) + len(correlation_names)

    with tqdm(
        total=number_of_tasks, desc="Summarize dataset", disable=disable_progress_bar
    ) as pbar:
        series_description = get_series_descriptions(df, pbar)

        pbar.set_postfix_str("Get variable types")
        variables = {
            column: description["type"]
            for column, description in series_description.items()
        }
        pbar.update()

        # Transform the series_description in a DataFrame
        pbar.set_postfix_str("Get variable statistics")
        variable_stats = pd.DataFrame(series_description)
        pbar.update()

        # Get correlations
        correlations = {}
        for correlation_name in correlation_names:
            pbar.set_postfix_str(f"Calculate {correlation_name} correlation")
            correlations[correlation_name] = calculate_correlation(
                df, variables, correlation_name
            )
            pbar.update()

        # Make sure correlations is not None
        correlations = {
            key: value for key, value in correlations.items() if value is not None
        }

        # Scatter matrix
        pbar.set_postfix_str("Get scatter matrix")
        scatter_matrix = get_scatter_matrix(df, variables)
        pbar.update()

        # Table statistics
        pbar.set_postfix_str("Get table statistics")
        table_stats = get_table_stats(df, variable_stats)
        pbar.update()

        # Missing diagrams
        pbar.set_postfix_str("Get missing diagrams")
        missing = get_missing_diagrams(df, table_stats)
        pbar.update()

        # Sample
        pbar.set_postfix_str("Take sample")
        sample = get_sample(df)
        pbar.update()

        # Duplicates
        pbar.set_postfix_str("Locating duplicates")
        supported_columns = [
            key
            for key, value in series_description.items()
            if value["type"] != Variable.S_TYPE_UNSUPPORTED
        ]
        duplicates = get_duplicates(df, supported_columns)
        pbar.update()

        # Clusters
        pbar.set_postfix_str("Searching for clusters")
        categoricals = [column_name for column_name, variable_type in variables.items() if variable_type == Variable.TYPE_CAT]
        df_without_missing = df.dropna()
        df_ohe = pd.concat([df_without_missing.drop(categoricals, axis=1), pd.get_dummies(df_without_missing[categoricals])], axis=1).reset_index()
        clusters = {
            name: pd.concat([df_ohe, pd.DataFrame({"Cluster": eval(clustering).fit(df_ohe).labels_})], axis=1)
            for name, clustering in config["clusters"]["clusterings"].get()
        }

        # Outliers
        pbar.set_postfix_str("Detecting outliers")
        outliers = {
            name: pd.concat([df_ohe, pd.DataFrame({"Outlier": eval(detector).fit_predict(df_ohe)})], axis=1)
            for name, detector in config["outliers"]["detectors"].get()
        }

        # Messages
        pbar.set_postfix_str("Get messages/warnings")
        messages = get_messages(table_stats, series_description, correlations)
        pbar.update()

        pbar.set_postfix_str("Get reproduction details")
        package = {
            "pandas_profiling_version": VERSION,
            "pandas_profiling_config": config.dump(),
        }
        pbar.update()

        pbar.set_postfix_str("Completed")

    date_end = datetime.utcnow()

    analysis = {
        "title": title,
        "date_start": date_start,
        "date_end": date_end,
        "duration": date_end - date_start,
    }

    return {
        # Analysis metadata
        "analysis": analysis,
        # Overall dataset description
        "table": table_stats,
        # Per variable descriptions
        "variables": series_description,
        # Bivariate relations
        "scatter": scatter_matrix,
        # Correlation matrices
        "correlations": correlations,
        # Missing values
        "missing": missing,
        # Warnings
        "messages": messages,
        # Package
        "package": package,
        # Sample
        "sample": sample,
        # Duplicates
        "duplicates": duplicates,
        # Clusters
        "clusters": clusters,
        # Outliers
        "outliers": outliers
    }
