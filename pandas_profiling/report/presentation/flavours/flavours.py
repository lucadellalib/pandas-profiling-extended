from typing import Dict, Type

from pandas_profiling.report.presentation.core.renderable import Renderable


def apply_renderable_mapping(mapping, structure, flavour):
    for key, value in mapping.items():
        if isinstance(structure, key):
            value.convert_to_class(structure, flavour)


def get_html_renderable_mapping() -> Dict[Type[Renderable], Type[Renderable]]:
    """Workaround variable type annotations not being supported in Python 3.5

    Returns:
        type annotated mapping dict
    """
    from pandas_profiling.report.presentation.flavours.html import (
        HTMLCollapse,
        HTMLContainer,
        HTMLDuplicate,
        HTMLFrequencyTable,
        HTMLFrequencyTableSmall,
        HTMLHTML,
        HTMLImage,
        HTMLRoot,
        HTMLSample,
        HTMLTable,
        HTMLToggleButton,
        HTMLVariable,
        HTMLVariableInfo,
        HTMLWarnings,
    )
    from pandas_profiling.report.presentation.core import (
        Collapse,
        Container,
        Duplicate,
        FrequencyTable,
        FrequencyTableSmall,
        HTML,
        Image,
        Root,
        Sample,
        Table,
        ToggleButton,
        Variable,
        VariableInfo,
        Warnings,
    )

    return {
        Collapse: HTMLCollapse,
        Container: HTMLContainer,
        Duplicate: HTMLDuplicate,
        FrequencyTable: HTMLFrequencyTable,
        FrequencyTableSmall: HTMLFrequencyTableSmall,
        HTML: HTMLHTML,
        Image: HTMLImage,
        Root: HTMLRoot,
        Sample: HTMLSample,
        Table: HTMLTable,
        ToggleButton: HTMLToggleButton,
        Variable: HTMLVariable,
        VariableInfo: HTMLVariableInfo,
        Warnings: HTMLWarnings,
    }


def HTMLReport(structure: Renderable):
    """Adds HTML flavour to Renderable

    Args:
        structure:

    Returns:

    """
    mapping = get_html_renderable_mapping()
    apply_renderable_mapping(mapping, structure, flavour=HTMLReport)
    return structure


def get_widget_renderable_mapping() -> Dict[Type[Renderable], Type[Renderable]]:
    from pandas_profiling.report.presentation.flavours.widget import (
        WidgetCollapse,
        WidgetContainer,
        WidgetDuplicate,
        WidgetFrequencyTable,
        WidgetFrequencyTableSmall,
        WidgetHTML,
        WidgetImage,
        WidgetRoot,
        WidgetSample,
        WidgetTable,
        WidgetToggleButton,
        WidgetVariable,
        WidgetVariableInfo,
        WidgetWarnings,
    )
    from pandas_profiling.report.presentation.core import (
        Collapse,
        Container,
        Duplicate,
        FrequencyTable,
        FrequencyTableSmall,
        HTML,
        Image,
        Root,
        Sample,
        Table,
        ToggleButton,
        Variable,
        VariableInfo,
        Warnings,
    )

    return {
        Collapse: WidgetCollapse,
        Container: WidgetContainer,
        Duplicate: WidgetDuplicate,
        FrequencyTable: WidgetFrequencyTable,
        FrequencyTableSmall: WidgetFrequencyTableSmall,
        HTML: WidgetHTML,
        Image: WidgetImage,
        Root: WidgetRoot,
        Sample: WidgetSample,
        Table: WidgetTable,
        ToggleButton: WidgetToggleButton,
        Variable: WidgetVariable,
        VariableInfo: WidgetVariableInfo,
        Warnings: WidgetWarnings,
    }


def WidgetReport(structure: Renderable):
    mapping = get_widget_renderable_mapping()
    apply_renderable_mapping(mapping, structure, flavour=WidgetReport)
    return structure


def get_qt_renderable_mapping() -> Dict[Type[Renderable], Type[Renderable]]:
    from pandas_profiling.report.presentation.flavours.qt import (
        QtCollapse,
        QtContainer,
        QtDuplicate,
        QtFrequencyTable,
        QtFrequencyTableSmall,
        QtHTML,
        QtImage,
        QtRoot,
        QtSample,
        QtTable,
        QtToggleButton,
        QtVariable,
        QtVariableInfo,
        QtWarnings,
    )
    from pandas_profiling.report.presentation.core import (
        Collapse,
        Container,
        Duplicate,
        FrequencyTable,
        FrequencyTableSmall,
        HTML,
        Image,
        Root,
        Sample,
        Table,
        ToggleButton,
        Variable,
        VariableInfo,
        Warnings,
    )

    return {
        Collapse: QtCollapse,
        Container: QtContainer,
        Duplicate: QtDuplicate,
        FrequencyTable: QtFrequencyTable,
        FrequencyTableSmall: QtFrequencyTableSmall,
        HTML: QtHTML,
        Image: QtImage,
        Root: QtRoot,
        Sample: QtSample,
        Table: QtTable,
        ToggleButton: QtToggleButton,
        Variable: QtVariable,
        VariableInfo: QtVariableInfo,
        Warnings: QtWarnings,
    }


def QtReport(structure: Renderable):
    mapping = get_qt_renderable_mapping()
    apply_renderable_mapping(mapping, structure, flavour=QtReport)
    return structure
