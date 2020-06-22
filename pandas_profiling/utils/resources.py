from importlib import resources


def get_resource(resource_path):
    path_resource = resource_path.rsplit("/", 1)
    package_name = "pandas_profiling.resources." + path_resource[0].replace("/", ".")
    resource_name = path_resource[1]
    with resources.path(package_name, resource_name) as path:
        return path
