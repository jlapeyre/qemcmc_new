from pathlib import Path
from importlib.resources import files as _files


# Return the directory of the package root.
def pkg_root_dir():
    return _files("qemcmc").parents[1]


# Return the path to the assets directory
def assets_dir():
    return pkg_root_dir().joinpath("assets")


# Return the path to the assets/graph directory
def assets_graph_dir():
    return pkg_root_dir().joinpath("assets", "graphs")


def assets_graph_file(fname):
    return Path(assets_graph_dir()) / fname
