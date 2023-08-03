import ctypes
import ctypes.util
import pathlib
import platform
import os
import logging
from typing import List

def load_library_dependencies_recursively(search_path : pathlib.Path, library_name: str, dependencies: List):
    """ Load required M2aia libraries recursively
    """
    lib_path = search_path.joinpath(library_name)
    while True:
        try:
            lib = ctypes.cdll.LoadLibrary(str(lib_path))
            dependencies.append(lib_path)
            return lib
        except Exception as e:
            missing_lib_name = str(e).split(':')[0].strip()
            # new_lib_path = str(pathlib.Path().joinpath(f"{missing_lib}"))
            load_library_dependencies_recursively(search_path, missing_lib_name, dependencies)


def load_m2aia_library():
    search_path = pathlib.Path(os.environ["M2AIA_PATH"])

    target_library_path_parts = ["libM2aiaCore.so"]
    # if search_path.joinpath("MitkCore").exists():
    #     target_library_path_parts.append("MitkCore")
    #     target_library_path_parts = target_library_path_parts[::-1]
    
    # Load custom M2aia libraries
    if "Linux" in platform.platform():
        path = str(search_path.joinpath(*target_library_path_parts))
        dependencies = []
        lib = load_library_dependencies_recursively(search_path, path, dependencies)
        os.environ["M2AIA_LIBRARIES"] = ';'.join([str(d) for d in dependencies])
        return lib

    if "Windows" in platform.platform():
        os.add_dll_directory(search_path)
        # os.add_dll_directory(search_path.joinpath("MitkCore"))
        return ctypes.cdll.LoadLibrary("M2aiaCore.dll")

def get_library():
    try:
        return load_m2aia_library()
    except SystemExit:
        pass
    except:
        raise ImportError(
"""Could not find the required M2aia libraries.
pyM2aia requires a valid M2aia installation/build. 
Go to https://m2aia.github.io/m2aia and download the latest version of M2aia. 
Then, follow the setup procedure for pyM2aia on https://github.com/m2aia/pym2aia.
"""
                        ,name="m2aia")
