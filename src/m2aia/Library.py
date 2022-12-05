import ctypes
import ctypes.util
import pathlib
import platform
import os
import logging


def load_library_dependencies_recursively(path):
    """ Load required M2aia libraries recursively
    """

    while True:
        try:
            lib = ctypes.cdll.LoadLibrary(path)
            return lib
        except Exception as e:
            missing_lib = str(e).split(':')[0].strip()
            new_libpath = str(pathlib.Path(os.environ["M2AIA_PATH"]).joinpath(f"{missing_lib}"))
            load_library_dependencies_recursively(new_libpath)


def load_m2aia_library():
    search_path = pathlib.Path(os.environ["M2AIA_PATH"])
    
    # Load custom M2aia libraries
    if "Linux" in platform.platform():
        path = str(search_path.joinpath("MitkCore","libM2aiaCoreIO.so"))
        return load_library_dependencies_recursively(path)

    if "Windows" in platform.platform():
        os.add_dll_directory(search_path)
        os.add_dll_directory(search_path.joinpath("MitkCore"))
        return ctypes.cdll.LoadLibrary("M2aiaCoreIO.dll")

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
