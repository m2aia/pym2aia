import ctypes
import ctypes.util
import pathlib
import platform
import os
import subprocess
import sys
from typing import List, Set

def get_shared_lib_dependencies(so_file_path):
    try:
        ldd_output = subprocess.check_output(['ldd', so_file_path], stderr=subprocess.STDOUT, universal_newlines=True)
        # print("ldd_output",ldd_output)
        dependencies = {line.split(' => ')[0].strip() for line in ldd_output.splitlines() if "not found" in line}
        return dependencies
    except subprocess.CalledProcessError:
        return {}

def load_library_dependencies_recursively(search_path : pathlib.Path, library_name: str, dependencies: List):
    """ Load required M2aia libraries recursively
    """
    lib_path = str(search_path.joinpath(library_name))
    lib_missing_dependencies = get_shared_lib_dependencies(lib_path)
    
    try:
        ctypes.cdll.LoadLibrary(lib_path)
        if lib_path not in dependencies:
            dependencies.append(lib_path)
        return
    except:
        while lib_missing_dependencies:
            lib_working = lib_missing_dependencies.pop()
            load_library_dependencies_recursively(search_path, lib_working, dependencies)
        
    ctypes.cdll.LoadLibrary(lib_path)
    if lib_path not in dependencies:
        dependencies.append(lib_path)
        

def load_m2aia_library():
    search_path = pathlib.Path(os.environ["M2AIA_PATH"])
    target_library_path_parts = ["libM2aiaCore.so"]
    
    if "Windows" in platform.platform():
        os.add_dll_directory(search_path)
        # os.add_dll_directory(search_path.joinpath("MitkCore"))
        return ctypes.cdll.LoadLibrary("M2aiaCore.dll")
    
    else: #"Linux" in platform.platform():

        if "Darwin" in platform.platform():
            raise ImportError("macOS/Darwin based systems are currently not tested.")

        dependencies = []
        for lib_name in target_library_path_parts:
            load_library_dependencies_recursively(search_path, lib_name, dependencies)
        
        os.environ["M2AIA_LIBRARIES"] = ';'.join(dependencies)       
        for dep in dependencies:
            ctypes.cdll.LoadLibrary(dep)
        
        return ctypes.cdll.LoadLibrary((search_path /  "libM2aiaCore.so").absolute())
    
    


def get_library():
    try:
        return load_m2aia_library()
    except SystemExit:
        pass
    except ImportError as e:
        print(e)
        raise ImportError(
"""Could not find the required M2aia libraries.
pyM2aia requires a valid M2aia installation/build. 
Go to https://m2aia.github.io/m2aia and download the latest version of M2aia. 
Then, follow the setup procedure for pyM2aia on https://github.com/m2aia/pym2aia.
"""
                        ,name="m2aia")
