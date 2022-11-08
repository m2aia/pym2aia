import ctypes
import ctypes.util
import pathlib
import platform
import os

def get_library_name():
    if "Linux" in platform.platform():
        p = ctypes.util.find_library("libM2aiaCoreIO.so")
        print(p)
        return "libM2aiaCoreIO.so"
    else:
        # M2aiaCoreIO is in InstallDir/bin/MitkCore/
        p = pathlib.Path(ctypes.util.find_library("M2aiaCoreIO"))
        if p:
            os.add_dll_directory(p.parent)
            os.add_dll_directory(p.parent.parent)
        return "M2aiaCoreIO.dll"

def load_m2aia_library():
    target_lib = get_library_name()
    try:
        # print("Load library:", target_lib)
        return ctypes.cdll.LoadLibrary(target_lib)
    except:
        raise ImportError(
"""Could not find the required M2aia libraries.
pyM2aia requires a valid M2aia installation/build. 
Go to https://m2aia.de and download the latest version of M2aia. 
Then, follow the setup procedure for pyM2aia on https://github.com/m2aia/pym2aia.
"""
                        ,name="m2aia")
