import pathlib
import os
import wget
import zipfile
import tarfile
import shutil
import sys

pathlib.Path("install_binaries").mkdir(exist_ok=True)

binaries_root=pathlib.Path("src/m2aia/bin/")
shutil.rmtree(binaries_root, ignore_errors=True)
binaries_root.mkdir(exist_ok=True, parents=True)
try:
    linux_archive = pathlib.Path("install_binaries/linux.tar.gz")
    linux_extracted = pathlib.Path("install_binaries/linux")
    if not linux_archive.exists():
        print("Try start download for:",f"https://data.jtfc.de/latest/linux/M2aia-{os.environ['M2AIA_VERSION']}.tar.gz")
        wget.download(f"https://data.jtfc.de/latest/linux/M2aia-{os.environ['M2AIA_VERSION']}.tar.gz", str(linux_archive))
        with tarfile.open(str(linux_archive)) as f:
            f.extractall(str(linux_extracted))
        print("linux_extracted:", linux_extracted)

    # find libraries
    linux_root=list(linux_extracted.glob("*linux*"))[0]
    print("linux_root:", linux_root)
    os.environ["M2AIA_PATH"] = str(linux_root.joinpath("bin").absolute())
    print("M2AIA_PATH:", os.environ["M2AIA_PATH"])
    
    # init module
    import src.m2aia
    invoked_libraries = os.environ["M2AIA_LIBRARIES"]
    print("invoked_libraries:", invoked_libraries)

    linux_bin_root = binaries_root.joinpath('linux')
    linux_bin_root.mkdir(exist_ok=True)
    print("linux_bin_root:", linux_bin_root)

    for d in invoked_libraries.split(';'):
        d = pathlib.Path(d)
        print(str(d), "->", str(linux_bin_root.joinpath(d.name)))
        shutil.copy(str(d),str(linux_bin_root.joinpath(d.name)))

except Exception as e:
    print("Linux binaries not found!", e)

# START WINDOWS
print("=================================================")
try:
    windows_archive = pathlib.Path("install_binaries/windows.zip")
    windows_extracted = pathlib.Path("install_binaries/windows")
    if not windows_archive.exists():
        print("Try start download for:",f"https://data.jtfc.de/latest/windows/M2aia-{os.environ['M2AIA_VERSION']}.zip")
        wget.download(f"https://data.jtfc.de/latest/windows/M2aia-{os.environ['M2AIA_VERSION']}.zip", str(windows_archive))
        with zipfile.ZipFile(str(windows_archive)) as f:
            f.extractall(str(windows_extracted))
    windows_root=list(windows_extracted.glob("*windows*"))[0]

    windows_bin_root = binaries_root.joinpath('windows')
    windows_bin_root.mkdir(exist_ok=True)    

    globs = [f for f in windows_root.joinpath('bin').glob("*.dll")]
    globs.extend([f for f in windows_root.joinpath('bin').glob("MitkCore/*.dll")])
    for lib in globs:
        if "Qt5" in str(lib):
            print("FOUND => ", lib)
            continue
        print(str(lib), "->", str(windows_bin_root.joinpath(lib.name)))
        shutil.copy(str(lib),str(windows_bin_root.joinpath(lib.name)))
except Exception as e:
    print("Windows binaries not found!", e)