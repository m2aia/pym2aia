import pathlib
import os
import wget
import zipfile
import tarfile
import shutil
import argparse
import sys 

def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--target", default="install_binaries")
    parser.add_argument("-l","--linux-archive", default="linux.tar.gz")
    parser.add_argument("-w","--windows-archive", default="windows.zip")
    parser.add_argument("-v","--version",default="M2aia-v2023.10")
    parser.add_argument("-d","--download", action="store_true")
    parser.add_argument("--linux", action="store_true")
    parser.add_argument("--windows", action="store_true")
    args = parser.parse_args()

    binaries_root=pathlib.Path("src/m2aia/bin/")
    # clean binaries
    shutil.rmtree(binaries_root, ignore_errors=True)
    binaries_root.mkdir(exist_ok=True)
    
    # ---------------------------------------------------------------
    if args.linux:
        linux_archive = pathlib.Path(args.linux_archive)
        if args.download:
            print("Try start download for:",f"https://data.jtfc.de/latest/linux/M2aia-{args.version}.tar.gz")
            if linux_archive.exists():
                os.remove(linux_archive)
            wget.download(f"https://data.jtfc.de/latest/linux/M2aia-{args.version}.tar.gz", str(linux_archive))

        if linux_archive.exists():
            linux_extracted = pathlib.Path(args.target) / pathlib.Path("linux")
            shutil.rmtree(linux_extracted, ignore_errors=True)
            with tarfile.open(str(linux_archive)) as f:
                f.extractall(str(linux_extracted))
            linux_root=list(linux_extracted.glob("M2aia*"))[0]
            os.environ["M2AIA_PATH"] = str(linux_root.joinpath("bin").absolute())
            # init module will identify all required libraries to load libM2aiaCore.so
            import src.m2aia

            invoked_libraries = os.environ["M2AIA_LIBRARIES"]

            # COPY dependent libraries to the installer location
            for d in invoked_libraries.split(';'):
                d = pathlib.Path(d)
                print(str(d), "->", str(binaries_root.joinpath(d.name)))
                shutil.copy(str(d),str(binaries_root.joinpath(d.name)))
        else:
            print("Linux Archive not found!")
            
    # ---------------------------------------------------------------
    if args.windows:
        windows_archive = pathlib.Path(args.windows_archive)
        if args.download:
            print("Try start download for:",f"https://data.jtfc.de/latest/windows/M2aia-{args.version}.zip")
            if windows_archive.exists():
                os.remove(windows_archive)
            wget.download(f"https://data.jtfc.de/latest/windows/M2aia-{args.version}.zip", str(windows_archive))

        if windows_archive.exists():
            windows_extracted = pathlib.Path(args.target) / pathlib.Path("windows")
            shutil.rmtree(windows_extracted, ignore_errors=True)
            with zipfile.ZipFile(str(windows_archive)) as f:
                f.extractall(str(windows_extracted))
            windows_root=list(windows_extracted.glob("M2aia*"))[0]
            os.environ["M2AIA_PATH"] = str(windows_root.joinpath("bin").absolute())
            
            globs = [f for f in windows_root.joinpath('bin').glob("*.dll")]
            for lib in globs:
                if "Qt5" in str(lib):
                    print("FOUND => ", lib)
                    continue
                print(str(lib), "->", str(binaries_root.joinpath(lib.name)))
                shutil.copy(str(lib),str(binaries_root.joinpath(lib.name)))
        else:
            print("Windows Archive not found!")
            


if __name__ == '__main__':
    prepare()