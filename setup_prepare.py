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
    parser.add_argument("-v","--version",default="v2023.08-alpha")
    parser.add_argument("-d","--download", action="store_true")
    parser.add_argument("--linux", action="store_true")
    parser.add_argument("--windows", action="store_true")
    args = parser.parse_args()

    # prepare source tree
    binaries_root=pathlib.Path("src/m2aia/bin/")
    # clean binaries
    shutil.rmtree(binaries_root, ignore_errors=True)
    # recreate
    binaries_root.mkdir(exist_ok=True, parents=True)
    
    # ---------------------------------------------------------------
    if args.linux:
        linux_archive = pathlib.Path(args.linux_archive)
        if args.download:
            print("Try start download for:",f"https://data.jtfc.de/latest/linux/M2aia-{args.version}.tar.gz")
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
            linux_bin_root = binaries_root.joinpath('linux')
            linux_bin_root.mkdir(exist_ok=True)

            # COPY dependent libraries to the installer location
            for d in invoked_libraries.split(';'):
                d = pathlib.Path(d)
                print(str(d), "->", str(linux_bin_root.joinpath(d.name)))
                shutil.copy(str(d),str(linux_bin_root.joinpath(d.name)))
        else:
            print("Linux Archive not found!")
            
    # ---------------------------------------------------------------
    if args.windows:
        windows_archive = pathlib.Path(args.windows_archive)
        if args.download:
            print("Try start download for:",f"https://data.jtfc.de/latest/windows/M2aia-{args.version}.zip")
            wget.download(f"https://data.jtfc.de/latest/windows/M2aia-{args.version}.zip", str(windows_archive))

        if windows_archive.exists():
            windows_extracted = pathlib.Path(args.target) / pathlib.Path("windows")
            shutil.rmtree(windows_extracted, ignore_errors=True)
            with zipfile.ZipFile(str(windows_archive)) as f:
                f.extractall(str(windows_extracted))
            windows_root=list(windows_extracted.glob("M2aia*"))[0]
            os.environ["M2AIA_PATH"] = str(windows_root.joinpath("bin").absolute())
            windows_bin_root = binaries_root.joinpath('windows')
            windows_bin_root.mkdir(exist_ok=True)

            globs = [f for f in windows_root.joinpath('bin').glob("*.dll")]
            for lib in globs:
                if "Qt5" in str(lib):
                    print("FOUND => ", lib)
                    continue
                print(str(lib), "->", str(windows_bin_root.joinpath(lib.name)))
                shutil.copy(str(lib),str(windows_bin_root.joinpath(lib.name)))
        else:
            print("Windows Archive not found!")
            


if __name__ == '__main__':
    prepare()