import wget
import pathlib
import zipfile
import tarfile
import shutil
import sys

pathlib.Path("install_binaries").mkdir(exist_ok=True)
binaries_root=pathlib.Path("src/m2aia/binaries")
binaries_root.mkdir(exist_ok=True)


if "linux" in sys.argv[1]:
    linux_archive = pathlib.Path("install_binaries/linux.tar.gz")
    linux_extracted = pathlib.Path("install_binaries/linux")
    if not linux_archive.exists():
        wget.download("https://data.jtfc.de/latest/ubuntu20_04/M2aia-latest.tar.gz", str(linux_archive))
        with tarfile.open(str(linux_archive)) as f:
            f.extractall(str(linux_extracted))
    linux_root=list(linux_extracted.glob("*linux*"))[0]
    shutil.copytree(linux_root, binaries_root, dirs_exist_ok=True)


if "win" in sys.argv[1]:
    windows_archive = pathlib.Path("install_binaries/windows.zip")
    windows_extracted = pathlib.Path("install_binaries/windows")
    if not windows_archive.exists():
        wget.download("https://data.jtfc.de/latest/windows/M2aia-2022.10.00-windows-x86_64.zip", str(windows_archive))
        with zipfile.ZipFile(str(windows_archive)) as f:
            f.extractall(str(windows_extracted))
    windows_root=list(windows_extracted.glob("*windows*"))[0]
    shutil.copytree(windows_root, binaries_root, dirs_exist_ok=True)



