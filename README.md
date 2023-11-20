## Features

### Data Input
- Support for centroid and profile data stored in imzML format:
    - Range-based ion images
    - Individual spectra
    - imzML metadata
- Gray-scale images (ie. ion images, mask images, normalization images) are accessible as SimpleITK's Image objects.
    - ImzML images and images exported using SimpleITK (e.g. in NRRD format) can be explored interactively in [M²aia](https://m2aia.de) desktop application.

### Data Structures & Functions:
- Signal Processing for profile imzML files:
    - Baseline correction
    - Normalization
    - Smoothing
- Batch generators (TensorFlow/PyTorch):
    - Spectral, spatial, and spatio-spectral access
    - Optional in-memory or on-disk batch/data buffering

### Data Output
- Write Continuous Centroid imzML files

#### Examples@ [https://github.com/m2aia/pym2aia-examples.git](https://github.com/m2aia/pym2aia-examples.git)

## Installation

#### Requirements on Windows

Microsoft Visual C++ Redistributable latest supported downloads:
https://learn.microsoft.com/de-de/cpp/windows/latest-supported-vc-redist?view=msvc-170

#### Requirements on Ubuntu/Linux

```
apt-get install -q -y --no-install-recommends \
    libglu1-mesa-dev \
    libgomp1 \
    libopenslide-dev \
    python3 \
    python3-pip \
    python3-venv
```

#### Package installation 
Consider to install pyM²aia in an [virtual environment](https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-and-using-virtual-environments).

Install [pyM²aia](https://pypi.org/project/m2aia/) on windows or linux using pip.

```sh
# cd projectDir/
python3 -m venv .venv
pip3 install --upgrade pip

source .venv/bin/activate
# 

pip install m2aia
```

_Note:_ Different versions of the pypi package exists for windows and linux based systems:
 - Linux: m2aia-x.x.x-py3-none-manylinux_2_31_x86_64.whl 
 - Windows: m2aia-x.x.x-py3-none-win-amd64.whl 
 - Source: m2aia-x.x.x.tar.gz

The Source distribution contains *.so and *.dll library files for both, windows and linux. You can try to install pyM²aia by extracting the archive and using the following command within the project folder:
```
python setup.py install
```



## Getting started

pyM2aia offers a interface to [M2aia’s](https://github.com/m2aia/m2aia) imzML reader and signal processing utilities.
Complete processing examples can be found on [pym2aia-examples](https://github.com/m2aia/pym2aia-examples)

Example create ion images:

```python
import m2aia as m2

I = m2.ImzMLReader("path/to/imzMl/file.imzML")
I.SetNormalization(m2.m2NormalizationTIC)
I.SetIntensityTransformation(m2.m2IntensityTransformationSquareRoot)
ys_2 = I.GetMeanSpectrum()
# get the ion image as array
center_mz = 1123.43
i_2 = I.GetArray(center_mz, 75)
```


Example write continuous centroid imzML:
```python
import m2aia as m2
import numpy as np

# load a coninuous profile imzML
I = m2.ImzMLReader("path/to/imzMl/file.imzML")

# Find/load centroids ...
# centroids = [x for x in range(2000,3000,20)]

# csv with i.e. 3 columns with header line ('mz','min','max')
centroids = np.genfromtxt("path/to/csv/centroids.csv", delimiter=',', skip_header=1)[:,0]

I.SetTolerance(50)
I.WriteContinuousCentroidImzML("/tmp/continuous_centroid.imzML", centroids)
```

[Developer Documentation](https://data.jtfc.de/pym2aia/sphinx-build/html/m2aia.html#module-m2aia.ImageIO)



### Known issues

The following Warnings can be ignored:
- "WARNING: In AutoLoadModulesFromPath at /opt/mitk/Modules/CppMicroServices/core/src/util/usUtils.cpp:176 : Auto-loading of module /usr/local/lib/python3.10/dist-packages/m2aia/binaries/bin/MitkCore/libMitk\<XY\>IO.so failed."

- "WARNING: In load_impl at /opt/mitk/Modules/CppMicroServices/core/src/util/usUtils.cpp:76 : libMitk\<XY\>.so: cannot open shared object file: No such file or directory"

## Development

### New python features for MSI processing
Python side processing utilities are currently limited. To change this, you can add new Python based features by adding new utilities.
New tools should focus on the utilization of pyM²aia's data access and structures.

```
./utils/<toolName>.py
```

_To promote your work you can just create a pull request or contact me personally._


### Improve the C++-Python interface
To Further enhance the C++-Python interface it is required to make changes in [m2PythonWrapper.cpp](https://github.com/m2aia/m2aia/blob/develop/Modules/M2aiaCore/src/IO/m2PythonWrapper.cpp). By using MITK's super build approach, a development environment can be setup following [these instructions](https://m2aia.de/development.html). It is required to change the library search path for pyM2aia to the modified libraries. This can be realized by setting a system environment variable. Following the example shown in the above mentioned instructions, ie. it is required to set it in windows to 
```
M2AIA_PATH=C:/M2aiaWorkDir/build/MITK-build/lib
```

For linux you can add the following lines to the end of your ~/.profile file:

```
M2AIA_PATH=/home/username/M2aiaWorkDir/build/MITK-build/lib
```

All pyM2aia libraries on the system will now use this custom library search path.

_To promote your work you can just create a pull requests in both repositories or contact me personally._


# Cite M²aia

M²aia is an interactive desktop application for visualization and processing of mass spectrometry (MS) imaging data. Please visit [https://m2aia.github.io/m2aia](https://m2aia.github.io/m2aia) for further information.

Cordes J; Enzlein T; Marsching C; Hinze M; Engelhardt S; Hopf C; Wolf I (July, 2021): M²aia - Interactive, fast and memory efficient analysis of 2D and 3D multi-modal mass spectrometry imaging data https://doi.org/10.1093/gigascience/giab049

[![DOI](https://zenodo.org/badge/554270311.svg)](https://zenodo.org/badge/latestdoi/554270311)
