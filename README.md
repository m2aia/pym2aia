# Installation

Install pyM2aia in a [virtual environment](https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-and-using-virtual-environments).

_Note: the current version of pyM2aia is currently hosted only for testing on https://test.pypi.org. This will change with the upcoming M2aia Release v2023.08_

``` pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pym2aia ```


# Getting started

pyM2aia offers a interface to [M2aia’s](https://github.com/m2aia/m2aia) imzML reader and signal processing utilities.
Complete processing examples with focus on deep learning can be found on [pym2aia-examples](https://github.com/m2aia/pym2aia-examples)

Example usage:

```
import m2aia as m2

I = m2.ImzMLReader("path/to/imzMl/file.imzML")
I.SetNormalization(m2.m2NormalizationTIC)
I.SetIntensityTransformation(m2.m2IntensityTransformationSquareRoot)
I.Execute()
ys_2 = I.GetMeanSpectrum()
i_2 = I.GetArray(imz, 75)
```

[Developer Documentation](https://data.jtfc.de/pym2aia/sphinx-build/html/m2aia.html#module-m2aia.ImageIO)

## Requirements on Windows

Microsoft Visual C++ Redistributable latest supported downloads:
https://learn.microsoft.com/de-de/cpp/windows/latest-supported-vc-redist?view=msvc-170

## Requirements on Ubuntu/Linux

Runtime requirements:
```
apt-get install -q -y --no-install-recommends \
    libglu1-mesa-dev \
    libgomp1 \
    libopenslide-dev \
    python3 \
    python3-pip
```

### Known issues

The following Warnings can be ignored:
- "WARNING: In AutoLoadModulesFromPath at /opt/mitk/Modules/CppMicroServices/core/src/util/usUtils.cpp:176 : Auto-loading of module /usr/local/lib/python3.10/dist-packages/m2aia/binaries/bin/MitkCore/libMitk\<XY\>IO.so failed."

- "WARNING: In load_impl at /opt/mitk/Modules/CppMicroServices/core/src/util/usUtils.cpp:76 : libMitk\<XY\>.so: cannot open shared object file: No such file or directory"


## Examples

[https://github.com/m2aia/pym2aia-examples.git](https://github.com/m2aia/pym2aia-examples.git)


M²aia is an interactive desktop application for visualization and processing of mass spectrometry (MS) imaging data. Please visit [https://m2aia.github.io/m2aia](https://m2aia.github.io/m2aia) for further information.

# Cite M²aia

Cordes J; Enzlein T; Marsching C; Hinze M; Engelhardt S; Hopf C; Wolf I (July, 2021): M²aia - Interactive, fast and memory efficient analysis of 2D and 3D multi-modal mass spectrometry imaging data https://doi.org/10.1093/gigascience/giab049

[![DOI](https://zenodo.org/badge/554270311.svg)](https://zenodo.org/badge/latestdoi/554270311)
