M²aia is an interactive desktop application for visualization and processing of mass spectrometry (MS) imaging data. Please visit [https://m2aia.github.io/m2aia](https://m2aia.github.io/m2aia) for further information.

# Installation

Install pyM2aia in a [virtual environment](https://packaging.python.org/en/latest/tutorials/installing-packages/#creating-and-using-virtual-environments).

## Ubuntu/Linux

Runtime requirements:
```
apt-get install -q -y --no-install-recommends \
    libglu1-mesa-dev \
    libgomp1 \
    libopenslide-dev \
    python3 \
    python3-pip
```

```pip install https://github.com/m2aia/pym2aia/releases/download/v0.2.6/pym2aia-0.2.6-py3-none-linux_x86_64.whl```

### Known issues

The following Warnings can be ignored:
- "WARNING: In AutoLoadModulesFromPath at /opt/mitk/Modules/CppMicroServices/core/src/util/usUtils.cpp:176 : Auto-loading of module /usr/local/lib/python3.10/dist-packages/m2aia/binaries/bin/MitkCore/libMitk\<XY\>IO.so failed."

- "WARNING: In load_impl at /opt/mitk/Modules/CppMicroServices/core/src/util/usUtils.cpp:76 : libMitk\<XY\>.so: cannot open shared object file: No such file or directory"



## Windows
```pip install https://github.com/m2aia/pym2aia/releases/download/v0.2.6/pym2aia-0.2.6-py3-none-win_amd64.whl```

## Examples

[https://github.com/m2aia/pym2aia-examples.git](https://github.com/m2aia/pym2aia-examples.git)
