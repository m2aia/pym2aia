M²aia is an interactive desktop application for visualization and processing of mass spectrometry (MS) imaging data. Please visit [https://m2aia.github.io/m2aia](https://m2aia.github.io/m2aia) for further information.

# Installation

## Ubuntu/Linux
Download the latest version of [M²aia](https://m2aia.de/).

```
sudo apt-get install -q -y --no-install-recommends \
  libglu1-mesa-dev \
  libgomp1 \
  libopenslide-dev \
  python3 \
  python3-pip \
  git \
  wget

cd /home/$USER
wget https://data.jtfc.de/latest/ubuntu20_04/M2aia-latest.tar.gz -nv
mkdir -p /home/$USER/m2aia
tar -xvf /home/$USER/M2aia-latest.tar.gz -C m2aia --strip-components=1

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USER/m2aia/bin
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/$USER/m2aia/bin/MitkCore

python3 -m venv .pym2aia
source .py2maia/bin/activate
pip install git+https://github.com/m2aia/pym2aia.git
```
 

Check the [Dockerfile](Dockerfile) for installation hints.



## Windows
Download the latest version of [M²aia](https://m2aia.de/).

It is required to promote a system variable pointing to:<br>
``` C:\Users\User\Documents\M2aiaWorkbench 2022.10.00\bin ``` <br>
``` C:\Users\User\Documents\M2aiaWorkbench 2022.10.00\bin\MitkCore ```


If working in a virtual environment is required:<br>
1) Create virtual environment:<br>
``` python3.exe -m venv .venv ```

(!) install python if required by typing python3.exe without arguments

2) Activate virtual environment<br>
``` .\.venv\Scripts\activate ```

(!) If it is not possible to execute, check the following: [ExecutionPolicy](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.2)

For example you can set the policy to ``` Set-ExecutionPolicy -ExecutionPolicy AllSigned -Scope CurrentUser ```

3) Install pyM2aia<br>
``` pip install git+https://github.com/m2aia/pym2aia.git ```



## Examples

[https://github.com/m2aia/pym2aia-examples.git](https://github.com/m2aia/pym2aia-examples.git)
