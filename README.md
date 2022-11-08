M²aia is an interactive desktop application for visualization and processing of mass spectrometry (MS) imaging data. Please visit [https://m2aia.github.io/m2aia](https://m2aia.github.io/m2aia) for further information.

## Installation

### Windows

Download the latest version of [M²aia (Windows)](https://m2aia.de/)

It is required to promote a system variable pointing to:

``` C:\Users\User\Documents\M2aiaWorkbench 2022.10.00\bin\MitkCore ```


If working in a virtual environment is required:

1) Create virtual environment:

``` python3.exe -m venv .venv ```

(!) install python if required by typing python3.exe without arguments

2) Activate virtual environment

``` .\.venv\Scripts\activate ```

(!) If it is not possible to execute, check the following: [ExecutionPolicy](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.2)

For example you can set the policy to ``` Set-ExecutionPolicy -ExecutionPolicy AllSigned -Scope CurrentUser ```

3) Install pyM2aia

``` pip install git+https://github.com/m2aia/pym2aia.git ```



## Examples

[https://github.com/m2aia/pym2aia-examples.git](https://github.com/m2aia/pym2aia-examples.git)
