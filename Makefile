# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= .venv/bin/sphinx-build
PYTHON   ?= .venv/bin/python
PYTEST   ?= .venv/bin/pytest
SOURCEDIR     = docs
BUILDDIR      = sphinx-build
PROJ = m2aia


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)  -Q

html:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) 
	# @google-chrome $(BUILDDIR)/html/m2aia.html
    

wheel:
	rm -f dist/*
	$(PYTHON) setup_prepare.py --linux --download -v v2023.10.6
	$(PYTHON) -m build
	mv dist/m2aia-0.5.2-py3-none-any.whl dist/m2aia-0.5.2-py3-none-manylinux_2_31_x86_64.whl
	$(PYTHON) setup_prepare.py --windows --download -v v2023.10.8
	$(PYTHON) -m build
	mv dist/m2aia-0.5.2-py3-none-any.whl dist/m2aia-0.5.2-py3-none-win-amd64.whl
	$(PYTHON) setup_prepare.py --windows --linux -v v2023.10.6
	$(PYTHON) -m build
	rm dist/m2aia-0.5.2-py3-none-any.whl
	$(PYTEST)
