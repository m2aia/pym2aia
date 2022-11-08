FROM m2aia/m2aia:latest-build as M2aiaBinaries

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get update && apt-get install -y -q --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git

RUN pip install tomli pyparsing packaging iniconfig exceptiongroup attrs pluggy pytest SimpleITK numpy

# m2aia/m2aia:latest-build containes the latest M2aia installer and required dependencies
RUN mkdir /opt/m2aia
# we extract all files to this location
RUN tar -xvf /opt/packages/m2aia.tar.gz -C /opt/m2aia --strip-components=1


# promote the required library path
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/m2aia/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/m2aia/bin/MitkCore

RUN mkdir -p /opt/pym2aia

# copy current pyM2aia sources
COPY . /opt/pym2aia/
RUN pip install -e /opt/pym2aia

# run pytest
WORKDIR /opt/pym2aia