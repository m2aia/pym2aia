FROM m2aia/m2aia:latest-build as M2aiaBinaries

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get update && apt-get install -y -q --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git

RUN pip install tomli pyparsing packaging iniconfig exceptiongroup attrs pluggy pytest SimpleITK numpy

RUN mkdir /opt/m2aia
RUN tar -xvf /opt/packages/m2aia.tar.gz -C /opt/m2aia --strip-components=1


ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/m2aia/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/m2aia/bin/MitkCore

RUN echo $LD_LIBRARY_PATH

RUN mkdir -p /opt/pym2aia

COPY . /opt/pym2aia/
RUN ls /opt/pym2aia
RUN ls /opt/m2aia/bin/MitkCore

RUN pip install -e /opt/pym2aia 
WORKDIR /opt/pym2aia
RUN pytest
