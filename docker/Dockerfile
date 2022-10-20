FROM m2aia/m2aia:latest-build as M2aiaBinaries

RUN mkdir /opt/m2aia

RUN tar -xvf /opt/packages/m2aia.tar.gz -C /opt/m2aia --strip-components=1

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get update && apt-get install -y -q --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git

RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/m2aia/bin
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/m2aia/bin/MitkCore

RUN git clone https://github.com/m2aia/pym2aia /opt/pym2aia
RUN pip install -e /opt/pym2aia pytest
WORKDIR /opt/pym2aia
RUN pytest