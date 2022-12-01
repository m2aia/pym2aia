FROM ubuntu:focal

RUN apt-get update 
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -q -y --no-install-recommends \
    libglu1-mesa-dev \
    libgomp1 \
    libopenslide-dev \
    python3 \
    python3-pip \
    git \
    wget

RUN pip install SimpleITK numpy

# m2aia/m2aia:latest-build containes the latest M2aia installer and required dependencies
RUN mkdir -p /opt/m2aia /opt/packages

RUN wget https://data.jtfc.de/latest/ubuntu20_04/M2aia-latest.tar.gz -nv
RUN mv M2aia-latest.tar.gz /opt/packages/m2aia.tar.gz 

# we extract all files to this location
RUN tar -xvf /opt/packages/m2aia.tar.gz -C /opt/m2aia --strip-components=1


# promote the required library path
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/m2aia/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/m2aia/bin/MitkCore

RUN pip install  git+https://github.com/m2aia/pym2aia.git
