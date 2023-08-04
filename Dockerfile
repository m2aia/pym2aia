FROM ubuntu:22.10

RUN apt-get update 
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -q -y --no-install-recommends \
    libglu1-mesa-dev \
    libgomp1 \
    libopenslide-dev \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget
RUN python3 -m venv /.venv

ENV VIRTUAL_ENV="/.venv"
RUN export VIRTUAL_ENV

ENV _OLD_VIRTUAL_PATH="$PATH"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN export PATH

RUN pip install SimpleITK numpy
