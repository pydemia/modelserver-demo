FROM python:3.7-slim
LABEL maintainer="pydemia@gmail.com"

# RUN apt-get update && \
#     apt-get install libgomp1 && \
#     rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN mkdir -p /workdir
WORKDIR /workdir

COPY sklearnserver sklearnserver
COPY setup.py setup.py
COPY requirements.txt requirements.txt
COPY README.md README.md
COPY run_sklearnserver /usr/local/bin/run_sklearnserver

RUN pip install --upgrade pip && \
    pip install -e . && \
    pip install -r ./requirements.txt

RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ENTRYPOINT ["./run_sklearnserver"]
