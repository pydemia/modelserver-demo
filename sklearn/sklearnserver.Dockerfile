FROM python:3.7-slim
LABEL maintainer="Youngju Kim yj.kim1@sk.com, Juyoung Jung jyjung16@sk.com"

# RUN apt-get update && \
#     apt-get install libgomp1 && \
#     rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY sklearnserver sklearnserver
COPY run_modelserver /usr/local/bin/run_modelserver

RUN pip install --upgrade pip && \
    pip install -e ./sklearnserver && \
    pip install -r ./sklearnserver/requirements.txt --use-feature=2020-resolver

ENTRYPOINT ["./run_modelserver"]
