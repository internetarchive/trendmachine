#!/usr/bin/env -S docker image build -t trendmachine . -f

FROM     python:3

ENV      STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN      adduser --disabled-password --gecos "" appuser

WORKDIR  /app

RUN      pip install --no-cache-dir \
           streamlit \
           requests

COPY     . ./

COPY     --chown=appuser:appuser . ./

USER     appuser

CMD      ["streamlit", "run", "main.py"]
