
FROM python:3.11.7 AS builder

ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

ARG POETRY_VERSION=1.6.1

COPY utilities/ utilities/
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

RUN pip install poetry==${POETRY_VERSION}
RUN poetry config virtualenvs.create false

FROM builder AS training

ARG MODEL_NAME

WORKDIR /training

COPY ${MODEL_NAME}/training/ ./src/
RUN poetry install

FROM builder AS serving

ARG MODEL_NAME

COPY /serving/main.py serving/main.py
RUN poetry install --with serving

CMD exec uvicorn serving.main:app --host "0.0.0.0" --port "$AIP_HTTP_PORT"