FROM python:3.11.7 AS builder

ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

ARG POETRY_VERSION=1.6.1

COPY utils/ utils/
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

RUN pip install poetry==${POETRY_VERSION}
RUN poetry config virtualenvs.create false

RUN poetry install