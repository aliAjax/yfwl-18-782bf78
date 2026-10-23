FROM python:3.13.17-slim-bookworm@sha256:e32be0e55814a08977501ca863271d9bd02443710db8f2afaf8372e2c4508530

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends tmux asciinema python3-pytest \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app /app

RUN groupadd --gid 10001 appusers \
    && useradd --create-home --uid 10001 --gid appusers operator \
    && mkdir -p /logs/verifier \
    && chown -R operator:operator /app /logs/verifier

USER operator
CMD ["sleep", "infinity"]
