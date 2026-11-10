FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY . .
RUN pip install --no-cache-dir .

FROM python:3.13-slim

# TRADINGAGENTS_IN_DOCKER lets an update notice name the rebuild command.
# TZ pins the container clock to ET: the worker dates its daily log and the
# close-of-day summary by local time, and in UTC anything after 8 PM ET is
# already tomorrow.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRADINGAGENTS_IN_DOCKER=1 \
    TZ=America/New_York

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN useradd --create-home appuser \
 && install -d -m 0755 -o appuser -g appuser /home/appuser/.tradingagents
USER appuser
WORKDIR /home/appuser/app

COPY --from=builder --chown=appuser:appuser /build .

# Build stamp (see tradingagents/build.py). deploy.sh passes both; declared last
# on purpose: they change on every build, and anything below an ARG/ENV is
# rebuilt, so keeping them here leaves the install layers cached. Left empty
# for an unstamped build (plain `docker compose up`), which build.py then
# reports as 1.dev rather than inventing a build number.
ARG BUILD_VERSION=""
ARG GIT_SHA=""
ENV EDINBURGH_BUILD=$BUILD_VERSION
ENV EDINBURGH_GIT_SHA=$GIT_SHA

ENTRYPOINT ["tradingagents"]
