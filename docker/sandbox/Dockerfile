FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_INDEX=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# No pip install: dynamic tools run pure Python only.
RUN groupadd --gid 10001 sandbox \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin sandbox

WORKDIR /code
COPY runner.py /sandbox/runner.py
RUN chmod 0555 /sandbox/runner.py

USER 10001:10001

# Default command
CMD ["python", "/sandbox/runner.py"]
