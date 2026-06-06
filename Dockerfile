FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production image ships only the runtime package. The DB starts empty on a
# fresh volume and is populated live from channel posts (hashtags). The
# one-time backfill and operator scripts are run locally, not in the image.
COPY app ./app

# DB lives on a persistent volume mounted at /data (see deploy docs)
ENV DB_PATH=/data/rqm.db \
    HOST=0.0.0.0 \
    PORT=8080
VOLUME ["/data"]
EXPOSE 8080

CMD ["python", "-m", "app.main"]
