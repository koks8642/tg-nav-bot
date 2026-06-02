FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY webapp ./webapp
COPY scripts ./scripts
# the export is needed for the one-time backfill; harmless to ship
COPY ChatExport/messages.html ./ChatExport/messages.html

# DB lives on a persistent volume mounted at /data (see deploy docs)
ENV DB_PATH=/data/rqm.db \
    HOST=0.0.0.0 \
    PORT=8080
VOLUME ["/data"]
EXPOSE 8080

CMD ["python", "-m", "app.main"]
