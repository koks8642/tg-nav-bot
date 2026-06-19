FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN addgroup --system app && adduser --system --ingroup app app \
    && mkdir -p /data && chown -R app:app /app /data

# Production image ships the runtime package. The DB starts empty on a fresh
# volume and is populated live from channel posts (hashtags). Operator scripts
# stay outside the image; smoke checks live in app/ so Docker can run them.
COPY --chown=app:app app ./app
# Persona cards + trigger lexicon for the AI group chat (read-only data)
COPY --chown=app:app personas ./personas
# Chapter index for the background KB builder (number → Telegraph path)
COPY --chown=app:app data ./data

# DB lives on a persistent volume mounted at /data (see deploy docs)
ENV DB_PATH=/data/rqm.db \
    HOST=0.0.0.0 \
    PORT=8080
VOLUME ["/data"]
EXPOSE 8080

CMD ["python", "-m", "app.entrypoint"]
