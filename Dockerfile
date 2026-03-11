FROM python:3.12-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN groupadd --system appgroup \
    && useradd --system --uid 10001 --gid appgroup --create-home appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appgroup /app \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

USER appuser

EXPOSE 9000

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
