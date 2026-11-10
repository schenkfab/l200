FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY blog_writer/ ./blog_writer/
COPY eval/ ./eval/

EXPOSE 8080

CMD ["adk", "api_server", "--host", "0.0.0.0", "--port", "8080"]
