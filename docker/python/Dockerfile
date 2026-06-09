FROM python:3.11-slim

WORKDIR /app

COPY docker/python/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONPATH=/app

CMD ["bash"]
