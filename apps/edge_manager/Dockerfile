FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app

ENV EDGE_REGISTRY_PATH=/var/lib/edge-manager/edges.json
VOLUME ["/var/lib/edge-manager"]

ENTRYPOINT ["python", "-m", "app"]
CMD ["serve"]
