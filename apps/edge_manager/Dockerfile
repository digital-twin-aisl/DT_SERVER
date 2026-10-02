# Control service image. A deployment must supply its CUDA worker environment
# inside this container (DT_SERVER_PYTHON); native services are the default path.
FROM python:3.11-slim
WORKDIR /workspace
COPY apps/edge_manager/requirements.txt /tmp/manager-requirements.txt
RUN pip install --no-cache-dir -r /tmp/manager-requirements.txt
COPY packages/dt_common /workspace/packages/dt_common
RUN pip install --no-cache-dir --no-deps /workspace/packages/dt_common
COPY apps /workspace/apps
COPY Ground.usd /workspace/Ground.usd
ENV EDGE_REGISTRY_PATH=/var/lib/dt/edges.json DT_DATA_DIR=/var/lib/dt/runtime
VOLUME ["/var/lib/dt"]
ENTRYPOINT ["python", "-m", "apps.edge_manager"]
CMD ["serve"]
