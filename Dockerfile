FROM python:3.12-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

ARG ENGINE_ORT_RUNTIME=cpu
ARG ENGINE_ORT_VERSION=1.22.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$ENGINE_ORT_RUNTIME" = "gpu" ]; then \
        pip install --no-cache-dir \
          onnxruntime-gpu==${ENGINE_ORT_VERSION} \
          nvidia-cublas-cu12 \
          nvidia-cuda-nvrtc-cu12 \
          nvidia-cuda-runtime-cu12 \
          nvidia-cudnn-cu12 \
          nvidia-cufft-cu12 \
          nvidia-curand-cu12 \
          nvidia-cusolver-cu12 \
          nvidia-cusparse-cu12 \
          nvidia-nvjitlink-cu12; \
    elif [ "$ENGINE_ORT_RUNTIME" = "cpu" ]; then \
        pip install --no-cache-dir onnxruntime==${ENGINE_ORT_VERSION}; \
    else \
        echo "Unsupported ENGINE_ORT_RUNTIME='$ENGINE_ORT_RUNTIME' (expected cpu|gpu)" && exit 1; \
    fi

# CUDA shared libs provided by pip `nvidia-*` packages (used when ENGINE_ORT_RUNTIME=gpu).
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/site-packages/nvidia/cufft/lib:/usr/local/lib/python3.12/site-packages/nvidia/curand/lib:/usr/local/lib/python3.12/site-packages/nvidia/cusolver/lib:/usr/local/lib/python3.12/site-packages/nvidia/cusparse/lib:/usr/local/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH}

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
