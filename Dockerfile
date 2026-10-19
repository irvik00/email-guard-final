# CPU-only image. Base digest resolved through the official Docker Registry API.
FROM --platform=linux/amd64 python:3.12.4-slim-bookworm@sha256:a3e58f9399353be051735f09be0316bfdeab571a5c6a24fd78b92df85bcb2d85
ENV PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    EMAIL_GUARD_DISABLE_LAYA=1 EMAIL_GUARD_ARTIFACT_ROOT=/artifacts EMAIL_GUARD_HOST=0.0.0.0
WORKDIR /opt/email_guard
COPY requirements-docker.lock.txt /tmp/requirements-docker.lock.txt
COPY deployment/wheelhouse-linux/ /tmp/wheels/
RUN python -m pip install --no-cache-dir --no-index --find-links=/tmp/wheels --no-deps --require-hashes -r /tmp/requirements-docker.lock.txt \
    && python -m pip check \
    && rm -rf /tmp/wheels /tmp/requirements-docker.lock.txt
COPY app/*.py app/release_manifest.json ./app/
COPY app/static/ ./app/static/
COPY artifacts/models/day3/ /artifacts/models/day3/
COPY src/read_email.py src/normalization_schema.py src/day5_common.py src/day5_laya_runtime.py src/decision_protocol.py src/final_eval_common.py src/predict_email.py src/baseline_common.py ./src/
COPY config/day4_experiments.json ./config/
COPY reports/final_evaluation/runs/20261017T132144Z-d663811e/FINAL_EVALUATION_PROTOCOL.json ./reports/final_evaluation/runs/20261017T132144Z-d663811e/
RUN chmod -R a+rX /opt/email_guard /artifacts
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/ready', timeout=3).read()"]
CMD ["python", "-m", "app.run", "--disable-laya"]
