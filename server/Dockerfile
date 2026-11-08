FROM python:3.11-slim
WORKDIR /app
COPY server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt
COPY ml/pipeline.py ml/pipeline.py
COPY web/public/models/damage_unet.onnx web/public/models/damage_unet.onnx
COPY web/public/grading_rules.json web/public/grading_rules.json
COPY server/app.py server/app.py
ENV INSPECT_DB=/data/inspect.db SOURCE_DIR=/source
VOLUME ["/data", "/source"]
EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
