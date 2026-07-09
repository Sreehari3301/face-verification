FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
# Server only needs the crypto + web stack, not the ML stack --
# keeping FaceNet/torch out of this image is itself a mild defense-in-depth
# measure (smaller attack surface, no accidental temptation to run inference
# server-side on decrypted data).
RUN pip install --no-cache-dir phe fastapi uvicorn pydantic numpy
COPY edge/paillier_protocol.py ./edge/paillier_protocol.py
COPY edge/__init__.py ./edge/__init__.py
COPY server/app.py ./server/app.py
COPY server/__init__.py ./server/__init__.py
EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
