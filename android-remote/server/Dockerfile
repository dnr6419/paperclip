FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY relay.py .

EXPOSE 8765
CMD ["uvicorn", "relay:app", "--host", "0.0.0.0", "--port", "8765", "--ws-max-size", "16777216"]
