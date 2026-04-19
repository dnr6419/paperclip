FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backtesting/ backtesting/
COPY dashboard/ dashboard/
COPY strategies/ strategies/
COPY migrations/ migrations/

EXPOSE 8000

CMD ["uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "8000"]
