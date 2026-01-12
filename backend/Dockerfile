FROM python:3.12-slim


WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements_backend.txt .
RUN pip install -r requirements_backend.txt

COPY app ./app

EXPOSE 8000

CMD ["sh", "-c", "python -m app.seed_db && python -m app.ml.train && uvicorn app.main:app --host 0.0.0.0 --port 8000"]