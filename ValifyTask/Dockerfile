FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./requirements.txt /app/requirements.txt

RUN pip install --default-timeout=10000 future --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app/app
