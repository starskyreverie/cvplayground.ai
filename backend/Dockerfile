FROM python:3.10-slim

WORKDIR /app

COPY ./app /app/app
COPY requirements.txt /app

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

CMD ["run.sh"]
