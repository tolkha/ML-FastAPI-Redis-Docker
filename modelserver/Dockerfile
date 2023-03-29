FROM python:3.7-slim-buster

COPY requirements.txt /app/

RUN pip install -r /app/requirements.txt

COPY Weights1.h5 /app/

COPY model.json /app/

COPY . /app

CMD ["python", "/app/main.py"]
