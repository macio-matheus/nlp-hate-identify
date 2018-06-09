FROM python:3.6.4

RUN mkdir /app
WORKDIR /app

# INSTALL DEPENDENCIES
COPY requirements.py /app
RUN pip install -r requirements.py

EXPOSE 6543
